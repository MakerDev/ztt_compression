import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import argparse
import json
import os

# Import improved components
from models.llama.LLaMaCompImp import (
    ImprovedLlamaCompForCausalLM,
    MMMLUSpecificConfig,
    create_mmlu_optimized_model,
    LayerImportanceAnalyzer
)
from mmlu_debugger import diagnose_mmlu_training_issues


class ImprovedMMLUTrainer(SFTTrainer):
    """Enhanced trainer for MMLU with better loss handling"""
    
    def __init__(self, *args, **kwargs):
        # Extract MMLU-specific args
        self.focus_on_answers = kwargs.pop('focus_on_answers', True)
        self.answer_loss_weight = kwargs.pop('answer_loss_weight', 5.0)
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation with answer focusing"""
        # Extract labels before passing to model
        labels = inputs.pop("labels") if "labels" in inputs else None
        
        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Enhanced loss for MMLU if focusing on answers
        if self.focus_on_answers and labels is not None:
            # Find answer positions (tokens after "Answer:")
            # This is a simplified version - you may need to adjust based on your tokenizer
            logits = outputs.logits
            
            # Create weight mask for answer tokens
            weight_mask = torch.ones_like(labels).float()
            
            # Simple heuristic: last few tokens are likely answers
            # Better approach would be to identify "Answer:" position
            batch_size, seq_len = labels.shape
            answer_start = int(seq_len * 0.8)  # Last 20% of sequence
            weight_mask[:, answer_start:] *= self.answer_loss_weight
            
            # Recompute weighted loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weight_mask[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Apply weights and average
            weighted_losses = token_losses * shift_weights.view(-1)
            loss = weighted_losses[shift_labels.view(-1) != -100].mean()
        
        return (loss, outputs) if return_outputs else loss


def create_mmlu_data_collator(tokenizer, focus_on_response=True):
    """Create a data collator that focuses on answer tokens"""
    if focus_on_response:
        # Find the answer template tokens
        response_template = "\nAnswer:"
        response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        
        return DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
            mlm=False
        )
    else:
        return None


def format_mmlu_for_training(examples, tokenizer, include_cot=False):
    """Format MMLU examples with optional chain-of-thought"""
    texts = []
    
    for i in range(len(examples['question'])):
        question = examples['question'][i]
        
        # Handle different choice formats
        if 'choices' in examples:
            choices = examples['choices'][i]
        else:
            choices = [
                examples.get('A', [""])[i],
                examples.get('B', [""])[i], 
                examples.get('C', [""])[i],
                examples.get('D', [""])[i]
            ]
        
        # Get answer
        answer = examples['answer'][i]
        if isinstance(answer, int):
            answer_letter = chr(65 + answer)
        else:
            answer_letter = answer
        
        # Format with optional CoT
        if include_cot:
            # Add reasoning before answer
            prompt = f"""Question: {question}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Let me think about this step by step:
First, I need to understand what the question is asking.
Then, I'll evaluate each option.
Based on my analysis, the correct answer is:

Answer: {answer_letter}"""
        else:
            # Standard format
            prompt = f"""Question: {question}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer: {answer_letter}"""
        
        texts.append(prompt)
    
    # Tokenize with proper truncation and padding
    model_inputs = tokenizer(
        texts,
        max_length=1536,
        truncation=True,
        padding=True,
        return_tensors=None
    )
    
    # Add labels (same as input_ids for causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs


def evaluate_with_multiple_metrics(model, tokenizer, dataloader, device="cuda"):
    """Comprehensive evaluation with multiple metrics"""
    model.eval()
    
    metrics = {
        'accuracy': 0,
        'top_3_accuracy': 0,
        'per_subject_accuracy': {},
        'answer_perplexity': [],
        'confidence_scores': [],
        'calibration_error': 0,
    }
    
    # Token IDs for answers
    answer_tokens = {}
    for letter in "ABCD":
        tokens = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if tokens:
            answer_tokens[letter] = tokens[0]
    
    answer_token_ids = list(answer_tokens.values())
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for q, choices, true_ans, subject in zip(
                batch['question'], batch['choices'], 
                batch['answer'], batch['subject']
            ):
                # Format prompt
                prompt = f"""Question: {q}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer:"""
                
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=1536
                ).to(device)
                
                # Get model predictions
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1]
                
                # Extract answer logits
                answer_logits = last_logits[answer_token_ids]
                answer_probs = torch.softmax(answer_logits, dim=0)
                
                # Predictions
                pred_idx = torch.argmax(answer_logits).item()
                true_idx = true_ans.item()
                
                # Update metrics
                total_samples += 1
                
                # Accuracy
                if pred_idx == true_idx:
                    metrics['accuracy'] += 1
                
                # Top-3 accuracy
                top_3 = torch.topk(answer_logits, k=3).indices
                if true_idx in top_3:
                    metrics['top_3_accuracy'] += 1
                
                # Per-subject accuracy
                if subject not in metrics['per_subject_accuracy']:
                    metrics['per_subject_accuracy'][subject] = {'correct': 0, 'total': 0}
                
                metrics['per_subject_accuracy'][subject]['total'] += 1
                if pred_idx == true_idx:
                    metrics['per_subject_accuracy'][subject]['correct'] += 1
                
                # Confidence and calibration
                confidence = answer_probs[pred_idx].item()
                metrics['confidence_scores'].append(confidence)
                
                # Perplexity of answer
                true_logit = answer_logits[true_idx]
                perplexity = torch.exp(-true_logit).item()
                metrics['answer_perplexity'].append(perplexity)
    
    # Compute final metrics
    metrics['accuracy'] /= total_samples
    metrics['top_3_accuracy'] /= total_samples
    metrics['avg_confidence'] = np.mean(metrics['confidence_scores'])
    metrics['avg_perplexity'] = np.mean(metrics['answer_perplexity'])
    
    # Compute per-subject accuracies
    for subject, scores in metrics['per_subject_accuracy'].items():
        scores['accuracy'] = scores['correct'] / scores['total']
    
    # Expected Calibration Error (ECE)
    metrics['calibration_error'] = compute_ece(
        metrics['confidence_scores'],
        [1 if i < metrics['accuracy'] * total_samples else 0 
         for i in range(total_samples)]
    )
    
    return metrics


def compute_ece(confidences, accuracies, n_bins=10):
    """Compute Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = [(conf >= bin_lower) and (conf < bin_upper) 
                  for conf in confidences]
        prop_in_bin = sum(in_bin) / len(confidences)
        
        if prop_in_bin > 0:
            accuracy_in_bin = sum([a for a, ib in zip(accuracies, in_bin) if ib]) / sum(in_bin)
            avg_confidence_in_bin = sum([c for c, ib in zip(confidences, in_bin) if ib]) / sum(in_bin)
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece




def main():
    parser = argparse.ArgumentParser(description="Improved MMLU training for LlamaComp")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument("--output_dir", type=str, default="./mmlu_improved_results")
    
    # Compression arguments
    parser.add_argument("--prune_ratio", type=float, default=0.4)
    parser.add_argument("--use_importance_pruning", action="store_true")
    parser.add_argument("--cycle_middle_layers", action="store_true")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    # MMLU-specific arguments
    parser.add_argument("--focus_on_answers", action="store_true", default=True)
    parser.add_argument("--answer_loss_weight", type=float, default=5.0)
    parser.add_argument("--use_cot", action="store_true", help="Use chain-of-thought")
    parser.add_argument("--debug_mode", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Create optimized model
    print("Creating MMLU-optimized model...")
    model = create_mmlu_optimized_model(
        args.model_name,
        prune_ratio=args.prune_ratio,
        cycle_middle_layers=args.cycle_middle_layers,
        use_importance_analysis=args.use_importance_pruning,
        gradient_checkpointing=args.gradient_checkpointing
    )
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Prepare datasets
    print("Loading MMLU datasets...")
    train_dataset = load_dataset("kz919/mmlu-auxiliary-train-auto-labelled", split="train")
    eval_dataset = load_dataset("cais/mmlu", "all", split="test[:1000]")  # Subset for faster eval
    
    # Format datasets
    train_dataset = train_dataset.map(
        lambda x: format_mmlu_for_training(x, tokenizer, include_cot=args.use_cot),
        batched=True,
        num_proc=8,
        remove_columns=train_dataset.column_names
    )
    
    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        eval_strategy="epoch",
        eval_steps=None,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    # Create data collator
    data_collator = create_mmlu_data_collator(tokenizer, focus_on_response=args.focus_on_answers)
    
    # Create trainer
    trainer = ImprovedMMLUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset[:100],  # Use small subset for eval during training
        processing_class=tokenizer,
        data_collator=data_collator,
        focus_on_answers=args.focus_on_answers,
        answer_loss_weight=args.answer_loss_weight,
    )
    
    # Pre-training evaluation
    print("\n=== Pre-training Evaluation ===")
    
    # Create eval dataloader
    def collate_fn(batch):
        return {
            'question': [item['question'] for item in batch],
            'choices': [item['choices'] for item in batch],
            'answer': torch.tensor([item['answer'] for item in batch]),
            'subject': [item['subject'] for item in batch]
        }
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    initial_metrics = evaluate_with_multiple_metrics(model, tokenizer, eval_dataloader, device)
    print(f"Initial accuracy: {initial_metrics['accuracy']:.2%}")
    print(f"Initial top-3 accuracy: {initial_metrics['top_3_accuracy']:.2%}")
    
    # Debug mode
    if args.debug_mode:
        print("\n=== Running diagnostic analysis ===")
        diagnose_mmlu_training_issues(model, tokenizer, device)
    
    # Train
    print("\n=== Starting Training ===")
    trainer.train()
    
    # Post-training evaluation
    print("\n=== Post-training Evaluation ===")
    final_metrics = evaluate_with_multiple_metrics(model, tokenizer, eval_dataloader, device)
    
    # Print results
    print(f"\nFinal accuracy: {final_metrics['accuracy']:.2%}")
    print(f"Final top-3 accuracy: {final_metrics['top_3_accuracy']:.2%}")
    print(f"Improvement: {(final_metrics['accuracy'] - initial_metrics['accuracy'])*100:.2f} pp")
    
    # Save detailed results
    results = {
        'args': vars(args),
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else str(model.config)
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    print(f"\nSaving model to {args.output_dir}/final_model")
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()