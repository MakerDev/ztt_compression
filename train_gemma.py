import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import argparse
import yaml

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime

# Import our GemmaComp implementation
from models.gemma.GemmaComp import GemmaCompForCausalLM
from mmlu_debugger import analyze_lora_parameters
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

def collate_fn_mmlu(batch):
    return {
        'question': [item['question'] for item in batch],
        'choices': [item['choices'] for item in batch],
        'answer': torch.tensor([item['answer'] for item in batch]),
        'subject': [item['subject'] for item in batch]
    }

def collate_fn_sciq(batch):
    return {
        'question': [item['question'] for item in batch],
        'distractor1': [item['distractor1'] for item in batch],
        'distractor2': [item['distractor2'] for item in batch],
        'distractor3': [item['distractor3'] for item in batch],
        'correct_answer': [item['correct_answer'] for item in batch],
        'support': [item.get('support', '') for item in batch]
    }

def collate_fn_code(batch):
    return {
        'java': [item['java'] for item in batch],
        'cs': [item['cs'] for item in batch]
    }

def parse_args():
    # 1) 초기 파서: config 파일 경로와 preset 이름만
    init_parser = argparse.ArgumentParser(add_help=False)
    init_parser.add_argument(
        "--config_file", "-c",
        type=str, default="/data/ztt_compression/configs.yaml",
        help="Path to YAML config"
    )
    init_parser.add_argument(
        "--preset", "-p",
        type=str, default="gemma_1b_cycle5",
        help="Which preset to use from YAML"
    )
    init_args, remaining_argv = init_parser.parse_known_args()

    # 2) YAML 로드 & preset 병합
    with open(init_args.config_file, "r") as f:
        yaml_cfg = yaml.safe_load(f)
        
    preset_cfg  = yaml_cfg.get(init_args.preset, {})
    cfg = {**preset_cfg}  # base 위에 preset 덮어쓰기

    parser = argparse.ArgumentParser(
        parents=[init_parser],
        description="Fine-tune GemmaComp on multiple datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ────────── 데이터셋 선택 ──────────
    parser.add_argument("--dataset", type=str, 
                        choices=["mmlu", "sciq", "code"],
                        default="mmlu",
                        help="Dataset to use for training and evaluation")

    # ────────── 모델 ──────────
    parser.add_argument("--model_name",   type=str, default=cfg["model_name"])
    parser.add_argument("--output_dir",   type=str, default=cfg["output_dir"])

    # ────────── 프루닝 ──────────
    parser.add_argument("--pruned_layers", type=int, nargs="+",
                        default=cfg.get("pruned_layers"))

    # ────────── 사이클링 ──────────
    parser.add_argument("--cycle_layers", type=int, nargs="+",
                        default=cfg.get("cycle_layers"))
    parser.add_argument("--cycle_count",  type=int,
                        default=cfg.get("cycle_count"))

    # ────────── LoRA ──────────
    parser.add_argument("--use_lora",      action="store_true" if cfg["use_lora"] else "store_false",
                        default=cfg["use_lora"])
    parser.add_argument("--lora_all",      action="store_true" if cfg["lora_all"] else "store_false",
                        default=cfg["lora_all"])
    parser.add_argument("--lora_rank",     type=int,   default=cfg["lora_rank"])
    parser.add_argument("--lora_alpha",    type=float, default=cfg["lora_alpha"])
    parser.add_argument("--lora_dropout",  type=float, default=cfg["lora_dropout"])
    parser.add_argument("--train_all",     action="store_true" if cfg["train_all"] else "store_false",
                        default=cfg["train_all"])

    # ────────── Distillation ──────────
    parser.add_argument("--use_distillation",
                        action="store_true" if cfg["use_distillation"] else "store_false",
                        default=cfg["use_distillation"])
    parser.add_argument("--distillation_temperature", type=float,
                        default=cfg["distillation_temperature"])
    parser.add_argument("--distillation_alpha", type=float,
                        default=cfg["distillation_alpha"])

    # ────────── 학습 ──────────
    parser.add_argument("--batch_size",                  type=int,
                        default=cfg["batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=cfg["gradient_accumulation_steps"])
    parser.add_argument("--learning_rate",               type=float,
                        default=cfg["learning_rate"])
    parser.add_argument("--num_epochs",                  type=float,
                        default=cfg["num_epochs"])
    parser.add_argument("--max_seq_length",              type=int,
                        default=cfg["max_seq_length"])
    parser.add_argument("--warmup_steps",                type=int,
                        default=cfg["warmup_steps"])
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    parser.add_argument("--skip_eval", action="store_true")

    return parser.parse_args()


class GemmaCompTrainer(SFTTrainer):
    """Custom trainer for GemmaComp with distillation support"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        outputs = model(**inputs, labels=labels)
        
        # If num_items_in_batch is provided, we could use it to normalize the loss
        # but for now we'll just accept it for compatibility
        
        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss


def format_mmlu_prompt(question, choices, answer=None):
    """Format MMLU question into a prompt"""
    prompt = f"Question: {question}\n\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    if answer is not None:
        prompt += f"\nAnswer: {chr(65+answer)}"
    else:
        prompt += "\nAnswer:"
    
    return prompt


def format_sciq_prompt(question, correct_answer, distractors, support=None, include_answer=True):
    """Format SciQ question into a prompt"""
    import random
    
    # Create choices list with correct answer and distractors
    choices = [correct_answer] + distractors
    random.shuffle(choices)
    correct_idx = choices.index(correct_answer)
    
    prompt = f"Question: {question}\n\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    if support and support.strip():
        prompt += f"\nContext: {support}\n"
    
    if include_answer:
        prompt += f"\nAnswer: {chr(65+correct_idx)}"
    else:
        prompt += "\nAnswer:"
    
    return prompt, correct_idx


def format_code_prompt(java_code, cs_code=None):
    """Format code translation prompt"""
    prompt = "Translate the following Java code to C#:\n\n"
    prompt += "```java\n"
    prompt += java_code
    prompt += "\n```\n\n"
    prompt += "C# translation:\n"
    
    if cs_code is not None:
        prompt += "```csharp\n"
        prompt += cs_code
        prompt += "\n```"
    
    return prompt


def prepare_mmlu_dataset(tokenizer, max_seq_length):
    """Load and prepare MMLU dataset for training"""
    print("Loading MMLU auxiliary train dataset...")
    dataset = load_dataset("kz919/mmlu-auxiliary-train-auto-labelled", split="train")
    
    def formatting_prompts_func(examples):
        texts = []
        
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            
            # Handle choices format
            if 'choices' in examples:
                choices = examples['choices'][i]
            else:
                choices = [
                    examples.get('choice_a', examples.get('A', [""]))[i],
                    examples.get('choice_b', examples.get('B', [""]))[i],
                    examples.get('choice_c', examples.get('C', [""]))[i],
                    examples.get('choice_d', examples.get('D', [""]))[i]
                ]
            
            # Convert answer to index if needed
            answer = examples['answer'][i]
            if isinstance(answer, str):
                answer = ord(answer.upper()) - 65  # A->0, B->1, etc.
            
            text = format_mmlu_prompt(question, choices, answer)
            texts.append(text)
        
        return {"text": texts}
    
    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def prepare_sciq_dataset(tokenizer, max_seq_length):
    """Load and prepare SciQ dataset for training"""
    print("Loading SciQ train dataset...")
    dataset = load_dataset("allenai/sciq", split="train")
    
    def formatting_prompts_func(examples):
        texts = []
        
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            correct_answer = examples['correct_answer'][i]
            distractors = [
                examples['distractor1'][i],
                examples['distractor2'][i],
                examples['distractor3'][i]
            ]
            support = examples.get('support', [''])[i] if 'support' in examples else ''
            
            text, _ = format_sciq_prompt(question, correct_answer, distractors, support, include_answer=True)
            texts.append(text)
        
        return {"text": texts}
    
    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def prepare_code_dataset(tokenizer, max_seq_length):
    """Load and prepare CodeXGLUE dataset for training"""
    print("Loading CodeXGLUE Java-C# translation dataset...")
    dataset = load_dataset("code_x_glue_cc_code_to_code_trans", split="train")
    
    def formatting_prompts_func(examples):
        texts = []
        
        for i in range(len(examples['java'])):
            java_code = examples['java'][i]
            cs_code = examples['cs'][i]
            
            text = format_code_prompt(java_code, cs_code)
            texts.append(text)
        
        return {"text": texts}
    
    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def evaluate_mmlu_accuracy(model, tokenizer, dataloader, device="cuda"):
    """Evaluate model accuracy on MMLU test set"""
    model.eval()
    
    # Token IDs for answer choices
    choice_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"]
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating MMLU", dynamic_ncols=True):
            for q, choices, ans in zip(batch['question'], batch['choices'], batch['answer']):
                # Format prompt without answer
                prompt = format_mmlu_prompt(q, choices)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                                 max_length=dataloader.dataset.max_seq_length).to(device)
                
                # Generate prediction
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]  # Last token logits
                
                # Get logits for answer choices only
                choice_logits = logits[choice_tokens]
                pred_idx = torch.argmax(choice_logits).item()
                
                if pred_idx == ans.item():
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, total


def evaluate_sciq_accuracy(model, tokenizer, dataloader, device="cuda"):
    """Evaluate model accuracy on SciQ test set"""
    model.eval()
    
    # Token IDs for answer choices
    choice_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"]
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating SciQ", dynamic_ncols=True):
            for i in range(len(batch['question'])):
                question = batch['question'][i]
                correct_answer = batch['correct_answer'][i]
                distractors = [
                    batch['distractor1'][i],
                    batch['distractor2'][i],
                    batch['distractor3'][i]
                ]
                support = batch['support'][i]
                
                # Format prompt without answer
                prompt, correct_idx = format_sciq_prompt(question, correct_answer, distractors, support, include_answer=False)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                                 max_length=dataloader.dataset.max_seq_length).to(device)
                
                # Generate prediction
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]  # Last token logits
                
                # Get logits for answer choices only
                choice_logits = logits[choice_tokens]
                pred_idx = torch.argmax(choice_logits).item()
                
                if pred_idx == correct_idx:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, total


def evaluate_code_bleu(model, tokenizer, dataloader, device="cuda", max_length=512):
    """Evaluate model BLEU score on code translation test set"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    model.eval()
    
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Code Translation", dynamic_ncols=True):
            for java_code, cs_code in zip(batch['java'], batch['cs']):
                # Format prompt without answer
                prompt = format_code_prompt(java_code)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                                 max_length=max_length).to(device)
                
                # Generate translation
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    # temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Extract generated code
                generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                # Extract code between ```csharp and ```
                if "```csharp" in generated:
                    start = generated.find("```csharp") + len("```csharp")
                    end = generated.find("```", start)
                    if end != -1:
                        generated = generated[start:end].strip()
                elif "```" in generated:
                    start = generated.find("```") + 3
                    end = generated.find("```", start)
                    if end != -1:
                        generated = generated[start:end].strip()
                
                # Calculate BLEU score
                reference = cs_code.split()
                hypothesis = generated.split()
                
                if hypothesis:
                    score = sentence_bleu([reference], hypothesis, smoothing_function=smoothie)
                    bleu_scores.append(score)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    return avg_bleu, len(bleu_scores)


def main():
    args = parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with compression configuration
    print(f"Loading GemmaComp model from {args.model_name}...")
    model = GemmaCompForCausalLM.from_pretrained(
        args.model_name,
        pruned_layers=args.pruned_layers,
        cycle_layers=args.cycle_layers,
        cycle_count=args.cycle_count,
        use_lora=args.use_lora,
        lora_all=args.lora_all,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_distillation=args.use_distillation,
        distillation_temperature=args.distillation_temperature,
        distillation_alpha=args.distillation_alpha,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="cuda"
    )
    
    # Print model configuration
    print("\nModel Configuration:")
    print(f"- Model: {args.model_name}")
    print(f"- Dataset: {args.dataset.upper()}")
    print(f"- Original layers: {model.config.num_hidden_layers}")
    print(f"- Pruned layers: {args.pruned_layers}")
    print(f"- Cycle layers: {args.cycle_layers}")
    print(f"- Cycle count: {args.cycle_count}")
    print(f"- Execution sequence length: {len(model.model.execution_sequence)}")
    print(f"- Using LoRA: {args.use_lora}")
    print(f"- LoRA Rank: {args.lora_rank}")
    print(f"- LoRA Alpha: {args.lora_alpha}")
    print(f"- LoRA to ALL: {args.lora_all}")
    print(f"- Using Distillation: {args.use_distillation}")
    print(f"- Train All: {args.train_all}")

    # Prepare for training - only train LoRA parameters if enabled
    if args.use_lora:
        print("\nSetting up LoRA training...")
        # Freeze all base parameters
        for param in model.parameters():
            param.requires_grad = args.train_all

        # Unfreeze LoRA parameters
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                
    model.config.use_cache = False
    model.generation_config.use_cache = False
    model.generation_config.cache_implementation = None

    analyze_lora_parameters(model)

    # Prepare dataset based on selection
    if args.dataset == "mmlu":
        train_dataset = prepare_mmlu_dataset(tokenizer, args.max_seq_length)
        test_dataset = load_dataset("cais/mmlu", "all", split="test")
        collate_fn = collate_fn_mmlu
    elif args.dataset == "sciq":
        train_dataset = prepare_sciq_dataset(tokenizer, args.max_seq_length)
        test_dataset = load_dataset("allenai/sciq", split="test")
        collate_fn = collate_fn_sciq
    elif args.dataset == "code":
        train_dataset = prepare_code_dataset(tokenizer, args.max_seq_length)
        test_dataset = load_dataset("code_x_glue_cc_code_to_code_trans", split="test")
        collate_fn = collate_fn_code

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{args.dataset}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Save configuration
    with open(f"{output_dir}/config.txt", "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Pruned Layers: {args.pruned_layers}\n")
        f.write(f"Cycle Layers: {args.cycle_layers}\n")
        f.write(f"Cycle Count: {args.cycle_count}\n")
        f.write(f"Use LoRA: {args.use_lora}\n")
        f.write(f"LoRA Rank: {args.lora_rank}\n")
        f.write(f"LoRA Alpha: {args.lora_alpha}\n")
        f.write(f"LoRA Dropout: {args.lora_dropout}\n")
        f.write(f"Use Distillation: {args.use_distillation}\n")
        f.write(f"Distillation Temperature: {args.distillation_temperature}\n")
        f.write(f"Distillation Alpha: {args.distillation_alpha}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Number of Epochs: {args.num_epochs}\n")
        f.write(f"Max Sequence Length: {args.max_seq_length}\n")
        f.write(f"Warmup Steps: {args.warmup_steps}\n")

    estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)

    # Training configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="epoch",
        save_total_limit=2,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=8,
        packing=False,
        dataset_text_field="text",
        logging_steps=100,
        save_safetensors=True,
        report_to=["tensorboard"],
    )
    
    model.config.use_cache = False

    # Create trainer
    trainer = GemmaCompTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=2
    )
    test_dataloader.dataset.max_seq_length = args.max_seq_length
    
    # Evaluate before training
    print(f"\n=== Evaluating on {args.dataset.upper()} test set (before training) ===")
    
    if args.skip_eval:
        if args.dataset in ["mmlu", "sciq"]:
            metric_name = "Accuracy"
            metric_format = ".2%"
        else:
            metric_name = "BLEU Score"
            metric_format = ".4f"

        if args.dataset == "mmlu":
            n_samples = len(test_dataloader.dataset)
        elif args.dataset == "sciq":
            n_samples = len(test_dataloader.dataset)
        elif args.dataset == "code":
            n_samples = len(test_dataloader.dataset)
        initial_score = 0.0
        print(f"Skipping initial evaluation. Setting initial {args.dataset.upper()} {metric_name} to 0.0 (on {n_samples} samples)")
    else:
        if args.dataset == "mmlu":
            initial_score, n_samples = evaluate_mmlu_accuracy(
                model, tokenizer, test_dataloader, device=model.device
            )
            metric_name = "Accuracy"
            metric_format = ".2%"
        elif args.dataset == "sciq":
            initial_score, n_samples = evaluate_sciq_accuracy(
                model, tokenizer, test_dataloader, device=model.device
            )
            metric_name = "Accuracy"
            metric_format = ".2%"
        elif args.dataset == "code":
            initial_score, n_samples = evaluate_code_bleu(
                model, tokenizer, test_dataloader, device=model.device
            )
            metric_name = "BLEU Score"
            metric_format = ".4f"
    
    print(f"Initial {args.dataset.upper()} {metric_name}: {initial_score:{metric_format}} (on {n_samples} samples)")
    
    # Train model
    print("\n=== Starting training ===")
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {output_dir}/final_model")
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    # Evaluate after training
    print(f"\n=== Evaluating on {args.dataset.upper()} test set (after training) ===")
    
    if args.dataset == "mmlu":
        final_score, _ = evaluate_mmlu_accuracy(
            model, tokenizer, test_dataloader, device=model.device
        )
    elif args.dataset == "sciq":
        final_score, _ = evaluate_sciq_accuracy(
            model, tokenizer, test_dataloader, device=model.device
        )
    elif args.dataset == "code":
        final_score, _ = evaluate_code_bleu(
            model, tokenizer, test_dataloader, device=model.device
        )
    
    print(f"Final {args.dataset.upper()} {metric_name}: {final_score:{metric_format}} (on {n_samples} samples)")
    
    if args.dataset in ["mmlu", "sciq"]:
        print(f"Improvement: {(final_score - initial_score)*100:.2f} percentage points")
    else:
        print(f"Improvement: {final_score - initial_score:.4f}")


if __name__ == "__main__":
    main()