import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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

# Import our LlamaComp implementation
from models.llama.LLaMaComp import LlamaCompForCausalLM
from mmlu_debugger import analyze_lora_parameters
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

def collate_fn(batch):
    return {
        'question': [item['question'] for item in batch],
        'choices': [item['choices'] for item in batch],
        'answer': torch.tensor([item['answer'] for item in batch]),
        'subject': [item['subject'] for item in batch]
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
        type=str, default="llama_1b_cycle5",
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
        description="Fine-tune LlamaComp on MMLU dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

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

    return parser.parse_args()


class LlamaCompTrainer(SFTTrainer):
    """Custom trainer for LlamaComp with distillation support"""
    
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


def prepare_mmlu_dataset(tokenizer, max_seq_length):
    """Load and prepare MMLU dataset for training"""
    print("Loading MMLU auxiliary train dataset...")
    dataset = load_dataset("kz919/mmlu-auxiliary-train-auto-labelled", split="train")
    # 20%만 사용
    # dataset = dataset.select(range(int(len(dataset) * 0.05)))
    
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


def evaluate_mmlu_accuracy(model, tokenizer, dataloader, device="cuda"):
    """Evaluate model accuracy on MMLU test set"""
    model.eval()
    
    # Token IDs for answer choices
    choice_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"]
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
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


def main():
    args = parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with compression configuration
    print(f"Loading LlamaComp model from {args.model_name}...")
    model = LlamaCompForCausalLM.from_pretrained(
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
                # print(f"  Training: {name}")
    
    analyze_lora_parameters(model)

    # Prepare dataset
    train_dataset = prepare_mmlu_dataset(tokenizer, args.max_seq_length)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    with open(f"{output_dir}/config.txt", "w") as f:
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
        output_dir=args.output_dir,
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
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        save_safetensors=True,
        report_to=["tensorboard"],
    )
    # model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Create trainer
    trainer = LlamaCompTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # Evaluate before training
    print("\n=== Evaluating on MMLU test set (before training) ===")
    mmlu_test = load_dataset("cais/mmlu", "all", split="test")
    
    # Create test dataloader

    
    test_dataloader = DataLoader(
        mmlu_test,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=2
    )
    test_dataloader.dataset.max_seq_length = args.max_seq_length
    
    initial_accuracy, n_samples = evaluate_mmlu_accuracy(
        model, tokenizer, test_dataloader, device=model.device
    )
    print(f"Initial MMLU Accuracy: {initial_accuracy:.2%} (on {n_samples} samples)")
    
    # Train model
    print("\n=== Starting training ===")
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {args.output_dir}/final_model")
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # model.gradient_checkpointing_disable()
    # model.config.use_cache = True  # Re-enable cache after training
    
    # Evaluate after training
    print("\n=== Evaluating on MMLU test set (after training) ===")
    final_accuracy, _ = evaluate_mmlu_accuracy(
        model, tokenizer, test_dataloader, device=model.device
    )
    print(f"Final MMLU Accuracy: {final_accuracy:.2%} (on {n_samples} samples)")
    print(f"Improvement: {(final_accuracy - initial_accuracy)*100:.2f} percentage points")


if __name__ == "__main__":
    main()