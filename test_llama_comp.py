#!/usr/bin/env python3
"""
Example usage of LlamaComp for model compression and fine-tuning

This script demonstrates:
1. Layer pruning - removing specific layers
2. Layer cycling - repeating layer blocks  
3. Layer-wise LoRA - different adapters per cycle
4. Knowledge distillation - learning from the original model
"""

import torch
from transformers import AutoTokenizer
from models.llama.LLaMaComp import LlamaCompForCausalLM


def main():
    # Example 1: Basic layer pruning
    print("=== Example 1: Layer Pruning ===")
    model_name = "meta-llama/Llama-3.2-1B"
    
    # # Prune layers 12, 13, 14 from a 16-layer model
    # model = LlamaCompForCausalLM.from_pretrained(
    #     model_name,
    #     pruned_layers=[12, 13, 14],
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    
    # print(f"Original model layers: 16")
    # print(f"Pruned layers: [12, 13, 14]")
    # print(f"Execution sequence: {[layer_idx for layer_idx, _ in model.model.execution_sequence]}")
    # print(f"Total execution steps: {len(model.model.execution_sequence)}")
    
    # # Example 2: Layer cycling without pruning
    # print("\n=== Example 2: Layer Cycling ===")
    
    # # Cycle layers 8, 9, 10, 11 twice
    # model = LlamaCompForCausalLM.from_pretrained(
    #     model_name,
    #     cycle_layers=[8, 9, 10, 11],
    #     cycle_count=2,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    
    # print(f"Cycle layers: [8, 9, 10, 11]")
    # print(f"Cycle count: 2")
    # print(f"Execution sequence:")
    # for i, (layer_idx, cycle_idx) in enumerate(model.model.execution_sequence):
    #     cycle_str = f" (cycle {cycle_idx})" if cycle_idx is not None else ""
    #     print(f"  Step {i}: Layer {layer_idx}{cycle_str}")
    
    # # Example 3: Combined pruning and cycling
    # print("\n=== Example 3: Pruning + Cycling ===")
    
    # # Prune layers 12, 13, 14 and cycle layers 8, 9, 10, 11 twice
    # model = LlamaCompForCausalLM.from_pretrained(
    #     model_name,
    #     pruned_layers=[12, 13, 14],
    #     cycle_layers=[8, 9, 10, 11],
    #     cycle_count=2,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    
    # print(f"Pruned layers: [12, 13, 14]")
    # print(f"Cycle layers: [8, 9, 10, 11]")
    # print(f"Cycle count: 2")
    # print(f"Execution flow: 0→1→2→...→7→[8→9→10→11]×2→15")
    # print(f"Total execution steps: {len(model.model.execution_sequence)}")
    
    # Example 4: With LoRA adapters
    print("\n=== Example 4: Cycling with LoRA ===")
    
    model = LlamaCompForCausalLM.from_pretrained(
        model_name,
        cycle_layers=[8, 9, 10, 11],
        cycle_count=2,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16.0,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"LoRA enabled for cycled layers")
    print(f"LoRA rank: 8, alpha: 16.0")
    print(f"Each cycled layer has separate LoRA adapters per cycle")
    
    # Count LoRA parameters
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}% of total)")
    
    # Example 5: Complete compression setup for MMLU
    print("\n=== Example 5: MMLU Compression Setup ===")
    
    # For a 16-layer Llama model, keep first 2 and last 2 layers,
    # sample 6 from middle layers, and cycle them twice
    # This gives us: 2 + 6×2 + 2 = 16 effective layers
    
    # Middle layers: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
    # Sample 6: let's take 4, 5, 7, 8, 10, 11
    # Prune: 2, 3, 6, 9, 12, 13
    
    model = LlamaCompForCausalLM.from_pretrained(
        model_name,
        pruned_layers=[2, 3, 6, 9, 12, 13],
        cycle_layers=[4, 5, 7, 8, 10, 11],
        cycle_count=2,
        use_lora=True,
        lora_rank=16,
        use_distillation=True,
        distillation_alpha=0.5,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Compression strategy:")
    print(f"- Keep layers: 0, 1 (front) + 14, 15 (back)")
    print(f"- Cycle layers: [4, 5, 7, 8, 10, 11] × 2")
    print(f"- Total effective layers: 16 (matches original)")
    print(f"- Using LoRA rank 16 for fine-tuning")
    print(f"- Distillation enabled with α=0.5")
    
    # Test inference
    print("\n=== Testing Inference ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Sample MMLU-style question
    prompt = """Question: What is the capital of France?

A. London
B. Berlin
C. Paris
D. Madrid

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.1,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model response: {response}")
    
    # Memory usage comparison
    print("\n=== Memory Usage ===")
    
    # Count active parameters
    active_layers = [i for i in range(16) if i not in [2, 3, 6, 9, 12, 13]]
    base_params = sum(p.numel() for i, layer in enumerate(model.model.layers) 
                     if i in active_layers for p in layer.parameters())
    
    # Add embedding and LM head
    other_params = (model.model.embed_tokens.weight.numel() + 
                   model.model.norm.weight.numel() +
                   model.lm_head.weight.numel())
    
    total_compressed = base_params + other_params + lora_params
    original_params = sum(p.numel() for p in model.parameters())
    
    print(f"Compressed model parameters: {total_compressed:,}")
    print(f"Original model parameters: {original_params:,}")
    print(f"Compression ratio: {original_params/total_compressed:.2f}x")


if __name__ == "__main__":
    main()