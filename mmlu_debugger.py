import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
import json


def analyze_lora_parameters(model):
    """
    Analyze LoRA parameter distribution and memory usage in a LlamaComp model.
    
    Args:
        model: LlamaCompForCausalLM instance
    
    Returns:
        dict: Analysis results including parameter counts and memory usage
    """
    analysis = {
        'lora_params_by_layer': {},
        'base_params_by_layer': {},
        'total_lora_params': 0,
        'total_base_params': 0,
        'total_params': 0,
        'lora_memory_mb': 0,
        'base_memory_mb': 0,
        'total_memory_mb': 0,
        'lora_ratio': 0,
    }
    
    # Analyze each layer
    for layer_idx, layer in enumerate(model.model.layers):
        layer_lora_params = 0
        layer_base_params = 0
        
        # Count parameters in this layer
        for name, param in layer.named_parameters():
            param_count = param.numel()
            if 'lora_' in name:
                layer_lora_params += param_count
                analysis['total_lora_params'] += param_count
            else:
                layer_base_params += param_count
                analysis['total_base_params'] += param_count
        
        analysis['lora_params_by_layer'][layer_idx] = layer_lora_params
        analysis['base_params_by_layer'][layer_idx] = layer_base_params
    
    # Count other model parameters (embeddings, LM head, etc.)
    for name, param in model.named_parameters():
        if not any(f'layers.{i}.' in name for i in range(len(model.model.layers))):
            param_count = param.numel()
            if 'lora_' in name:
                analysis['total_lora_params'] += param_count
            else:
                analysis['total_base_params'] += param_count
    
    # Calculate totals
    analysis['total_params'] = analysis['total_lora_params'] + analysis['total_base_params']
    
    # Calculate memory usage (assuming float16)
    bytes_per_param = 2  # 2 bytes for float16
    analysis['lora_memory_mb'] = (analysis['total_lora_params'] * bytes_per_param) / (1024 * 1024)
    analysis['base_memory_mb'] = (analysis['total_base_params'] * bytes_per_param) / (1024 * 1024)
    analysis['total_memory_mb'] = (analysis['total_params'] * bytes_per_param) / (1024 * 1024)
    
    # Calculate ratio
    if analysis['total_params'] > 0:
        analysis['lora_ratio'] = analysis['total_lora_params'] / analysis['total_params']
    
    print(f"Total LoRA parameters: {analysis['total_lora_params']}"
          f" ({analysis['lora_memory_mb']:.2f} MB)"
          f" ({analysis['lora_ratio']:.2%} of total)"
          f" across {len(analysis['lora_params_by_layer'])} layers"
          f" with {len(model.model.layers)} layers in total"
          f" ({analysis['total_memory_mb']:.2f} MB)")

    return analysis


class MMLUDebugger:
    """Debug and analyze MMLU training issues"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def analyze_answer_token_prediction(self, dataloader, num_samples=100):
        """Analyze if model is correctly predicting answer tokens"""
        self.model.eval()
        
        results = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'answer_logit_ranks': [],
            'answer_probabilities': [],
            'top_5_accuracy': 0,
            'confusion_matrix': defaultdict(lambda: defaultdict(int))
        }
        
        # Get token IDs for A, B, C, D
        answer_tokens = {
            'A': self.tokenizer.encode(" A", add_special_tokens=False)[0],
            'B': self.tokenizer.encode(" B", add_special_tokens=False)[0],
            'C': self.tokenizer.encode(" C", add_special_tokens=False)[0],
            'D': self.tokenizer.encode(" D", add_special_tokens=False)[0],
        }
        answer_token_ids = list(answer_tokens.values())
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Analyzing predictions")):
                if batch_idx >= num_samples:
                    break
                    
                for q, choices, true_answer in zip(batch['question'], batch['choices'], batch['answer']):
                    # Format prompt
                    prompt = self._format_mmlu_prompt(q, choices)
                    inputs = self.tokenizer(prompt, return_tensors="pt", 
                                          truncation=True, max_length=1536).to(self.device)
                    
                    # Get model prediction
                    outputs = self.model(**inputs)
                    last_logits = outputs.logits[0, -1]
                    
                    # Analyze answer token predictions
                    answer_logits = last_logits[answer_token_ids]
                    answer_probs = torch.softmax(answer_logits, dim=0)
                    
                    # Get predicted answer
                    pred_idx = torch.argmax(answer_logits).item()
                    true_idx = true_answer.item()
                    
                    # Record results
                    results['total_predictions'] += 1
                    if pred_idx == true_idx:
                        results['correct_predictions'] += 1
                    
                    # Record probability of correct answer
                    results['answer_probabilities'].append(answer_probs[true_idx].item())
                    
                    # Get rank of correct answer in all logits
                    sorted_indices = torch.argsort(last_logits, descending=True)
                    correct_token_id = answer_token_ids[true_idx]
                    rank = (sorted_indices == correct_token_id).nonzero()[0].item()
                    results['answer_logit_ranks'].append(rank)
                    
                    # Top-5 accuracy
                    if correct_token_id in sorted_indices[:5]:
                        results['top_5_accuracy'] += 1
                    
                    # Confusion matrix
                    true_letter = chr(65 + true_idx)
                    pred_letter = chr(65 + pred_idx)
                    results['confusion_matrix'][true_letter][pred_letter] += 1
        
        # Calculate final metrics
        results['accuracy'] = results['correct_predictions'] / results['total_predictions']
        results['top_5_accuracy'] /= results['total_predictions']
        results['avg_correct_answer_prob'] = np.mean(results['answer_probabilities'])
        results['avg_correct_answer_rank'] = np.mean(results['answer_logit_ranks'])
        
        return results
    
    def analyze_gradient_flow(self, dataloader, num_batches=10):
        """Analyze gradient flow through the model"""
        self.model.train()
        
        gradient_norms = defaultdict(list)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            # Forward pass
            inputs = {k: v.to(self.device) for k, v in batch.items() 
                     if k in ['input_ids', 'attention_mask', 'labels']}
            outputs = self.model(**inputs)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Collect gradient norms
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norms[name].append(grad_norm)
            
            self.model.zero_grad()
        
        # Analyze gradient statistics
        gradient_stats = {}
        for name, norms in gradient_norms.items():
            gradient_stats[name] = {
                'mean': np.mean(norms),
                'std': np.std(norms),
                'max': np.max(norms),
                'min': np.min(norms),
            }
        
        return gradient_stats
    
    def analyze_layer_outputs(self, dataloader, num_samples=50):
        """Analyze hidden state statistics across layers"""
        self.model.eval()
        
        layer_stats = defaultdict(lambda: {
            'mean_norm': [],
            'std_norm': [],
            'dead_neurons': [],
            'activation_sparsity': []
        })
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break
                
                inputs = {k: v.to(self.device) for k, v in batch.items() 
                         if k in ['input_ids', 'attention_mask']}
                
                # Get outputs with hidden states
                outputs = self.model.model(**inputs, output_hidden_states=True)
                
                # Analyze each layer's output
                for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                    # Compute statistics
                    norms = hidden_state.norm(dim=-1)
                    layer_stats[layer_idx]['mean_norm'].append(norms.mean().item())
                    layer_stats[layer_idx]['std_norm'].append(norms.std().item())
                    
                    # Check for dead neurons (very low activation)
                    dead_mask = (hidden_state.abs() < 1e-3).float()
                    dead_ratio = dead_mask.mean().item()
                    layer_stats[layer_idx]['dead_neurons'].append(dead_ratio)
                    
                    # Activation sparsity
                    sparsity = (hidden_state == 0).float().mean().item()
                    layer_stats[layer_idx]['activation_sparsity'].append(sparsity)
        
        # Aggregate statistics
        final_stats = {}
        for layer_idx, stats in layer_stats.items():
            final_stats[f'layer_{layer_idx}'] = {
                key: {'mean': np.mean(values), 'std': np.std(values)}
                for key, values in stats.items()
            }
        
        return final_stats
    
    def check_cycling_consistency(self, dataloader, num_samples=20):
        """Check if cycling layers produce consistent outputs"""
        if not hasattr(self.model.config, 'cycle_layers') or not self.model.config.cycle_layers:
            return {"message": "No cycling layers configured"}
        
        self.model.eval()
        consistency_scores = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break
                
                inputs = {k: v.to(self.device) for k, v in batch.items() 
                         if k in ['input_ids', 'attention_mask']}
                
                # Track outputs at each cycle
                cycle_outputs = defaultdict(list)
                
                # Custom forward to track cycle outputs
                # This requires modifying the model temporarily
                original_forward = self.model.model.forward
                
                def tracking_forward(*args, **kwargs):
                    # Implementation would track outputs per cycle
                    return original_forward(*args, **kwargs)
                
                self.model.model.forward = tracking_forward
                outputs = self.model(**inputs)
                self.model.model.forward = original_forward
                
                # Analyze consistency between cycles
                # (Simplified version - actual implementation would be more complex)
                
        return consistency_scores
    
    def _format_mmlu_prompt(self, question, choices):
        """Format MMLU question into prompt"""
        prompt = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nAnswer:"
        return prompt
    
    def visualize_results(self, results, save_path="mmlu_debug_results.png"):
        """Visualize debugging results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Answer probability distribution
        ax = axes[0, 0]
        ax.hist(results['answer_probabilities'], bins=30, alpha=0.7, color='blue')
        ax.axvline(x=0.25, color='red', linestyle='--', label='Random guess')
        ax.set_xlabel('Probability of Correct Answer')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Correct Answer Probabilities')
        ax.legend()
        
        # 2. Answer rank distribution
        ax = axes[0, 1]
        rank_counts = np.bincount(results['answer_logit_ranks'][:100])  # Top 100 ranks
        ax.bar(range(len(rank_counts)), rank_counts)
        ax.set_xlabel('Rank of Correct Answer Token')
        ax.set_ylabel('Count')
        ax.set_title('Where Correct Answer Ranks in All Logits')
        ax.set_xlim(0, 50)
        
        # 3. Confusion matrix
        ax = axes[1, 0]
        confusion_matrix = np.zeros((4, 4))
        for i, true_label in enumerate(['A', 'B', 'C', 'D']):
            for j, pred_label in enumerate(['A', 'B', 'C', 'D']):
                confusion_matrix[i, j] = results['confusion_matrix'][true_label][pred_label]
        
        sns.heatmap(confusion_matrix, annot=True, fmt='g', 
                   xticklabels=['A', 'B', 'C', 'D'],
                   yticklabels=['A', 'B', 'C', 'D'],
                   cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        # 4. Summary metrics
        ax = axes[1, 1]
        metrics_text = f"""
        Accuracy: {results['accuracy']:.2%}
        Top-5 Accuracy: {results['top_5_accuracy']:.2%}
        
        Avg Correct Answer Probability: {results['avg_correct_answer_prob']:.3f}
        Avg Correct Answer Rank: {results['avg_correct_answer_rank']:.1f}
        
        Total Predictions: {results['total_predictions']}
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        ax.axis('off')
        ax.set_title('Summary Metrics')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path


def diagnose_mmlu_training_issues(model, tokenizer, device="cuda"):
    """Comprehensive diagnosis of MMLU training issues"""
    print("=== MMLU Training Diagnosis ===\n")
    
    # Load MMLU test data
    print("Loading MMLU test dataset...")
    mmlu_test = load_dataset("cais/mmlu", "all", split="test[:500]")  # Use subset for speed
    
    # Create dataloader
    def collate_fn(batch):
        return {
            'question': [item['question'] for item in batch],
            'choices': [item['choices'] for item in batch],
            'answer': torch.tensor([item['answer'] for item in batch]),
            'subject': [item['subject'] for item in batch]
        }
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        mmlu_test,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Initialize debugger
    debugger = MMLUDebugger(model, tokenizer, device)
    
    # 1. Analyze answer predictions
    print("\n1. Analyzing answer token predictions...")
    answer_results = debugger.analyze_answer_token_prediction(dataloader, num_samples=50)
    
    print(f"   - Accuracy: {answer_results['accuracy']:.2%}")
    print(f"   - Top-5 Accuracy: {answer_results['top_5_accuracy']:.2%}")
    print(f"   - Avg probability of correct answer: {answer_results['avg_correct_answer_prob']:.3f}")
    print(f"   - Avg rank of correct answer: {answer_results['avg_correct_answer_rank']:.1f}")
    
    # 2. Check gradient flow
    print("\n2. Checking gradient flow...")
    gradient_stats = debugger.analyze_gradient_flow(dataloader, num_batches=5)
    
    # Find layers with very small or very large gradients
    problematic_layers = []
    for name, stats in gradient_stats.items():
        if stats['mean'] < 1e-6:
            problematic_layers.append((name, "vanishing gradients"))
        elif stats['mean'] > 10:
            problematic_layers.append((name, "exploding gradients"))
    
    if problematic_layers:
        print("   - Found problematic layers:")
        for layer, issue in problematic_layers[:5]:
            print(f"     * {layer}: {issue}")
    else:
        print("   - Gradient flow appears normal")
    
    # 3. Analyze layer outputs
    print("\n3. Analyzing layer outputs...")
    layer_stats = debugger.analyze_layer_outputs(dataloader, num_samples=20)
    
    # Check for dead neurons
    dead_neuron_layers = []
    for layer_name, stats in layer_stats.items():
        if stats['dead_neurons']['mean'] > 0.1:
            dead_neuron_layers.append(layer_name)
    
    if dead_neuron_layers:
        print(f"   - Found {len(dead_neuron_layers)} layers with >10% dead neurons")
    else:
        print("   - Layer activations appear healthy")
    
    # 4. Visualize results
    print("\n4. Creating visualization...")
    viz_path = debugger.visualize_results(answer_results)
    print(f"   - Saved visualization to: {viz_path}")
    
    # 5. Recommendations
    print("\n=== Recommendations ===")
    recommendations = []
    
    if answer_results['accuracy'] < 0.3:
        recommendations.append("- Accuracy is below random (25%). Check data formatting and loss computation.")
    
    if answer_results['avg_correct_answer_prob'] < 0.25:
        recommendations.append("- Model is not learning to predict answer tokens. Consider:")
        recommendations.append("  * Increasing learning rate")
        recommendations.append("  * Checking if answer tokens are properly masked in loss")
        recommendations.append("  * Verifying tokenization of answer choices")
    
    if problematic_layers:
        recommendations.append("- Found gradient flow issues. Consider:")
        recommendations.append("  * Using gradient clipping")
        recommendations.append("  * Adjusting learning rate")
        recommendations.append("  * Checking layer initialization")
    
    if dead_neuron_layers:
        recommendations.append("- Found dead neurons. Consider:")
        recommendations.append("  * Using different activation functions")
        recommendations.append("  * Reducing dropout rate")
        recommendations.append("  * Checking for numerical instabilities")
    
    if answer_results['avg_correct_answer_rank'] > 100:
        recommendations.append("- Correct answers have very low logit rankings. Consider:")
        recommendations.append("  * Training for more epochs")
        recommendations.append("  * Using a larger LoRA rank")
        recommendations.append("  * Checking if model architecture changes are too aggressive")
    
    for rec in recommendations:
        print(rec)
    
    return {
        'answer_results': answer_results,
        'gradient_stats': gradient_stats,
        'layer_stats': layer_stats,
        'recommendations': recommendations
    }