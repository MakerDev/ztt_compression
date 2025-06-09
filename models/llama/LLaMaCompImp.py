import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import warnings

# Import base components
from .LLaMaComp import (
    LlamaCompConfig, LlamaCompModel, LlamaCompForCausalLM,
    LoRALayer, LlamaCompDecoderLayer
)


class ImprovedLoRALayer(nn.Module):
    """Improved LoRA layer with better initialization and residual connection"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        use_rslora: bool = True,  # Use Rank-Stabilized LoRA
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.use_rslora = use_rslora
        
        # Improved scaling factor
        if use_rslora:
            self.scaling = alpha / np.sqrt(rank)
        else:
            self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        
        # Better initialization
        nn.init.normal_(self.lora_A.weight, std=1 / np.sqrt(rank))
        nn.init.zeros_(self.lora_B.weight)
        
        # Optional: learnable gate for adaptive mixing
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA with gating mechanism"""
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x))) * self.scaling
        # Use sigmoid gate to control LoRA contribution
        gate_value = torch.sigmoid(self.gate)
        return lora_out * gate_value


class LayerImportanceAnalyzer:
    """Analyze layer importance to guide pruning decisions"""
    
    @staticmethod
    def compute_layer_importance(model: PreTrainedModel, dataloader, num_samples: int = 100):
        """Compute importance scores for each layer using gradient-based method"""
        model.eval()
        importance_scores = {}
        
        # Hook to capture gradients
        handles = []
        gradients = {}
        
        def backward_hook(module, grad_input, grad_output, name):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach().abs().mean().item()
        
        # Register hooks
        for idx, layer in enumerate(model.model.layers):
            name = f"layer_{idx}"
            handle = layer.register_backward_hook(
                lambda m, gi, go, n=name: backward_hook(m, gi, go, n)
            )
            handles.append(handle)
        
        # Compute gradients
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            inputs = {k: v.to(model.device) for k, v in batch.items() 
                     if k in ['input_ids', 'attention_mask', 'labels']}
            
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            
            # Store gradients
            for name, grad in gradients.items():
                if name not in importance_scores:
                    importance_scores[name] = []
                importance_scores[name].append(grad)
            
            model.zero_grad()
            gradients.clear()
        
        # Clean up hooks
        for handle in handles:
            handle.remove()
        
        # Average importance scores
        final_scores = {}
        for name, scores in importance_scores.items():
            final_scores[name] = np.mean(scores)
        
        return final_scores
    
    @staticmethod
    def suggest_layers_to_prune(importance_scores: Dict[str, float], 
                               prune_ratio: float = 0.3) -> List[int]:
        """Suggest which layers to prune based on importance scores"""
        # Sort layers by importance
        sorted_layers = sorted(importance_scores.items(), 
                             key=lambda x: x[1], reverse=False)
        
        # Select bottom k layers to prune
        num_to_prune = int(len(sorted_layers) * prune_ratio)
        layers_to_prune = []
        
        for layer_name, _ in sorted_layers[:num_to_prune]:
            layer_idx = int(layer_name.split('_')[1])
            layers_to_prune.append(layer_idx)
        
        # Don't prune first and last layers
        layers_to_prune = [l for l in layers_to_prune 
                          if l not in [0, len(sorted_layers)-1]]
        
        return sorted(layers_to_prune)


class ImprovedLlamaCompForCausalLM(LlamaCompForCausalLM):
    """Improved LlamaComp with better training dynamics"""
    
    def __init__(self, config: LlamaCompConfig):
        super().__init__(config)
        
        # Add layer normalization for cycling stability
        if config.cycle_layers and config.cycle_count > 1:
            self.cycle_norm = nn.ModuleDict({
                f"cycle_{i}": nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
                for i in range(1, config.cycle_count)  # Skip first cycle
            })
        else:
            self.cycle_norm = None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        # Get base outputs with modified forward for cycling stability
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_layer_states=self.config.use_distillation and self.teacher_model is not None,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Improved loss computation with label smoothing
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Label smoothing
            label_smoothing = getattr(self.config, 'label_smoothing', 0.1)
            if label_smoothing > 0:
                loss = self._compute_loss_with_label_smoothing(
                    shift_logits, shift_labels, label_smoothing
                )
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), 
                               shift_labels.view(-1))
            
            # Distillation with improved alignment
            if self.config.use_distillation and self.teacher_model is not None:
                distill_loss = self._compute_improved_distillation_loss(
                    outputs, input_ids, attention_mask, logits
                )
                loss = (1 - self.config.distillation_alpha) * loss + \
                       self.config.distillation_alpha * distill_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _compute_loss_with_label_smoothing(self, logits, labels, smoothing=0.1):
        """Compute cross-entropy loss with label smoothing"""
        vocab_size = logits.size(-1)
        confidence = 1.0 - smoothing
        smoothing_value = smoothing / (vocab_size - 1)
        
        # Create smoothed target distribution
        one_hot = torch.zeros_like(logits).scatter(-1, labels.unsqueeze(-1), 1)
        smoothed_targets = one_hot * confidence + (1 - one_hot) * smoothing_value
        
        # Compute KL divergence
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
        
        return loss
    
    def _compute_improved_distillation_loss(self, student_outputs, input_ids, 
                                           attention_mask, student_logits):
        """Improved distillation with better layer alignment"""
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs.logits
        
        # Logit distillation with attention to important tokens
        # Focus on answer tokens in MMLU
        answer_positions = self._find_answer_positions(input_ids)
        
        # Temperature-scaled KL divergence
        T = self.config.distillation_temperature
        kl_loss = nn.KLDivLoss(reduction="none")
        
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        token_losses = kl_loss(student_log_probs, teacher_probs).sum(dim=-1)
        
        # Weight answer tokens more heavily
        if answer_positions is not None:
            weights = torch.ones_like(token_losses)
            weights[answer_positions] *= 5.0  # Emphasize answer tokens
            token_losses = token_losses * weights
        
        distill_loss = token_losses.mean() * (T ** 2)
        
        # Hidden state distillation with dynamic alignment
        if hasattr(student_outputs, 'layer_states'):
            layer_loss = self._compute_layer_alignment_loss(
                student_outputs.layer_states,
                teacher_outputs.hidden_states
            )
            distill_loss += 0.5 * layer_loss
        
        return distill_loss
    
    def _find_answer_positions(self, input_ids):
        """Find positions of answer tokens in MMLU format"""
        # Look for "Answer:" pattern
        # This is a simplified version - you might need to adjust based on tokenizer
        batch_size, seq_len = input_ids.shape
        answer_positions = []
        
        # Convert to list for easier searching
        for b in range(batch_size):
            ids = input_ids[b].tolist()
            # Look for common answer patterns
            for i in range(len(ids) - 5):
                # This is tokenizer-dependent
                if "answer" in str(ids[i:i+5]).lower():
                    answer_positions.append((b, i+5))
                    break
        
        return answer_positions if answer_positions else None
    
    def _compute_layer_alignment_loss(self, student_states, teacher_states):
        """Compute alignment loss between student and teacher layers"""
        # Dynamic programming alignment
        num_student = len(student_states)
        num_teacher = len(teacher_states) - 1  # Exclude embedding
        
        # Create alignment matrix
        alignment_scores = torch.zeros(num_student, num_teacher)
        
        for i, s_state in enumerate(student_states):
            for j in range(num_teacher):
                t_state = teacher_states[j + 1]  # Skip embedding
                # Compute cosine similarity
                cos_sim = F.cosine_similarity(
                    s_state.mean(dim=(0, 1)),
                    t_state.mean(dim=(0, 1)),
                    dim=0
                )
                alignment_scores[i, j] = cos_sim
        
        # Soft alignment using attention
        alignment_weights = F.softmax(alignment_scores * 5, dim=1)  # Temperature 0.2
        
        # Compute weighted MSE loss
        total_loss = 0
        for i, s_state in enumerate(student_states):
            aligned_teacher = sum(
                alignment_weights[i, j] * teacher_states[j + 1]
                for j in range(num_teacher)
            )
            total_loss += F.mse_loss(s_state, aligned_teacher)
        
        return total_loss / num_student


class MMMLUSpecificConfig(LlamaCompConfig):
    """MMLU-optimized configuration"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # MMLU-specific defaults
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        self.answer_weight = kwargs.get('answer_weight', 5.0)
        self.warmup_cycles = kwargs.get('warmup_cycles', 100)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', True)
        
        # Better defaults for compression
        if self.use_lora:
            self.lora_target_modules = kwargs.get('lora_target_modules', 
                ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def create_mmlu_optimized_model(
    base_model_name: str,
    prune_ratio: float = 0.2,
    cycle_middle_layers: bool = True,
    use_importance_analysis: bool = True,
    **kwargs
) -> ImprovedLlamaCompForCausalLM:
    """Create an MMLU-optimized compressed model"""
    
    # Load base configuration
    from transformers import AutoConfig
    base_config = AutoConfig.from_pretrained(base_model_name)
    num_layers = base_config.num_hidden_layers
    
    # Determine layers to prune
    if use_importance_analysis:
        print("Analyzing layer importance... (this may take a few minutes)")
        # You would need to pass a small dataloader here
        # For now, we'll use a heuristic
        layers_to_prune = list(range(2, num_layers - 2, 3))[:int(num_layers * prune_ratio)]
    else:
        # Simple heuristic: prune every 3rd layer in the middle
        layers_to_prune = list(range(2, num_layers - 2, 3))[:int(num_layers * prune_ratio)]
    
    # Determine layers to cycle
    if cycle_middle_layers:
        middle_start = num_layers // 4
        middle_end = 3 * num_layers // 4
        cycle_layers = [i for i in range(middle_start, middle_end) 
                       if i not in layers_to_prune]
        cycle_count = 2
    else:
        cycle_layers = None
        cycle_count = 1
    
    # Create configuration
    config = MMMLUSpecificConfig.from_pretrained(
        base_model_name,
        pruned_layers=layers_to_prune,
        cycle_layers=cycle_layers,
        cycle_count=cycle_count,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32.0,
        use_distillation=True,
        distillation_temperature=4.0,
        distillation_alpha=0.5,
        **kwargs
    )
    
    # Create model
    model = ImprovedLlamaCompForCausalLM.from_pretrained(
        base_model_name,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    print(f"Created MMLU-optimized model:")
    print(f"  - Pruned layers: {layers_to_prune}")
    print(f"  - Cycling layers: {cycle_layers}")
    print(f"  - Total execution steps: {len(model.model.execution_sequence)}")
    print(f"  - LoRA parameters: {sum(p.numel() for n, p in model.named_parameters() if 'lora' in n):,}")
    
    return model