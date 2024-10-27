import torch
import torch.nn.functional as F
import logging
import math
from collections import deque
from enum import Enum
from typing import Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SamplerState(Enum):
    COLD = 0    # Very certain - low temperature
    HOT = 1     # Very uncertain - high temperature
    MIXED = 2   # Intermediate state

class SamplerConfig:
    def __init__(self, tokenizer=None):  # Accept tokenizer parameter but don't use it
        # Base parameters
        self.base_temp = 0.4
        self.base_top_p = 0.85
        self.base_top_k = 40
        self.base_rep_penalty = 1.0
        
        # Phase transition points
        self.COLD_POINT = 0.6
        self.HOT_POINT = 1.8
        
        # Speculative decoding parameters
        self.spec_top_k = 3
        self.spec_top_p = 0.9
        self.entropy_weight = 0.8
        self.var_entropy_weight = 0.2

class AdaptiveEntropixSampler:
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.alpha_window = deque(maxlen=100)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def calculate_entropy(self, probs: torch.Tensor, log_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate normalized entropy."""
        if log_probs is None:
            log_probs = torch.log2(torch.clamp(probs, min=1e-10))
        entropy = -torch.sum(probs * log_probs, dim=-1)
        max_entropy = math.log2(probs.size(-1))
        return entropy / max_entropy

    def calculate_alpha(self, logits: torch.Tensor, attention: torch.Tensor) -> float:
        """Calculate alpha using token and attention entropy with normalization."""
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
            
        # Token Distribution Uncertainty
        log_probs = F.log_softmax(logits, dim=-1) / math.log(2)
        probs = torch.exp(log_probs * math.log(2))
        token_entropy = -torch.sum(probs * log_probs, dim=-1)
        max_token_entropy = math.log2(logits.size(-1))
        norm_token_entropy = (token_entropy / max_token_entropy).mean().item()
        
        # Attention Coherence
        if len(attention.shape) == 4:
            attention = attention.mean(dim=1)
        elif len(attention.shape) == 3:
            attention = attention.unsqueeze(0)
        
        attention_probs = F.softmax(attention, dim=-1)
        attn_entropy = self.calculate_entropy(attention_probs)
        norm_attn_entropy = attn_entropy.mean().item()
        
        # Combined metrics
        alpha = 0.5 * norm_token_entropy + 0.5 * norm_attn_entropy
        return alpha * 2.0

    def speculative_decode(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, float]:
        """Implement speculative decoding for next token selection."""
        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                output_attentions=True
            )
            
            first_token_logits = F.softmax(outputs.logits[0, -1, :], dim=-1)
            top_k_logits, top_k_indices = torch.topk(first_token_logits, self.config.spec_top_k)
            
            # Apply top-p filtering
            cumulative_probs = torch.cumsum(top_k_logits, dim=0)
            top_p_mask = cumulative_probs <= self.config.spec_top_p
            if not torch.any(top_p_mask):
                top_p_mask[0] = True
            top_k_logits, top_k_indices = top_k_logits[top_p_mask], top_k_indices[top_p_mask]

        min_diff = float('inf')
        best_idx = None
        new_attention_mask = torch.cat([attention_mask, torch.ones(1, 1).long().to(device)], dim=-1)

        # Evaluate candidates
        for idx in top_k_indices:
            new_token = idx.unsqueeze(0).unsqueeze(0)
            new_tokens = torch.cat([input_ids, new_token], dim=-1)
            
            with torch.no_grad():
                output = model.generate(
                    new_tokens,
                    attention_mask=new_attention_mask,
                    max_new_tokens=1,
                    output_scores=True,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    past_key_values=past_key_values
                )
                
                all_attentions = output.attentions[0][-1]
                attn_probs = F.softmax(all_attentions[:, -1, :], dim=-1)
                entropy = -torch.sum(attn_probs * torch.log2(attn_probs + 1e-12), dim=-1)
                
                avg_entropy = torch.mean(entropy)
                avg_varentropy = torch.var(entropy)
                diff = (avg_entropy * self.config.entropy_weight + 
                       avg_varentropy * self.config.var_entropy_weight)

                if diff < min_diff:
                    min_diff, best_idx = diff, idx

        return best_idx.unsqueeze(0).unsqueeze(0), min_diff

    def update_alpha_window(self, alpha: float) -> float:
        """Update and return moving average of alpha."""
        self.alpha_window.append(alpha)
        return sum(self.alpha_window) / len(self.alpha_window)

    def calculate_temperature(self, alpha: float) -> float:
        """Calculate adaptive temperature."""
        return 0.1 + 1.2 / (1 + math.exp(-1.5 * (alpha - 1.0)))
        
    def calculate_top_k(self, alpha: float) -> int:
        """Calculate adaptive top-k."""
        return int(10 + 190 * (math.atan(alpha - 1.0) / math.pi + 0.5))
        
    def calculate_top_p(self, alpha: float) -> float:
        """Calculate adaptive top-p."""
        return 0.85 + 0.1 * (2 / (1 + math.exp(-0.8 * alpha)) - 1)

    def calculate_rep_penalty(self, alpha: float) -> float:
        """Calculate adaptive repetition penalty."""
        return 1.0 + 0.3 / (1 + math.exp(-2 * (alpha - 1.0)))

    def determine_state(self, alpha: float) -> SamplerState:
        """Determine current thermodynamic state."""
        if alpha < self.config.COLD_POINT:
            return SamplerState.COLD
        elif alpha > self.config.HOT_POINT:
            return SamplerState.HOT
        return SamplerState.MIXED

    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to recent tokens."""
        if input_ids.size(1) > 128:
            recent_ids = input_ids[:, -128:]
        else:
            recent_ids = input_ids
            
        unique_tokens = torch.unique(recent_ids)
        logits[:, unique_tokens] = logits[:, unique_tokens] / penalty
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        attention: torch.Tensor,
        input_ids: torch.Tensor,
        model: Optional[AutoModelForCausalLM] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Enhanced sampling using both thermodynamic parameters and speculative decoding.
        """
        try:
            # Calculate and update alpha
            alpha = self.calculate_alpha(logits, attention)
            avg_alpha = self.update_alpha_window(alpha)
            
            # Calculate adaptive parameters
            temperature = self.calculate_temperature(avg_alpha)
            top_k = self.calculate_top_k(avg_alpha)
            top_p = self.calculate_top_p(avg_alpha)
            rep_penalty = self.calculate_rep_penalty(avg_alpha)
            
            # Ensure proper shape
            if len(logits.shape) == 3:
                logits = logits.squeeze(0)
                
            # Apply repetition penalty
            logits = self.apply_repetition_penalty(logits, input_ids, rep_penalty)
            
            # Use speculative decoding in COLD state if model is provided
            if (self.determine_state(avg_alpha) == SamplerState.COLD and 
                model is not None and attention_mask is not None):
                sampled_token, entropy_score = self.speculative_decode(
                    model, input_ids, attention_mask, past_key_values
                )
            else:
                # Traditional sampling for HOT and MIXED states
                scaled_logits = logits / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    values, _ = torch.topk(scaled_logits, top_k)
                    min_values = values[..., -1].unsqueeze(-1).expand_as(scaled_logits)
                    scaled_logits = torch.where(
                        scaled_logits < min_values,
                        torch.full_like(scaled_logits, float('-inf')),
                        scaled_logits
                    )
                
                probs = F.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )
                
                scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
                final_probs = F.softmax(scaled_logits, dim=-1)
                sampled_token = torch.multinomial(final_probs, 1)
                entropy_score = self.calculate_entropy(final_probs)

            return sampled_token, {
                'alpha': alpha,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'rep_penalty': rep_penalty,
                'entropy_score': entropy_score.item() if isinstance(entropy_score, torch.Tensor) else entropy_score,
                'state': self.determine_state(avg_alpha)
            }

        except Exception as e:
            self.logger.error(f"Error during sampling: {str(e)}")
            raise

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 500):
    cfg = SamplerConfig(tokenizer)
    sampler = AdaptiveEntropixSampler(cfg)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    generated_text = ""
    
    logger.info(f"Generating response for prompt: '{prompt}'")
    print("\nGenerating: ", end="", flush=True)

    for _ in range(max_tokens):
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True
                )
            
            logits = outputs.logits[:, -1:, :]
            
            if hasattr(outputs, 'attentions') and outputs.attentions:
                attention = outputs.attentions[-1].to(device)
            else:
                seq_len = input_ids.size(1)
                attention = torch.ones(1, 1, seq_len, seq_len, device=device)
            
            sampled_token, params = sampler.sample(logits, attention, input_ids)
            sampled_token = sampled_token.view(1, 1)
            
            if sampled_token.item() == tokenizer.eos_token_id:
                break

            next_token_text = tokenizer.decode(sampled_token.view(-1).item())
            generated_text += next_token_text
            print(next_token_text, end="", flush=True)

            input_ids = torch.cat([input_ids, sampled_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            logger.error(f"Current shapes - input_ids: {input_ids.shape}, "
                        f"attention_mask: {attention_mask.shape}, "
                        f"logits: {logits.shape}, "
                        f"attention: {attention.shape}, "
                        f"sampled_token: {sampled_token.shape}")
            break

    print("\n")
    return generated_text

def main():
    try:
        if sys.platform == 'win32':
            _ = system('cls')
        else:
            _ = system('clear')

        print("EntropixSampler Real-time Generation Demo")
        print("----------------------------------------")
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading model {model_name}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_attentions=True,
            return_dict_in_generate=True,
            attn_implementation="eager"
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded successfully!")
        print("\nType your prompts below ('quit' to exit)")
        print("----------------------------------------")

        while True:
            prompt = input("\nPrompt> ")
            if prompt.lower() == 'quit':
                break

            response = generate_response(model, tokenizer, prompt)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    
    finally:
        print("\nThank you for using EntropixSampler!")

if __name__ == "__main__":
    from os import system
    main()