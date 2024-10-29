import torch
import torch.nn.functional as F
import logging
import math
from typing import Dict, Tuple, Optional, NamedTuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from collections import deque 

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OptimizationMetrics(NamedTuple):
    entropy: float
    alpha: float

class OptimizationParameters(NamedTuple):
    temperature: float
    top_k: float
    top_p: float
    rep_penalty: float
    candidate_size: float

class SamplerConfig:
    def __init__(self):
        # Base parameters
        self.base_temp = 0.2
        self.base_top_p = 0.80
        self.base_top_k = 10
        self.base_rep_penalty = 1.0
        self.base_candidate_size = 20
        
        # Entropy weights
        self.entropy_weight = 0.5
        self.attention_weight = 0.5
        
        # Parameter bounds
        self.temp_min = 0.1
        self.temp_max = 1.0
        self.top_k_min = 5
        self.top_k_max = 100
        self.top_p_min = 0.1
        self.top_p_max = 1.0
        self.rep_penalty_min = 1.1
        self.rep_penalty_max = 1.5
        self.candidate_size_min = 1
        self.candidate_size_max = 100
        
        # Optimization parameters
        self.opt_max_iters = 10
        self.opt_tolerance = 0.01
        self.opt_learning_rate = 0.1
        self.opt_min_grad = 1e-8
        
        # Parameter smoothing
        self.param_smoothing_factor = 0.8

        # Alpha window size
        self.alpha_window_size = 5

class AdaptiveEntropixSampler:
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.alpha_window = deque(maxlen=config.alpha_window_size)
        self.previous_temp = config.base_temp
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def update_alpha_window(self, alpha: float) -> float:
        """Update and return moving average of alpha."""
        self.alpha_window.append(alpha)
        return sum(self.alpha_window) / len(self.alpha_window)

    def _apply_sampling_params(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: int,
        top_p: torch.Tensor,
        rep_penalty: torch.Tensor,
        candidate_size: int
    ) -> torch.Tensor:
        """Apply all sampling parameters to logits."""
        try:
            # Ensure logits has correct shape [batch_size, seq_len, vocab_size]
            if len(logits.shape) == 2:
                logits = logits.unsqueeze(0)
            
            # Apply temperature
            scaled_logits = logits / temperature
            
            # Apply top-k
            if top_k > 0:
                values, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)), dim=-1)
                min_values = values[..., -1, None].expand_as(scaled_logits)
                scaled_logits = torch.where(
                    scaled_logits < min_values,
                    torch.full_like(scaled_logits, float('-inf')),
                    scaled_logits
                )
            
            # Convert to probabilities
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Apply top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted indices
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            
            scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Apply candidate size filtering
            probs = F.softmax(scaled_logits, dim=-1)
            if candidate_size < probs.size(-1):
                top_probs, _ = torch.topk(probs, min(candidate_size, probs.size(-1)), dim=-1)
                min_prob = top_probs[..., -1, None].expand_as(probs)
                scaled_logits = torch.where(
                    probs < min_prob,
                    torch.full_like(scaled_logits, float('-inf')),
                    scaled_logits
                )
            
            return scaled_logits
        except Exception as e:
            self.logger.error(f"Error in _apply_sampling_params: {str(e)}")
            raise

    def calculate_entropy_and_varentropy(self, probs: torch.Tensor, log_probs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate both entropy and varentropy of a probability distribution."""
        if log_probs is None:
            log_probs = torch.log2(torch.clamp(probs, min=1e-10))
        
        # Calculate entropy
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Calculate varentropy (variance of surprise)
        surprise = -log_probs  # Individual surprise values
        mean_surprise = entropy  # Mean surprise is the entropy
        squared_diff = (surprise - mean_surprise.unsqueeze(-1)) ** 2
        varentropy = torch.sum(probs * squared_diff, dim=-1)
        
        return entropy, varentropy

    def calculate_entropy(self, probs: torch.Tensor, log_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate normalized entropy."""
        if log_probs is None:
            log_probs = torch.log2(torch.clamp(probs, min=1e-10))
        entropy = -torch.sum(probs * log_probs, dim=-1)
        max_entropy = math.log2(probs.size(-1))
        return entropy / max_entropy

    def calculate_alpha(self, logits: torch.Tensor, attention: torch.Tensor) -> float:
        """Calculate alpha using token and attention entropy/varentropy with normalization."""
        try:
            # Token Distribution Analysis
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            token_entropy, token_varentropy = self.calculate_entropy_and_varentropy(probs, log_probs)
            
            # Normalize token entropy and varentropy
            max_token_entropy = math.log2(logits.size(-1))
            norm_token_entropy = (token_entropy / max_token_entropy).mean().item()
            max_token_varentropy = max_token_entropy ** 2  # Maximum possible variance
            norm_token_varentropy = (token_varentropy / max_token_varentropy).mean().item()
            
            # Attention Analysis
            if attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
                attention = attention.mean(dim=1)  # Average over heads
            
            # Get last position attention - these are already probabilities from the model
            attn_probs = attention[:, -1, :]  # [batch, seq_len]
            
            # Calculate log probs directly from the existing probabilities
            attn_log_probs = torch.log2(torch.clamp(attn_probs, min=1e-10))
            
            # Calculate attention entropy and varentropy
            attn_entropy, attn_varentropy = self.calculate_entropy_and_varentropy(attn_probs, attn_log_probs)
            
            # Normalize attention metrics
            max_attn_entropy = math.log2(attention.size(-1))
            norm_attn_entropy = (attn_entropy / max_attn_entropy).mean().item()
            max_attn_varentropy = max_attn_entropy ** 2
            norm_attn_varentropy = (attn_varentropy / max_attn_varentropy).mean().item()
            
            # Combined metrics with specified weights
            alpha = (
                0.4 * norm_token_entropy +      # Token entropy weight
                0.2 * norm_token_varentropy +   # Token varentropy weight
                0.3 * norm_attn_entropy +       # Attention entropy weight
                0.1 * norm_attn_varentropy      # Attention varentropy weight
            )
            
            # Log the components for debugging
            self.logger.debug(f"""
                Alpha Components:
                - Token Entropy: {norm_token_entropy:.4f}
                - Token Varentropy: {norm_token_varentropy:.4f}
                - Attention Entropy: {norm_attn_entropy:.4f}
                - Attention Varentropy: {norm_attn_varentropy:.4f}
                - Final Alpha: {alpha:.4f}
            """)
            
            return alpha
            
        except Exception as e:
            self.logger.error(f"Error in calculate_alpha: {str(e)}")
            raise

    def _bound_parameters(self, params: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        """Apply bounds to parameters using smooth sigmoid scaling."""
        normalized = torch.sigmoid(params)
        ranges = bounds[:, 1] - bounds[:, 0]
        bounded = bounds[:, 0] + ranges * normalized
        return bounded

    def find_optimal_parameters(
        self,
        logits: torch.Tensor,
        attention: torch.Tensor,
        target_alpha: float
    ) -> Dict[str, float]:
        """Find optimal parameters using gradient-based optimization."""
        try:
            params = torch.tensor(
                [
                    self.previous_temp,           # temperature
                    float(self.config.base_top_k),# top_k
                    self.config.base_top_p,       # top_p
                    self.config.base_rep_penalty, # rep_penalty
                    50.0,                         # candidate_size (starting value)
                ],
                device=logits.device,
                dtype=logits.dtype,
                requires_grad=True
            )
            
            bounds = torch.tensor([
                [self.config.temp_min, self.config.temp_max],  # temperature
                [5.0, 200.0],                                  # top_k
                [0.1, 1.0],                                    # top_p
                [1.0, 1.5],                                    # rep_penalty
                [1.0, 100.0],                                  # candidate_size
            ], device=logits.device)
            
            optimizer = torch.optim.Adam([params], lr=self.config.opt_learning_rate)
            
            target_entropy = target_alpha / 2.0  # Since alpha was multiplied by 2.0
            
            for iter in range(self.config.opt_max_iters):
                optimizer.zero_grad()
                
                bounded_params = self._bound_parameters(params, bounds)
                
                modified_logits = self._apply_sampling_params(
                    logits,
                    temperature=bounded_params[0],
                    top_k=int(bounded_params[1].item()),
                    top_p=bounded_params[2],
                    rep_penalty=bounded_params[3],
                    candidate_size=int(bounded_params[4].item())
                )
                
                modified_probs = F.softmax(modified_logits, dim=-1)
                modified_entropy = -torch.sum(
                    modified_probs * torch.log2(modified_probs + 1e-10),
                    dim=-1
                ).mean()
                
                max_entropy = math.log2(logits.size(-1))
                norm_modified_entropy = modified_entropy / max_entropy
                
                loss = (norm_modified_entropy - target_entropy) ** 2
                
                if loss.item() < self.config.opt_tolerance:
                    break
                    
                loss.backward()
                optimizer.step()
            
            final_params = self._bound_parameters(params, bounds)
            
            self.previous_temp = final_params[0].item()
            
            return {
                'temperature': final_params[0].item(),
                'top_k': int(final_params[1].item()),
                'top_p': final_params[2].item(),
                'rep_penalty': final_params[3].item(),
                'candidate_size': int(final_params[4].item())
            }
        except Exception as e:
            self.logger.error(f"Error in find_optimal_parameters: {str(e)}")
            raise

    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to recent tokens with proper dimension handling."""
        try:
            # Create a copy of logits to avoid modifying the original
            penalized_logits = logits.clone()
            
            # Get recent tokens
            if input_ids.size(1) > 128:
                recent_ids = input_ids[:, -128:]
            else:
                recent_ids = input_ids
                
            # Ensure we're working with the correct dimensions
            if len(penalized_logits.shape) == 3:
                # Shape: [batch_size, sequence_length, vocab_size]
                batch_size, seq_len, vocab_size = penalized_logits.shape
                
                # Get unique tokens and ensure they're within vocabulary size
                unique_tokens = torch.unique(recent_ids)
                valid_tokens = unique_tokens[unique_tokens < vocab_size]
                
                if len(valid_tokens) > 0:
                    # Apply penalty to all sequences in the batch
                    for batch_idx in range(batch_size):
                        for seq_idx in range(seq_len):
                            penalized_logits[batch_idx, seq_idx, valid_tokens] /= penalty
                            
            elif len(penalized_logits.shape) == 2:
                # Shape: [sequence_length, vocab_size]
                seq_len, vocab_size = penalized_logits.shape
                
                # Get unique tokens and ensure they're within vocabulary size
                unique_tokens = torch.unique(recent_ids)
                valid_tokens = unique_tokens[unique_tokens < vocab_size]
                
                if len(valid_tokens) > 0:
                    # Apply penalty to the sequence
                    for seq_idx in range(seq_len):
                        penalized_logits[seq_idx, valid_tokens] /= penalty
                        
            else:
                raise ValueError(f"Unexpected logits shape: {penalized_logits.shape}")
                
            return penalized_logits
            
        except Exception as e:
            self.logger.error(f"Error in apply_repetition_penalty: {str(e)}")
            # If there's an error, return original logits without penalty
            self.logger.warning("Returning original logits without repetition penalty")
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
        """Enhanced sampling using optimized parameters with improved error handling."""
        try:
            # Ensure logits are in float32
            logits = logits.to(torch.float32)
            
            # Validate tensor shapes
            if len(logits.shape) != 3:
                logits = logits.unsqueeze(0) if len(logits.shape) == 2 else logits
                
            if len(logits.shape) != 3:
                raise ValueError(f"Unable to process logits with shape: {logits.shape}")
                
            # Calculate and update alpha
            alpha = self.calculate_alpha(logits, attention)
            avg_alpha = self.update_alpha_window(alpha)
            
            # Get optimized parameters
            params = self.find_optimal_parameters(logits, attention, avg_alpha)
            
            # Apply repetition penalty with proper shape handling
            penalized_logits = self.apply_repetition_penalty(
                logits, 
                input_ids, 
                params['rep_penalty']
            )
            
            # Apply all sampling parameters
            modified_logits = self._apply_sampling_params(
                penalized_logits,
                torch.tensor(params['temperature'], device=logits.device),
                params['top_k'],
                torch.tensor(params['top_p'], device=logits.device),
                torch.tensor(params['rep_penalty'], device=logits.device),
                params['candidate_size']
            )
            
            # Calculate final probabilities
            final_probs = F.softmax(modified_logits, dim=-1)
            
            # Get probabilities for the last token
            if len(final_probs.shape) == 3:
                final_probs = final_probs[0, -1]
            elif len(final_probs.shape) == 2:
                final_probs = final_probs[-1]
                
            # Handle any remaining zeros from masked values
            final_probs = torch.where(
                final_probs > 0,
                final_probs,
                torch.zeros_like(final_probs)
            )
            
            # Renormalize if necessary
            if final_probs.sum() == 0:
                final_probs = torch.ones_like(final_probs) / final_probs.size(0)
            else:
                final_probs = final_probs / final_probs.sum()
            
            # Sample token
            sampled_token = torch.multinomial(final_probs, 1)
            
            # Calculate entropy for logging
            entropy_score = self.calculate_entropy(final_probs.unsqueeze(0))
            
            # Return sampled token and parameters
            return sampled_token.view(1, 1), {
                'alpha': float(alpha),
                'temperature': float(params['temperature']),
                'top_k': int(params['top_k']),
                'top_p': float(params['top_p']),
                'rep_penalty': float(params['rep_penalty']),
                'candidate_size': int(params['candidate_size']),
                'entropy_score': float(entropy_score.item() if isinstance(entropy_score, torch.Tensor) else entropy_score)
            }

        except Exception as e:
            self.logger.error(f"Error during sampling: {str(e)}")
            raise

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 2000):
    """Generate response using the enhanced sampler."""
    try:
        cfg = SamplerConfig()
        sampler = AdaptiveEntropixSampler(cfg)

        # Encode and validate input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        if input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("Empty input sequence")
            
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
                
                # Validate model outputs
                if not hasattr(outputs, 'logits'):
                    raise ValueError("Model output missing logits")
                    
                logits = outputs.logits[:, -1:, :].to(torch.float32)
                
                # Handle attention tensor
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    attention = outputs.attentions[-1].to(device)
                else:
                    # Create dummy attention if not available
                    seq_len = input_ids.size(1)
                    attention = torch.ones(1, 1, seq_len, seq_len, device=device)
                
                # Validate tensor dimensions
                if logits.dim() != 3:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
                if attention.dim() not in [3, 4]:
                    raise ValueError(f"Unexpected attention shape: {attention.shape}")
                
                # Sample next token
                sampled_token, params = sampler.sample(logits, attention, input_ids)
                
                # Check for EOS token
                if sampled_token.item() == tokenizer.eos_token_id:
                    break

                # Decode and append token
                next_token_text = tokenizer.decode([sampled_token.item()])
                generated_text += next_token_text
                print(next_token_text, end="", flush=True)

                # Update input tensors
                input_ids = torch.cat([input_ids, sampled_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=device)
                ], dim=1)

                # Log progress
                if _ % 10 == 0:
                    logger.debug(
                        f"Step {_}: alpha={params['alpha']:.3f}, "
                        f"temp={params['temperature']:.3f}, "
                        f"top_k={params['top_k']}, "
                        f"top_p={params['top_p']:.3f}, "
                        f"rep_penalty={params['rep_penalty']:.3f}, "
                        f"candidate_size={params['candidate_size']}"
                    )

            except Exception as e:
                logger.error(f"Error during generation step {_}: {str(e)}")
                logger.error(f"Current shapes - input_ids: {input_ids.shape}, logits: {logits.shape}")
                break

        print("\n")
        return generated_text

    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise

def main():
    """Main function to run the enhanced sampler."""
    try:
        if sys.platform == 'win32':
            _ = system('cls')
        else:
            _ = system('clear')

        print("Enhanced EntropixSampler with Multi-Parameter Optimization")
        print("------------------------------------------------------")
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading model {model_name}...")
        
        # Initialize model with proper configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            return_dict_in_generate=True,
            attn_implementation="eager",
            trust_remote_code=True  # Required for some models
        ).to(device)
        
        # Initialize tokenizer with proper configuration
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True  # Required for some models
        )
        
        # Ensure required tokens are set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded successfully!")
        print("\nType your prompts below ('quit' to exit)")
        print("----------------------------------------")

        while True:
            try:
                prompt = input("\nPrompt> ")
                if prompt.lower() == 'quit':
                    break
                
                if not prompt.strip():
                    print("Please enter a non-empty prompt")
                    continue

                response = generate_response(model, tokenizer, prompt)
                if response:
                    print(f"\nFinal response: {response}")
                else:
                    print("\nNo response generated. Please try again with a different prompt.")
                    
            except KeyboardInterrupt:
                print("\nGeneration interrupted by user")
                continue
            except Exception as e:
                logger.error(f"Error in prompt processing: {str(e)}")
                print("\nAn error occurred. Please try again with a different prompt.")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print("\nFatal error occurred. Please restart the application.")
    
    finally:
        print("\nThank you for using Enhanced EntropixSampler!")

if __name__ == "__main__":
    from os import system
    main()