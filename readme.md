# AlphaSampler

AlphaSampler is an advanced text generation sampling system that implements a novel approach combining thermodynamic sampling principles with speculative decoding. It dynamically adapts sampling parameters based on the model's uncertainty levels and attention coherence patterns to produce higher quality text generation.

## Core Concept

The system operates on the principle that language model uncertainty can be quantified and used to optimize the sampling strategy. It does this by:

1. Analyzing token distribution entropy
2. Evaluating attention pattern coherence
3. Combining these metrics to determine the system's thermodynamic state
4. Adaptively adjusting sampling parameters based on the current state

## Technical Architecture

### 1. Uncertainty Quantification

The system uses two primary metrics to quantify uncertainty:

#### Token Distribution Entropy
```python
token_entropy = -Σ(p_i * log_2(p_i))
```
- Calculated from the logits of the next token distribution
- Normalized by maximum possible entropy
- Indicates how "confident" the model is in its next token predictions

#### Attention Coherence
```python
attention_entropy = -Σ(a_i * log_2(a_i))
```
- Derived from the model's attention patterns
- Measures how focused vs. dispersed the model's attention is
- Higher values indicate more uncertain or complex relationships

### 2. Thermodynamic States

The system operates in three distinct states:

#### COLD State (α < 0.6)
- High model certainty
- Uses speculative decoding
- Narrow sampling distribution
- Parameters:
  - Low temperature (≈ 0.1-0.3)
  - Low top-k (10-30)
  - High precision sampling

#### HOT State (α > 1.8)
- High uncertainty
- Broader sampling distribution
- Parameters:
  - High temperature (≈ 0.8-1.3)
  - High top-k (150-200)
  - More exploratory sampling

#### MIXED State (0.6 ≤ α ≤ 1.8)
- Intermediate uncertainty
- Balanced parameter settings
- Parameters:
  - Moderate temperature (≈ 0.4-0.7)
  - Medium top-k (40-150)
  - Balanced sampling approach

### 3. Adaptive Parameter Calculation

Each sampling parameter is calculated using specialized functions:

#### Temperature
```python
temperature = 0.1 + 1.2 / (1 + exp(-1.5 * (α - 1.0)))
```
- Ranges from 0.1 to 1.3
- Sigmoid curve centered at α = 1.0
- Steeper transition in the MIXED state

#### Top-k
```python
top_k = 10 + 190 * (arctan(α - 1.0) / π + 0.5)
```
- Ranges from 10 to 200
- Uses arctangent function for smooth transitions
- Centered around α = 1.0

#### Top-p
```python
top_p = 0.85 + 0.1 * (2 / (1 + exp(-0.8 * α)) - 1)
```
- Ranges from 0.85 to 0.95
- Sigmoid-based scaling
- More conservative than traditional top-p

#### Repetition Penalty
```python
rep_penalty = 1.0 + 0.3 / (1 + exp(-2 * (α - 1.0)))
```
- Ranges from 1.0 to 1.3
- Stronger penalty in high uncertainty states
- Applies to previous 128 tokens

### 4. Speculative Decoding

In COLD states, the system implements speculative decoding:

1. Selects top-k candidate tokens
2. For each candidate:
   - Generates next token probability distribution
   - Calculates attention entropy
   - Computes variance in attention patterns
3. Scores candidates using:
   ```python
   score = entropy_weight * avg_entropy + var_entropy_weight * avg_varentropy
   ```
4. Selects token minimizing the score

## Implementation Details

### Main Components

#### SamplerConfig
```python
class SamplerConfig:
    def __init__(self):
        self.base_temp = 0.4
        self.base_top_p = 0.85
        self.base_top_k = 40
        self.base_rep_penalty = 1.0
        self.COLD_POINT = 0.6
        self.HOT_POINT = 1.8
```

#### AdaptiveEntropixSampler
Core class implementing:
- Entropy calculations
- State determination
- Parameter adaptation
- Sampling logic
- Speculative decoding

### Usage Example

```python
# Initialize
config = SamplerConfig()
sampler = AdaptiveEntropixSampler(config)

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Generate text
def generate_response(prompt, max_tokens=500):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    for _ in range(max_tokens):
        outputs = model(input_ids=input_ids, output_attentions=True)
        sampled_token, params = sampler.sample(
            outputs.logits,
            outputs.attentions[-1],
            input_ids
        )
        input_ids = torch.cat([input_ids, sampled_token], dim=1)
    
    return tokenizer.decode(input_ids[0])
```

## Performance Optimization

### Memory Management
- Uses rolling window for alpha history (100 tokens)
- Limits attention analysis to last layer
- Implements efficient tensor operations

### Computational Efficiency
- Caches intermediate calculations
- Vectorized operations for entropy calculation
- Optimized repetition penalty application

## System Requirements

### Hardware
- CUDA-capable GPU recommended
- Minimum 8GB GPU memory for base models
- 16GB+ recommended for larger models

### Software
- Python 3.7+
- PyTorch 2.0+
- Transformers library
- CUDA toolkit (for GPU support)

## Installation

```bash
# Basic installation
pip install torch transformers

# For CUDA support
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

## Limitations and Considerations

1. Computational Overhead
   - Speculative decoding increases computation time
   - Multiple forward passes in COLD state
   - Memory usage scales with sequence length

2. Model Compatibility
   - Requires attention pattern output
   - Works best with transformer-based models
   - May need parameter tuning for different model sizes

3. Performance Tradeoffs
   - Speed vs. quality tradeoff in speculative decoding
   - Memory usage vs. history length
   - Precision vs. creativity balance

## License

This project is available under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.
