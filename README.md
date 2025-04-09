# Fine-Tuning Gemma-2B for Math Word Problems

This repository contains code for fine-tuning the Gemma-2B language model from Google on math word problems using the Microsoft Orca Math Word Problems dataset. The implementation uses 4-bit quantization and LoRA (Low-Rank Adaptation) for efficient training on consumer hardware.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training Process](#training-process)
- [Results](#results)
- [Model Comparison](#model-comparison)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project demonstrates how to fine-tune the Gemma-2B model specifically for solving math word problems. The model is adapted using:
- 4-bit quantization (via BitsAndBytes)
- LoRA (Low-Rank Adaptation)
- The Microsoft Orca Math Word Problems dataset (200k examples)

The fine-tuned model shows improved performance on math problems compared to the base Gemma-2B model.

## Features

- Efficient training with 4-bit quantization
- LoRA adaptation targeting key model projections
- Integration with Weights & Biases for training monitoring
- Example comparison between base and fine-tuned models
- Only 0.64% of model parameters are trainable (9.8M out of 1.53B)

## Requirements

To run this project, you'll need:

- Python 3.8+
- PyTorch (with CUDA if using GPU)
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes
- TRL (Transformer Reinforcement Learning)
- Weights & Biases (for logging)
- A Hugging Face account and access token (for Gemma model access)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gemma-math-finetuning.git
   cd gemma-math-finetuning
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install torch transformers datasets peft bitsandbytes trl wandb
   ```

4. Authenticate with Hugging Face:
   ```bash
   huggingface-cli login
   ```
   (Enter your Hugging Face access token when prompted)

## Usage

### Training the Model

The main training process is in the Jupyter notebook `gemma.ipynb`. Key steps include:

1. Loading the Gemma-2B model with 4-bit quantization
2. Setting up LoRA configuration
3. Loading the Microsoft Orca Math Word Problems dataset
4. Configuring the SFTTrainer
5. Running the training process

### Evaluating the Model

After training, you can compare the original and fine-tuned models:

```python
# Load models
original_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0})
fine_tuned_model = AutoModelForCausalLM.from_pretrained("fine_tuned_gemma_math", quantization_config=bnb_config, device_map={"":0})

# Generate responses
math_problem = "What is 125 * 45 + 678? Show your work."
print("=== ORIGINAL MODEL ===")
print(generate_response(original_model, math_problem))
print("\n=== FINE-TUNED MODEL ===")
print(generate_response(fine_tuned_model, math_problem))
```

## Training Process

### Configuration

- **Model**: google/gemma-2b
- **Quantization**: 4-bit NF4 with double quantization
- **LoRA**:
  - Rank (r): 8
  - Target modules: q_proj, o_proj, k_proj, v_proj, gate_proj, up_proj, down_proj
- **Training**:
  - Batch size: 1 (per device)
  - Gradient accumulation: 4 steps
  - Total steps: 500
  - Learning rate: 2e-4
  - Optimizer: paged_adamw_8bit

### Training Statistics

- Trainable parameters: 9,805,824 (0.64% of total)
- Training loss decreased from ~1.90 to ~1.50
- Training time: ~8 minutes 40 seconds for 500 steps

## Results

The fine-tuned model demonstrates improved performance on math problems. For example:

**Original Model Output:**
```
What is 125 * 45 + 678? Show your work.

Answer:

Step 1/2
First, we need to multiply 125 by 45. To do this, we can use the traditional method of multiplying each digit of 125 by the corresponding digit of 45. 1 x 4 = 4 2 x 5 = 10 5 x 5 = 25 1 x 4 = 4 2 x 5 = 10 5 x 5 = 25...
```

**Fine-Tuned Model Output:**
```
What is 125 * 45 + 678? Show your work.

Answer:

Step 1/2
First, we need to multiply 125 by 45: 125 * 45 = 5625

Step 2/2
Next, we need to add 678 to the result: 5625 + 678 = 6303 Therefore, the answer is 6303.
```

The fine-tuned model provides more concise and accurate solutions to math problems.

## Model Comparison

| Metric               | Original Model | Fine-Tuned Model |
|----------------------|----------------|------------------|
| Math Accuracy        | Low            | Improved         |
| Solution Clarity     | Verbose        | Concise          |
| Step-by-Step Logic   | Less organized | More structured  |
| Computational Cost   | Same           | Same             |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google for the Gemma model
- Microsoft for the Orca Math Word Problems dataset
- Hugging Face for the Transformers library and PEFT implementation
- Weights & Biases for training monitoring
