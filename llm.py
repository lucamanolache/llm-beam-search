import torch
from typing import Dict, List
from transformers import AutoModel, AutoTokenizer, Pipeline, pipeline
from transformers.integrations.ggml import processors

SYSTEM_PROMPT = """
Solve the following math problem efficiently and clearly:

    - For simple problems (2 steps or fewer):
    Provide a concise solution with minimal explanation.

    - For complex problems (3 steps or more):
    Use this step-by-step format:

    ## Step 1: [Concise description]
    [Brief explanation and calculations]

    ## Step 2: [Concise description]
    [Brief explanation and calculations]

    ...


    Regardless of the approach, always conclude with:

    Therefore, the final answer is: $\boxed{answer}$. I hope it is correct.

    Where [answer] is just the final number or expression that solves the problem.
"""

def generate_part(pipeline: Pipeline, previous: List[Dict], branches: int, max_tokens: int = 250) -> List[str]:
    """
    Generate multiple possible continuations from the previous text

    Args:
        pipeline: HuggingFace pipeline for text generation
        previous: Previous text to continue from
        branches: Number of different continuations to generate
        max_tokens: Maximum number of new tokens to generate

    Returns:
        List of generated continuations
    """

    # Configure generation parameters for branching
    generation_config = {
        "max_new_tokens": max_tokens,
        "num_return_sequences": branches,
        "do_sample": True,
        "temperature": 0.7,  # Control randomness
        "top_p": 0.95,      # Nucleus sampling
        "pad_token_id": pipeline.tokenizer.pad_token_id,
        "eos_token_id": pipeline.tokenizer.eos_token_id,
    }

    # Generate multiple continuations
    outputs = pipeline(
        previous,
        **generation_config,
        return_full_text=False,  # Only return the new generated text
        continue_final_message=True,
    )

    # Process outputs to stop at newlines
    processed_outputs = []
    for output in outputs:
        generated_text = output["generated_text"]

        if "## Step" in generated_text:
            step = generated_text.split("##")[1]
            next = previous.copy()
            next[-1]["content"] += f"##{step}"
            processed_outputs.append((next, False))
            continue
        else:
            next = previous.copy()
            next[-1]["content"] += generated_text
            processed_outputs.append((next, True))
            continue

    return processed_outputs

model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_pipeline = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
