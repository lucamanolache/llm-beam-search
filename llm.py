import torch
from transformers import AutoModel, AutoTokenizer, Pipeline, pipeline

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

def generate_part(pipeline: Pipeline, previous=List[str], branches: int, max_tokens=100) -> List[List[str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": df["problem"][0]},
        {"role": "system", "content":}
    ]
    outputs = pipe(
        messages,
        max_new_tokens=512,
        early_stopping=True,
        num_beam_groups=branches,
    )

model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_pipeline = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
