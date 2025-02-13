import os
import re

import polars as pl
import torch
import torch.nn.functional as F
from rich import print
from transformers import AutoModel, AutoTokenizer, pipeline

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

df = None
if not os.path.exists("data.csv"):
    df = pl.read_ndjson("hf://datasets/DongfuJiang/MATH-500/test.jsonl")
    df.write_json("data.csv")
else:
    df = pl.read_json("data.csv")
print(df.head())


def format_conversation(conversation: str):
    # Match steps using Markdown headers (## Step X:)
    step_pattern = re.compile(r"(## Step \d+:)(.*)", re.MULTILINE)

    # Split into lines and process
    lines = conversation.split("\n")
    for line in lines:
        step_match = step_pattern.match(line)
        if step_match:
            # Print step headers in bold red
            print(f"[bold red]{step_match.group(1)}[/bold red]{step_match.group(2)}")
        else:
            # Print normal lines
            print(line)


def split_conversation(conversation):
    content = conversation[-1]["content"]

    steps = []
    final_answer = None
    current_step = []
    lines = content.split("\n")

    for line in lines:
        if "Therefore, the final answer is:" in line:
            if current_step:
                steps.append("\n".join(current_step))
            final_answer = line.strip()
            break

        if line.startswith("##"):
            if current_step:
                steps.append("\n".join(current_step))
            current_step = []
        if line.strip():
            current_step.append(line.replace("##", "").strip())

    if current_step and "Therefore, the final answer is:" not in "\n".join(
        current_step
    ):
        steps.append("\n".join(current_step))

    data = {
        "system": conversation[0]["content"],
        "query": conversation[1]["content"],
        "response": steps + [final_answer] if final_answer else steps,
    }

    messages = [
        {"role": "system", "content": data["system"]},
        {"role": "user", "content": data["query"]},
    ]
    messages.append(
        {
            "role": "assistant",
            "content": "<extra_0>" + "<extra_0>".join(steps) + "<extra_0>",
        }
    )

    return messages


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[
            :, 1
        ]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


model_name = "Qwen/Qwen2.5-Math-PRM-7B"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": df["problem"][0]},
]
outputs = pipe(
    messages,
    max_new_tokens=512,
)
messages.append(outputs[0]["generated_text"][-1])
format_conversation(outputs[0]["generated_text"][-1]["content"])

print("SOLUTION\n", df["solution"][0])

print("====================")
analyze = split_conversation(messages)
conversation_str = tokenizer.apply_chat_template(
    analyze, tokenize=False, add_generation_prompt=False
)

input_ids = tokenizer.encode(
    conversation_str,
    return_tensors="pt",
).to(model.device)
outputs = model(input_ids=input_ids)

step_sep_id = tokenizer.encode("<extra_0>")[0]
token_masks = input_ids == step_sep_id
step_reward = make_step_rewards(outputs[0], token_masks)
print(step_reward)
print(analyze)
