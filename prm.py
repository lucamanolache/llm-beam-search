import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, pipeline
from typing import List

model_name = "Qwen/Qwen2.5-Math-PRM-7B"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

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

def evaluate(messages: List[str]):
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

    return make_step_rewards(outputs[0], token_masks)[0][-1] # return the last step reward
