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

def evaluate(conversation: List[str]):
    conversation_str = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )

    input_ids = tokenizer.encode(
        conversation_str,
        return_tensors="pt",
    ).to(model.device)
    outputs = model(input_ids=input_ids)

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = input_ids == step_sep_id

    return make_step_rewards(outputs[0], token_masks)[-1] # return the last step reward
