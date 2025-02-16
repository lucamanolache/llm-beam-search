import os
import re
from heapq import nlargest
from typing import Any, Callable, List

import polars as pl
import torch
import torch.nn.functional as F
from rich import print
from transformers import AutoModel, AutoTokenizer, pipeline


def beam_search(
    initial_state: Any,
    beam_width: int,
    branching_factor: int,
    evaluate: Callable[[Any], float],
    branch: Callable[[Any, int], List[Any]],
    is_terminal: Callable[[Any], bool],
    num_terminals: int,
    max_steps: int = 100,
) -> Any:
    """
    Perform beam search.

    Parameters:
    - initial_state: The starting state of the search.
    - beam_width: The number of states to keep at each step.
    - branching_factor: The number of branches to generate per state.
    - evaluate: A function that assigns a score to a state.
    - branch: A function that generates possible next states, given a branching factor.
    - is_terminal: A function that determines if a state is terminal.
    - num_terminals: The number of terminal states to collect before returning the best one.
    - max_steps: Maximum number of steps to run the search.

    Returns:
    - The best final state found among collected terminal states.
    """
    beam = [initial_state]  # Initialize beam with the initial state
    terminal_states = []

    for _ in range(max_steps):
        candidates = []
        for state in beam:
            if is_terminal(state):
                terminal_states.append(state)
                if len(terminal_states) >= num_terminals:
                    return max(
                        terminal_states, key=evaluate
                    )  # Return the best terminal state found
            candidates.extend(
                branch(state, branching_factor)
            )  # Generate possible next states

        if not candidates:
            break  # Stop if there are no more possible expansions

        beam = nlargest(beam_width, candidates, key=evaluate)  # Keep top states

    return (
        max(beam + terminal_states, key=evaluate)
        if terminal_states
        else max(beam, key=evaluate)
    )  # Return the best state found


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


del pipe

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
