import os
import re
from heapq import nlargest
from typing import Any, Callable, List
import sympy
from sympy.parsing.latex import parse_latex
import antlr4

import polars as pl
import torch
import torch.nn.functional as F
from rich import print
from transformers import AutoModel, AutoTokenizer, pipeline

from llm import *
from prm import *
from beam import *

df = None
if not os.path.exists("data.csv"):
    df = pl.read_ndjson("hf://datasets/DongfuJiang/MATH-500/test.jsonl")
    df.write_json("data.csv")
else:
    df = pl.read_json("data.csv")

print(df[0])

problem = df["problem"][0]

initial_state = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": problem},
    {"role": "assistant", "content": ""}
]

def branch_fn(state, branching_factor):
    return generate_part(model_pipeline, state, branching_factor)

result = beam_search(
    initial_state=initial_state,
    beam_width=3,
    branching_factor=2,
    evaluate=evaluate,
    branch=branch_fn,
    num_terminals=5,
    max_steps=10,
    verbose=False
)

print(result[-1]["content"])

pattern = r'The final answer is: \$\\boxed\{(.*)\}\$'
matches = re.findall(pattern, result[-1]["content"])

if matches:
    # Find the longest answer
    longest_answer = max(matches, key=len)
    print("Longest answer:", longest_answer)

    # Get ground truth
    ground_truth = df["answer"][0]
    print("Ground truth:", ground_truth)

    try:
        # Parse both latex expressions
        parsed_answer = parse_latex(longest_answer)
        parsed_truth = parse_latex(ground_truth)

        # Calculate difference
        difference = sympy.simplify(parsed_answer - parsed_truth)

        print("Difference:", difference)
        print("Correct:" if difference == 0 else "Incorrect")

    except Exception as e:
        print("Error parsing latex expressions:", e)
