import os
import re
from heapq import nlargest
from typing import Any, Callable, List

import polars as pl
import torch
import torch.nn.functional as F
from rich import print
from transformers import AutoModel, AutoTokenizer, pipeline

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
