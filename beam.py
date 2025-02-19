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
    branch: Callable[[Any, int], List[tuple[str, bool]]],
    num_terminals: int,
    max_steps: int = 100,
    verbose: bool = False,
) -> Any:
    """
    Perform beam search with branch function returning (string, is_terminal) tuples.

    Parameters:
    - initial_state: The starting state of the search.
    - beam_width: The number of states to keep at each step.
    - branching_factor: The number of branches to generate per state.
    - evaluate: A function that assigns a score to a state.
    - branch: A function that returns list of (string, is_terminal) tuples.
    - num_terminals: The number of terminal states to collect before returning the best one.
    - max_steps: Maximum number of steps to run the search.
    - verbose: Whether to print detailed logging information.

    Returns:
    - The best final state found among collected terminal states.
    """
    beam = [initial_state]  # Initialize beam with the initial state
    terminal_states = []

    if verbose:
        print(f"\n[bold blue]Starting beam search:[/bold blue]")
        print(f"Initial state: {initial_state}")
        print(f"Beam width: {beam_width}")
        print(f"Branching factor: {branching_factor}")

    for step in range(max_steps):
        if verbose:
            print(f"\n[bold green]Step {step + 1}:[/bold green]")
            print(f"Current beam size: {len(beam)}")

        candidates = []
        for state in beam:
            # Get next states and their terminal status
            next_states = branch(state, branching_factor)

            if verbose:
                print(f"\nBranching from state: {state}")
                print(f"Generated {len(next_states)} new states")

            for new_state, is_terminal in next_states:
                if is_terminal:
                    terminal_states.append(new_state)
                    if verbose:
                        print(f"Found terminal state: {new_state}")
                        print(f"Score: {evaluate(new_state)}")

                    if len(terminal_states) >= num_terminals:
                        best_terminal = max(terminal_states, key=evaluate)
                        if verbose:
                            print(f"\n[bold red]Search complete![/bold red]")
                            print(f"Best terminal state found: {best_terminal}")
                            print(f"Final score: {evaluate(best_terminal)}")
                        return best_terminal

                candidates.append(new_state)

        if not candidates:
            if verbose:
                print("\n[bold red]Search ended: No more candidates[/bold red]")
            break

        # Keep top states
        beam = nlargest(beam_width, candidates, key=evaluate)

        if verbose:
            print(f"\nAfter pruning:")
            print(f"New beam size: {len(beam)}")
            print("Top 3 states in beam:")
            for i, state in enumerate(beam[:3], 1):
                print(f"{i}. {state} (score: {evaluate(state)})")

    # Return the best state found
    final_states = beam + terminal_states
    best_state = max(final_states, key=evaluate) if final_states else max(beam, key=evaluate)

    if verbose:
        print(f"\n[bold red]Search complete![/bold red]")
        print(f"Total terminal states found: {len(terminal_states)}")
        print(f"Best state found: {best_state}")
        print(f"Final score: {evaluate(best_state)}")

    return best_state
