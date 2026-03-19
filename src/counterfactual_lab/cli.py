"""Command-line interface for multimodal-counterfactual-lab."""

from __future__ import annotations

import json
import sys
import logging

import click

from counterfactual_lab import CounterfactualGenerator


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def main():
    """Multimodal Counterfactual Lab — generate & evaluate text-side counterfactuals."""
    pass


@main.command("generate")
@click.option("--text", "-t", required=True, help="Input caption / description.")
@click.option(
    "--attributes",
    "-a",
    default="gender",
    show_default=True,
    help="Comma-separated protected attributes to perturb.",
)
@click.option(
    "--output",
    "-o",
    default="-",
    help="Output file path (default: stdout).",
)
def cmd_generate(text: str, attributes: str, output: str):
    """Generate counterfactual pairs for a text input."""
    attr_list = [a.strip() for a in attributes.split(",") if a.strip()]
    gen = CounterfactualGenerator(attributes=attr_list)
    result = gen.generate(text)

    data = {
        "original_text": result.original_text,
        "n_pairs": result.n_pairs,
        "metadata": result.metadata,
        "pairs": [
            {
                "strategy": p.perturbation.strategy,
                "attribute": p.perturbation.attribute,
                "direction": p.perturbation.direction,
                "original_text": p.original_text,
                "counterfactual_text": p.counterfactual_text,
                "changed_tokens": p.perturbation.changed_tokens,
                "edit_distance_words": p.perturbation.edit_distance_words,
            }
            for p in result.pairs
        ],
    }

    out_str = json.dumps(data, indent=2)
    if output == "-":
        click.echo(out_str)
    else:
        with open(output, "w") as f:
            f.write(out_str)
        click.echo(f"Saved to {output}", err=True)


if __name__ == "__main__":
    main()
