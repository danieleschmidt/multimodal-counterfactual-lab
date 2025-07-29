"""Command-line interface for Multimodal Counterfactual Lab."""

import click
from pathlib import Path


@click.group()
@click.version_option()
def main():
    """Multimodal Counterfactual Lab CLI."""
    pass


@main.command()
@click.option("--input", "-i", type=click.Path(exists=True), help="Input image path")
@click.option("--text", "-t", type=str, help="Input text description")
@click.option("--method", "-m", default="modicf", help="Generation method")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
def generate(input, text, method, output):
    """Generate counterfactual examples."""
    click.echo(f"Generating counterfactuals using {method} method...")
    # Implementation would go here


@main.command()
@click.option("--model", "-m", type=str, help="Model to evaluate")
@click.option("--data", "-d", type=click.Path(exists=True), help="Counterfactual dataset")
@click.option("--output", "-o", type=click.Path(), help="Report output path")
def evaluate(model, data, output):
    """Evaluate model bias using counterfactuals."""
    click.echo(f"Evaluating bias in {model}...")
    # Implementation would go here


@main.command()
def web():
    """Launch the web interface."""
    click.echo("Starting Streamlit web interface...")
    # Would start Streamlit app


if __name__ == "__main__":
    main()