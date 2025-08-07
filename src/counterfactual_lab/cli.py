"""Command-line interface for Multimodal Counterfactual Lab."""

import click
import json
from pathlib import Path
from PIL import Image
import logging

from counterfactual_lab.core import CounterfactualGenerator, BiasEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def main():
    """Multimodal Counterfactual Lab CLI."""
    pass


@main.command()
@click.option("--input", "-i", type=click.Path(exists=True), required=True, help="Input image path")
@click.option("--text", "-t", type=str, required=True, help="Input text description")
@click.option("--method", "-m", default="modicf", type=click.Choice(["modicf", "icg"]), help="Generation method")
@click.option("--attributes", "-a", default="gender,race,age", help="Comma-separated attributes to modify")
@click.option("--samples", "-s", default=5, type=int, help="Number of samples to generate")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--device", default="auto", help="Compute device (cuda/cpu/auto)")
def generate(input, text, method, attributes, samples, output, device):
    """Generate counterfactual examples."""
    click.echo(f"🔄 Generating {samples} counterfactuals using {method} method...")
    
    try:
        # Parse attributes
        attr_list = [attr.strip() for attr in attributes.split(",")]
        
        # Initialize generator
        generator = CounterfactualGenerator(method=method, device=device)
        
        # Load and generate
        image = Image.open(input).convert("RGB")
        results = generator.generate(image, text, attr_list, samples)
        
        # Save results
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
            
            # Save visualization
            viz_path = output_dir / "counterfactuals_grid.png"
            generator.visualize_grid(results, str(viz_path))
            
            # Save metadata
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                metadata = {k: v for k, v in results.items() if k != "counterfactuals"}
                metadata["num_counterfactuals"] = len(results["counterfactuals"])
                json.dump(metadata, f, indent=2, default=str)
            
            click.echo(f"✅ Results saved to {output_dir}")
        else:
            click.echo(f"✅ Generated {len(results['counterfactuals'])} counterfactuals")
            click.echo(f"📊 Generation time: {results['metadata']['generation_time']:.2f}s")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


@main.command()
@click.option("--counterfactuals", "-c", type=click.Path(exists=True), required=True, help="Counterfactual data (JSON)")
@click.option("--metrics", "-m", default="demographic_parity,equalized_odds,cits_score", help="Metrics to compute")
@click.option("--output", "-o", type=click.Path(), help="Report output path")
@click.option("--format", "-f", default="technical", type=click.Choice(["regulatory", "academic", "technical"]), help="Report format")
def evaluate(counterfactuals, metrics, output, format):
    """Evaluate model bias using counterfactuals."""
    click.echo(f"🔍 Evaluating bias with metrics: {metrics}")
    
    try:
        # Load counterfactuals
        with open(counterfactuals, 'r') as f:
            cf_data = json.load(f)
        
        # Mock model for evaluation (in real use, would load actual model)
        class MockModel:
            def __init__(self):
                self.name = "mock-vlm"
        
        mock_model = MockModel()
        
        # Parse metrics
        metric_list = [metric.strip() for metric in metrics.split(",")]
        
        # Initialize evaluator and run evaluation
        evaluator = BiasEvaluator(mock_model)
        results = evaluator.evaluate(cf_data, metric_list)
        
        # Generate report
        report = evaluator.generate_report(results, format=format, export_path=output)
        
        if output:
            click.echo(f"📄 Report saved to {output}")
        else:
            click.echo("📊 Evaluation Results:")
            click.echo(f"  Overall Fairness Score: {results['summary']['overall_fairness_score']:.3f}")
            click.echo(f"  Rating: {results['summary']['fairness_rating']}")
            click.echo(f"  Passed Metrics: {results['summary']['passed_metrics']}/{results['summary']['total_metrics']}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


@main.command()
@click.option("--port", "-p", default=8501, type=int, help="Port for web interface")
@click.option("--host", "-h", default="localhost", help="Host for web interface")
def web(port, host):
    """Launch the web interface."""
    click.echo(f"🚀 Starting Streamlit web interface on {host}:{port}...")
    try:
        import subprocess
        import sys
        
        # Create minimal Streamlit app if it doesn't exist
        app_path = Path("streamlit_app.py")
        if not app_path.exists():
            app_content = '''
import streamlit as st
from counterfactual_lab import CounterfactualGenerator
from PIL import Image

st.title("🧪 Multimodal Counterfactual Lab")

st.sidebar.header("Configuration")
method = st.sidebar.selectbox("Method", ["modicf", "icg"])
attributes = st.sidebar.multiselect("Attributes", ["gender", "race", "age"], default=["gender"])
samples = st.sidebar.slider("Samples", 1, 10, 5)

st.header("Generate Counterfactuals")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
text_input = st.text_input("Text description")

if uploaded_file and text_input:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", width=300)
    
    if st.button("Generate"):
        with st.spinner("Generating counterfactuals..."):
            generator = CounterfactualGenerator(method=method)
            results = generator.generate(image, text_input, attributes, samples)
            
        st.success(f"Generated {len(results['counterfactuals'])} counterfactuals!")
        
        for i, cf in enumerate(results['counterfactuals']):
            st.subheader(f"Counterfactual {i+1}")
            st.image(cf['generated_image'], width=300)
            st.text(f"Attributes: {cf['target_attributes']}")
            if 'explanation' in cf:
                st.text(f"Explanation: {cf['explanation']}")
'''
            with open(app_path, 'w') as f:
                f.write(app_content)
        
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--server.address", host]
        subprocess.run(cmd)
        
    except ImportError:
        click.echo("❌ Streamlit not installed. Install with: pip install streamlit", err=True)
    except Exception as e:
        click.echo(f"❌ Error starting web interface: {str(e)}", err=True)


if __name__ == "__main__":
    main()