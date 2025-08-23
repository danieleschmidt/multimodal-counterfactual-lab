"""Command-line interface for Multimodal Counterfactual Lab."""

import click
import json
from pathlib import Path
from PIL import Image
import logging
import sys
import time
from datetime import datetime

from counterfactual_lab.core import CounterfactualGenerator, BiasEvaluator

# Try to import Generation 5 NACS-CF
try:
    from counterfactual_lab.generation_5_breakthrough import (
        NeuromorphicAdaptiveCounterfactualSynthesis,
        demonstrate_nacs_cf_breakthrough
    )
    NACS_CF_AVAILABLE = True
except ImportError:
    NACS_CF_AVAILABLE = False

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
@click.option("--method", "-m", default="modicf", type=click.Choice(["modicf", "icg", "nacs-cf"]), help="Generation method")
@click.option("--attributes", "-a", default="gender,race,age", help="Comma-separated attributes to modify")
@click.option("--samples", "-s", default=5, type=int, help="Number of samples to generate")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--device", default="auto", help="Compute device (cuda/cpu/auto)")
@click.option("--consciousness", "-c", is_flag=True, help="Enable consciousness-guided generation (NACS-CF)")
@click.option("--quantum", "-q", is_flag=True, help="Enable quantum entanglement effects (NACS-CF)")
def generate(input, text, method, attributes, samples, output, device, consciousness, quantum):
    """Generate counterfactual examples."""
    
    # Check NACS-CF availability
    if method == "nacs-cf" and not NACS_CF_AVAILABLE:
        click.echo("❌ NACS-CF not available. Missing advanced dependencies.", err=True)
        raise click.Abort()
    
    # Show advanced features notice
    if method == "nacs-cf":
        click.echo("🧠 Using Generation 5 NACS-CF: Neuromorphic Adaptive Counterfactual Synthesis")
        click.echo("   Features: Consciousness • Quantum Entanglement • Holographic Memory")
    
    click.echo(f"🔄 Generating {samples} counterfactuals using {method} method...")
    
    try:
        # Parse attributes
        attr_list = [attr.strip() for attr in attributes.split(",")]
        target_attributes = {attr: "varied" for attr in attr_list}  # NACS-CF expects dict format
        
        # Load image
        image = Image.open(input).convert("RGB")
        
        if method == "nacs-cf" and NACS_CF_AVAILABLE:
            # Use Generation 5 NACS-CF
            mock_image = image  # Would need actual PIL Image handling in full implementation
            
            # Initialize NACS-CF system
            nacs_cf = NeuromorphicAdaptiveCounterfactualSynthesis(
                consciousness_threshold=0.7,
                quantum_coherence_time=10.0,
                memory_dimensions=256,  # Reduced for CLI performance
                adaptive_topology=True
            )
            
            # Generate using consciousness and quantum features
            results = nacs_cf.generate_neuromorphic_counterfactuals(
                image=mock_image,
                text=text,
                target_attributes=target_attributes,
                num_samples=samples,
                consciousness_guidance=consciousness,
                quantum_entanglement=quantum
            )
            
            # Display NACS-CF specific metrics
            click.echo(f"🧠 Consciousness coherence: {results['neuromorphic_metrics']['consciousness_coherence']:.3f}")
            click.echo(f"⚛️ Quantum entanglement fidelity: {results['neuromorphic_metrics']['quantum_entanglement_fidelity']:.3f}")
            click.echo(f"🔮 Holographic memory efficiency: {results['neuromorphic_metrics']['holographic_memory_efficiency']:.3f}")
            
        else:
            # Use traditional methods
            generator = CounterfactualGenerator(method=method, device=device)
            results = generator.generate(image, text, attr_list, samples)
        
        # Save results
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
            
            # Save comprehensive results for NACS-CF
            if method == "nacs-cf" and NACS_CF_AVAILABLE:
                results_path = output_dir / "nacs_cf_results.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                click.echo(f"🧠 NACS-CF results saved to {results_path}")
            else:
                # Traditional visualization
                viz_path = output_dir / "counterfactuals_grid.png"
                generator.visualize_grid(results, str(viz_path))
            
            # Save metadata
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                if method == "nacs-cf":
                    metadata = results.get("generation_metadata", {})
                    metadata["neuromorphic_metrics"] = results.get("neuromorphic_metrics", {})
                else:
                    metadata = {k: v for k, v in results.items() if k != "counterfactuals"}
                    metadata["num_counterfactuals"] = len(results["counterfactuals"])
                json.dump(metadata, f, indent=2, default=str)
            
            click.echo(f"✅ Results saved to {output_dir}")
        else:
            click.echo(f"✅ Generated {len(results['counterfactuals'])} counterfactuals")
            if method == "nacs-cf":
                click.echo(f"📊 Generation time: {results['generation_metadata']['generation_time']:.2f}s")
            else:
                click.echo(f"📊 Generation time: {results['metadata']['generation_time']:.2f}s")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        logger.exception("Generation failed")
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


@main.command()
@click.option("--output", "-o", type=click.Path(), help="Output directory for demonstration results")
def demo_nacs_cf(output):
    """Demonstrate Generation 5 NACS-CF breakthrough capabilities."""
    if not NACS_CF_AVAILABLE:
        click.echo("❌ NACS-CF not available. Missing advanced dependencies.", err=True)
        raise click.Abort()
    
    click.echo("🚀 NACS-CF BREAKTHROUGH DEMONSTRATION")
    click.echo("=" * 60)
    
    try:
        # Run the breakthrough demonstration
        nacs_cf_system, results = demonstrate_nacs_cf_breakthrough()
        
        # Save results if output specified
        if output:
            output_dir = Path(output)
            output_dir.mkdir(exist_ok=True)
            
            results_file = output_dir / "nacs_cf_demo_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            click.echo(f"💾 Demo results saved to {results_file}")
        
        click.echo("🎉 NACS-CF demonstration completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Demo failed: {str(e)}", err=True)
        logger.exception("NACS-CF demo failed")
        raise click.Abort()


@main.command()
@click.option("--format", "-f", default="json", type=click.Choice(["json", "yaml", "table"]), help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def status(format, output):
    """Show system status and capabilities."""
    click.echo("📊 MULTIMODAL COUNTERFACTUAL LAB - SYSTEM STATUS")
    click.echo("=" * 60)
    
    try:
        # Gather system information
        system_status = {
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0",
            "capabilities": {
                "traditional_methods": ["modicf", "icg"],
                "advanced_methods": ["nacs-cf"] if NACS_CF_AVAILABLE else [],
                "generation_5_breakthrough": NACS_CF_AVAILABLE,
                "consciousness_inspired_fairness": NACS_CF_AVAILABLE,
                "quantum_entanglement_simulation": NACS_CF_AVAILABLE,
                "holographic_memory_system": NACS_CF_AVAILABLE
            },
            "features": {
                "cli_interface": True,
                "web_interface": True,
                "bias_evaluation": True,
                "report_generation": True,
                "batch_processing": True
            }
        }
        
        # Add NACS-CF system status if available
        if NACS_CF_AVAILABLE:
            try:
                nacs_cf = NeuromorphicAdaptiveCounterfactualSynthesis()
                nacs_status = nacs_cf.get_comprehensive_system_status()
                system_status["nacs_cf_status"] = {
                    "consciousness_level": nacs_status["consciousness_state"]["ethical_reasoning_level"],
                    "generation_history": nacs_status["generation_history_length"],
                    "system_health": nacs_status["system_health"]
                }
            except Exception as e:
                system_status["nacs_cf_status"] = {"error": str(e)}
        
        # Format and display output
        if format == "json":
            output_text = json.dumps(system_status, indent=2, default=str)
            click.echo(output_text)
        
        elif format == "yaml":
            try:
                import yaml
                output_text = yaml.dump(system_status, default_flow_style=False)
                click.echo(output_text)
            except ImportError:
                click.echo("❌ YAML format requires PyYAML. Using JSON instead.")
                output_text = json.dumps(system_status, indent=2, default=str)
                click.echo(output_text)
        
        elif format == "table":
            click.echo(f"Timestamp: {system_status['timestamp']}")
            click.echo(f"Version: {system_status['version']}")
            click.echo(f"Generation 5 NACS-CF: {'✅ Available' if NACS_CF_AVAILABLE else '❌ Not Available'}")
            
            click.echo("\nCapabilities:")
            for category, items in system_status["capabilities"].items():
                if isinstance(items, list):
                    click.echo(f"  {category}: {', '.join(items) if items else 'None'}")
                else:
                    status_icon = "✅" if items else "❌"
                    click.echo(f"  {category}: {status_icon}")
            
            click.echo("\nFeatures:")
            for feature, available in system_status["features"].items():
                status_icon = "✅" if available else "❌"
                click.echo(f"  {feature}: {status_icon}")
            
            if NACS_CF_AVAILABLE and "nacs_cf_status" in system_status:
                nacs_status = system_status["nacs_cf_status"]
                if "error" not in nacs_status:
                    click.echo(f"\nNACS-CF System:")
                    click.echo(f"  Consciousness Level: {nacs_status['consciousness_level']:.3f}")
                    click.echo(f"  Generation History: {nacs_status['generation_history']} generations")
                    
                    health = nacs_status['system_health']
                    click.echo(f"  System Health:")
                    click.echo(f"    Consciousness Coherence: {health['consciousness_coherence']:.3f}")
                    click.echo(f"    Memory Efficiency: {health['memory_efficiency']:.3f}")
                    click.echo(f"    Quantum Coherence: {'✅' if health['quantum_coherence_active'] else '❌'}")
        
        # Save output if specified
        if output:
            with open(output, 'w') as f:
                if format == "json":
                    json.dump(system_status, f, indent=2, default=str)
                elif format == "yaml":
                    try:
                        import yaml
                        yaml.dump(system_status, f, default_flow_style=False)
                    except ImportError:
                        json.dump(system_status, f, indent=2, default=str)
                else:
                    f.write(str(system_status))
            
            click.echo(f"💾 Status saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Error getting system status: {str(e)}", err=True)
        logger.exception("Status command failed")
        raise click.Abort()


@main.command()
def version():
    """Show version and build information."""
    click.echo("🧪 Multimodal Counterfactual Lab")
    click.echo(f"Version: 0.1.0")
    click.echo(f"Generation 5 NACS-CF: {'✅ Available' if NACS_CF_AVAILABLE else '❌ Not Available'}")
    click.echo(f"Build Date: {datetime.now().strftime('%Y-%m-%d')}")
    click.echo("Author: Daniel Schmidt <daniel@terragon.ai>")
    click.echo("Repository: https://github.com/terragon-labs/multimodal-counterfactual-lab")


if __name__ == "__main__":
    main()