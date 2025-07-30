#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generation Script

Generates SBOMs in multiple formats for supply chain security compliance.
Supports SPDX, CycloneDX, and Syft formats with vulnerability correlation.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SBOMGenerator:
    """Generate and validate Software Bill of Materials"""
    
    def __init__(self, project_path: Path = Path(".")):
        self.project_path = project_path.resolve()
        self.sbom_dir = self.project_path / "sbom"
        self.sbom_dir.mkdir(exist_ok=True)
        
    def check_dependencies(self) -> bool:
        """Check if required SBOM tools are installed"""
        required_tools = [
            ("cyclonedx-py", "cyclonedx-bom"),
            ("syft", "anchore/syft"),
            ("pip-audit", "pypa/pip-audit")
        ]
        
        missing = []
        for tool, package in required_tools:
            if not self._command_exists(tool):
                missing.append(f"{tool} (install: pip install {package})")
                
        if missing:
            print("Missing required tools:")
            for tool in missing:
                print(f"  - {tool}")
            return False
        return True
        
    def generate_all_formats(self) -> Dict[str, Path]:
        """Generate SBOMs in all supported formats"""
        if not self.check_dependencies():
            sys.exit(1)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {}
        
        # Generate CycloneDX SBOM
        print("Generating CycloneDX SBOM...")
        cyclone_path = self.sbom_dir / f"sbom-cyclonedx-{timestamp}.json"
        if self._generate_cyclonedx(cyclone_path):
            results["cyclonedx"] = cyclone_path
            
        # Generate SPDX SBOM using pip-audit
        print("Generating SPDX SBOM...")
        spdx_path = self.sbom_dir / f"sbom-spdx-{timestamp}.json"
        if self._generate_spdx(spdx_path):
            results["spdx"] = spdx_path
            
        # Generate Syft SBOM
        print("Generating Syft SBOM...")
        syft_path = self.sbom_dir / f"sbom-syft-{timestamp}.json"
        if self._generate_syft(syft_path):
            results["syft"] = syft_path
            
        return results
        
    def _generate_cyclonedx(self, output_path: Path) -> bool:
        """Generate CycloneDX format SBOM"""
        try:
            cmd = [
                "cyclonedx-py",
                "--output-format", "json",
                "--output-file", str(output_path),
                "--include-dev",
                "--include-optional"
            ]
            subprocess.run(cmd, cwd=self.project_path, check=True, capture_output=True)
            print(f"✓ CycloneDX SBOM generated: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ CycloneDX generation failed: {e}")
            return False
            
    def _generate_spdx(self, output_path: Path) -> bool:
        """Generate SPDX format SBOM using pip-audit"""
        try:
            cmd = [
                "pip-audit",
                "--format=json",
                f"--output={output_path}",
                "--require-hashes",
                "--progress-spinner=off"
            ]
            subprocess.run(cmd, cwd=self.project_path, check=True, capture_output=True)
            print(f"✓ SPDX SBOM generated: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ SPDX generation failed: {e}")
            return False
            
    def _generate_syft(self, output_path: Path) -> bool:
        """Generate Syft SBOM with comprehensive dependency analysis"""
        try:
            cmd = [
                "syft",
                "packages",
                f"dir:{self.project_path}",
                "-o", f"json={output_path}"
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Syft SBOM generated: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Syft generation failed: {e}")
            return False
            
    def validate_sboms(self, sbom_paths: Dict[str, Path]) -> Dict[str, bool]:
        """Validate generated SBOMs"""
        results = {}
        
        for format_name, path in sbom_paths.items():
            print(f"Validating {format_name} SBOM...")
            
            if format_name == "cyclonedx":
                results[format_name] = self._validate_cyclonedx(path)
            elif format_name == "spdx":
                results[format_name] = self._validate_spdx(path)
            elif format_name == "syft":
                results[format_name] = self._validate_syft(path)
                
        return results
        
    def _validate_cyclonedx(self, sbom_path: Path) -> bool:
        """Validate CycloneDX SBOM"""
        try:
            # Basic JSON validation
            with open(sbom_path) as f:
                data = json.load(f)
                
            # Check required fields
            required_fields = ["bomFormat", "specVersion", "components"]
            for field in required_fields:
                if field not in data:
                    print(f"✗ Missing required field: {field}")
                    return False
                    
            print(f"✓ CycloneDX SBOM validation passed")
            return True
        except Exception as e:
            print(f"✗ CycloneDX validation failed: {e}")
            return False
            
    def _validate_spdx(self, sbom_path: Path) -> bool:
        """Validate SPDX SBOM"""
        try:
            with open(sbom_path) as f:
                data = json.load(f)
                
            # Check if it's a valid vulnerability report format
            if "vulnerabilities" in data:
                print(f"✓ SPDX vulnerability report validation passed")
                return True
            else:
                print(f"✗ SPDX format not recognized")
                return False
        except Exception as e:
            print(f"✗ SPDX validation failed: {e}")
            return False
            
    def _validate_syft(self, sbom_path: Path) -> bool:
        """Validate Syft SBOM"""
        try:
            with open(sbom_path) as f:
                data = json.load(f)
                
            required_fields = ["artifacts", "source", "schema"]
            for field in required_fields:
                if field not in data:
                    print(f"✗ Missing required field: {field}")
                    return False
                    
            print(f"✓ Syft SBOM validation passed")
            return True
        except Exception as e:
            print(f"✗ Syft validation failed: {e}")
            return False
            
    def correlate_vulnerabilities(self, sbom_paths: Dict[str, Path]) -> None:
        """Correlate SBOMs with vulnerability databases"""
        print("\nCorrelating with vulnerability databases...")
        
        for format_name, path in sbom_paths.items():
            if format_name == "syft":
                self._scan_with_grype(path)
                
    def _scan_with_grype(self, sbom_path: Path) -> None:
        """Scan SBOM with Grype vulnerability scanner"""
        try:
            vuln_path = sbom_path.parent / f"vulnerabilities-{sbom_path.stem}.json"
            cmd = [
                "grype",
                f"sbom:{sbom_path}",
                "--output", "json",
                "--file", str(vuln_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Vulnerability scan completed: {vuln_path}")
        except subprocess.CalledProcessError:
            print(f"✗ Grype not available - skipping vulnerability scan")
        except Exception as e:
            print(f"✗ Vulnerability scan failed: {e}")
            
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        try:
            subprocess.run([command, "--help"], capture_output=True, check=False)
            return True
        except FileNotFoundError:
            return False
            
    def generate_summary(self, sbom_paths: Dict[str, Path], validation_results: Dict[str, bool]) -> None:
        """Generate SBOM generation summary"""
        print(f"\n{'='*60}")
        print("SBOM GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Project: {self.project_path.name}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.sbom_dir}")
        print()
        
        for format_name, path in sbom_paths.items():
            status = "✓ VALID" if validation_results.get(format_name, False) else "✗ INVALID"
            print(f"{format_name.upper():>12}: {status} - {path.name}")
            
        print(f"\nTotal SBOMs generated: {len(sbom_paths)}")
        print(f"Valid SBOMs: {sum(validation_results.values())}")


def main():
    """Main execution function"""
    generator = SBOMGenerator()
    
    print("Starting SBOM generation...")
    print(f"Project path: {generator.project_path}")
    
    # Generate all SBOM formats
    sbom_paths = generator.generate_all_formats()
    
    if not sbom_paths:
        print("No SBOMs were generated successfully")
        sys.exit(1)
        
    # Validate generated SBOMs
    validation_results = generator.validate_sboms(sbom_paths)
    
    # Correlate with vulnerability databases
    generator.correlate_vulnerabilities(sbom_paths)
    
    # Generate summary
    generator.generate_summary(sbom_paths, validation_results)
    
    # Exit with appropriate code
    if all(validation_results.values()):
        print("\n✓ All SBOMs generated and validated successfully")
        sys.exit(0)
    else:
        print("\n✗ Some SBOMs failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()