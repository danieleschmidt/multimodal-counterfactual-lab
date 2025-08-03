"""Storage manager for handling file operations and data persistence."""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from PIL import Image
import hashlib

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages file storage operations for counterfactual data."""
    
    def __init__(self, base_dir: str = "./data"):
        """Initialize storage manager.
        
        Args:
            base_dir: Base directory for all storage operations
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create storage subdirectories
        self.images_dir = self.base_dir / "images"
        self.results_dir = self.base_dir / "results"
        self.reports_dir = self.base_dir / "reports"
        self.datasets_dir = self.base_dir / "datasets"
        self.exports_dir = self.base_dir / "exports"
        
        for dir_path in [self.images_dir, self.results_dir, self.reports_dir, 
                        self.datasets_dir, self.exports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"Storage manager initialized: {self.base_dir}")
    
    def _generate_file_hash(self, content: Union[bytes, str]) -> str:
        """Generate SHA-256 hash for file content."""
        if isinstance(content, str):
            content = content.encode()
        
        hash_obj = hashlib.sha256(content)
        return hash_obj.hexdigest()[:16]  # Use first 16 characters
    
    def _generate_image_hash(self, image: Image.Image) -> str:
        """Generate hash for PIL Image."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return self._generate_file_hash(buffer.getvalue())
    
    def save_image(self, image: Image.Image, filename: Optional[str] = None, 
                  subdir: str = "generated") -> Dict[str, str]:
        """Save PIL Image to storage.
        
        Args:
            image: PIL Image to save
            filename: Optional filename (will generate if not provided)
            subdir: Subdirectory within images directory
            
        Returns:
            Dictionary with file information
        """
        try:
            # Create subdirectory
            save_dir = self.images_dir / subdir
            save_dir.mkdir(exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                image_hash = self._generate_image_hash(image)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{image_hash}.png"
            
            # Ensure .png extension
            if not filename.endswith('.png'):
                filename += '.png'
            
            file_path = save_dir / filename
            
            # Save image
            image.save(file_path, format='PNG', optimize=True)
            
            file_info = {
                "filename": filename,
                "path": str(file_path),
                "relative_path": str(file_path.relative_to(self.base_dir)),
                "size_bytes": file_path.stat().st_size,
                "timestamp": datetime.now().isoformat(),
                "image_size": image.size,
                "image_mode": image.mode
            }
            
            logger.info(f"Saved image: {file_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise
    
    def load_image(self, path: Union[str, Path]) -> Image.Image:
        """Load PIL Image from storage.
        
        Args:
            path: File path (absolute or relative to base_dir)
            
        Returns:
            PIL Image
        """
        try:
            if not Path(path).is_absolute():
                path = self.base_dir / path
            
            image = Image.open(path)
            logger.info(f"Loaded image: {path}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            raise
    
    def save_counterfactual_result(self, result: Dict[str, Any], 
                                 experiment_id: Optional[str] = None) -> Dict[str, str]:
        """Save complete counterfactual generation result.
        
        Args:
            result: Counterfactual generation result
            experiment_id: Optional experiment identifier
            
        Returns:
            Dictionary with saved file information
        """
        try:
            # Generate experiment ID if not provided
            if experiment_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_id = f"exp_{timestamp}"
            
            # Create experiment directory
            exp_dir = self.results_dir / experiment_id
            exp_dir.mkdir(exist_ok=True)
            
            saved_files = {}
            
            # Save original image if present
            if "original_image" in result and isinstance(result["original_image"], Image.Image):
                original_info = self.save_image(
                    result["original_image"], 
                    "original.png", 
                    str(exp_dir.relative_to(self.images_dir))
                )
                saved_files["original_image"] = original_info
            
            # Save counterfactual images
            if "counterfactuals" in result:
                cf_images = []
                for i, cf in enumerate(result["counterfactuals"]):
                    if "generated_image" in cf and isinstance(cf["generated_image"], Image.Image):
                        cf_info = self.save_image(
                            cf["generated_image"],
                            f"counterfactual_{i}.png",
                            str(exp_dir.relative_to(self.images_dir))
                        )
                        cf_images.append(cf_info)
                
                saved_files["counterfactual_images"] = cf_images
            
            # Prepare result data for JSON serialization
            result_data = self._prepare_for_json(result)
            
            # Save result metadata
            metadata_path = exp_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            
            saved_files["metadata"] = {
                "path": str(metadata_path),
                "relative_path": str(metadata_path.relative_to(self.base_dir))
            }
            
            # Create summary
            summary = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "method": result.get("method", "unknown"),
                "num_counterfactuals": len(result.get("counterfactuals", [])),
                "generation_time": result.get("metadata", {}).get("generation_time", 0),
                "saved_files": saved_files
            }
            
            summary_path = exp_dir / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Saved counterfactual result: {exp_dir}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to save counterfactual result: {e}")
            raise
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization by handling non-serializable objects."""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in ["original_image", "generated_image"] and isinstance(value, Image.Image):
                    # Store image info instead of the image object
                    result[f"{key}_info"] = {
                        "size": value.size,
                        "mode": value.mode,
                        "format": getattr(value, 'format', 'PNG')
                    }
                else:
                    result[key] = self._prepare_for_json(value)
            return result
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif hasattr(data, '__dict__'):
            # Handle objects with __dict__
            return self._prepare_for_json(data.__dict__)
        else:
            return data
    
    def load_counterfactual_result(self, experiment_id: str) -> Dict[str, Any]:
        """Load counterfactual generation result.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Loaded result data
        """
        try:
            exp_dir = self.results_dir / experiment_id
            
            if not exp_dir.exists():
                raise FileNotFoundError(f"Experiment not found: {experiment_id}")
            
            # Load metadata
            metadata_path = exp_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                result = json.load(f)
            
            # Load images back
            # Load original image
            original_path = exp_dir / "original.png"
            if original_path.exists():
                result["original_image"] = self.load_image(original_path)
            
            # Load counterfactual images
            if "counterfactuals" in result:
                for i, cf in enumerate(result["counterfactuals"]):
                    cf_path = exp_dir / f"counterfactual_{i}.png"
                    if cf_path.exists():
                        cf["generated_image"] = self.load_image(cf_path)
            
            logger.info(f"Loaded counterfactual result: {experiment_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load counterfactual result {experiment_id}: {e}")
            raise
    
    def save_evaluation_report(self, report: Dict[str, Any], 
                             report_name: Optional[str] = None) -> Dict[str, str]:
        """Save bias evaluation report.
        
        Args:
            report: Evaluation report data
            report_name: Optional report name
            
        Returns:
            Dictionary with saved file information
        """
        try:
            if report_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_type = report.get("report_type", "evaluation").lower().replace(" ", "_")
                report_name = f"{report_type}_{timestamp}"
            
            # Ensure .json extension
            if not report_name.endswith('.json'):
                report_name += '.json'
            
            report_path = self.reports_dir / report_name
            
            # Add storage metadata
            report_with_metadata = {
                **report,
                "storage_info": {
                    "saved_at": datetime.now().isoformat(),
                    "file_path": str(report_path),
                    "report_name": report_name
                }
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_with_metadata, f, indent=2, default=str)
            
            file_info = {
                "report_name": report_name,
                "path": str(report_path),
                "relative_path": str(report_path.relative_to(self.base_dir)),
                "size_bytes": report_path.stat().st_size,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Saved evaluation report: {report_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
            raise
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all stored experiments.
        
        Returns:
            List of experiment information
        """
        experiments = []
        
        try:
            for exp_dir in self.results_dir.iterdir():
                if exp_dir.is_dir():
                    summary_path = exp_dir / "summary.json"
                    
                    if summary_path.exists():
                        try:
                            with open(summary_path, 'r') as f:
                                summary = json.load(f)
                            experiments.append(summary)
                        except Exception as e:
                            logger.warning(f"Failed to read summary for {exp_dir.name}: {e}")
                    else:
                        # Create basic info if summary doesn't exist
                        experiments.append({
                            "experiment_id": exp_dir.name,
                            "timestamp": "unknown",
                            "method": "unknown",
                            "has_summary": False
                        })
            
            # Sort by timestamp (newest first)
            experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
        
        return experiments
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """List all stored reports.
        
        Returns:
            List of report information
        """
        reports = []
        
        try:
            for report_file in self.reports_dir.glob("*.json"):
                try:
                    stat = report_file.stat()
                    reports.append({
                        "name": report_file.name,
                        "path": str(report_file),
                        "relative_path": str(report_file.relative_to(self.base_dir)),
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to get info for {report_file}: {e}")
            
            # Sort by modification time (newest first)
            reports.sort(key=lambda x: x["modified"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list reports: {e}")
        
        return reports
    
    def export_experiment(self, experiment_id: str, export_format: str = "zip") -> str:
        """Export experiment data.
        
        Args:
            experiment_id: Experiment to export
            export_format: Export format ("zip", "tar")
            
        Returns:
            Path to exported file
        """
        try:
            exp_dir = self.results_dir / experiment_id
            
            if not exp_dir.exists():
                raise FileNotFoundError(f"Experiment not found: {experiment_id}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format == "zip":
                export_path = self.exports_dir / f"{experiment_id}_{timestamp}.zip"
                shutil.make_archive(str(export_path.with_suffix('')), 'zip', exp_dir)
            elif export_format == "tar":
                export_path = self.exports_dir / f"{experiment_id}_{timestamp}.tar.gz"
                shutil.make_archive(str(export_path.with_suffix('').with_suffix('')), 'gztar', exp_dir)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            logger.info(f"Exported experiment {experiment_id} to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Failed to export experiment {experiment_id}: {e}")
            raise
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up data older than specified days.
        
        Args:
            days: Number of days to keep data
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        logger.info(f"Cleaning up data older than {days} days")
        
        cleaned_count = 0
        
        try:
            # Clean old experiments
            for exp_dir in self.results_dir.iterdir():
                if exp_dir.is_dir() and exp_dir.stat().st_mtime < cutoff_time:
                    shutil.rmtree(exp_dir)
                    cleaned_count += 1
                    logger.info(f"Removed old experiment: {exp_dir.name}")
            
            # Clean old reports
            for report_file in self.reports_dir.glob("*.json"):
                if report_file.stat().st_mtime < cutoff_time:
                    report_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Removed old report: {report_file.name}")
            
            # Clean old exports
            for export_file in self.exports_dir.glob("*"):
                if export_file.is_file() and export_file.stat().st_mtime < cutoff_time:
                    export_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Removed old export: {export_file.name}")
            
            logger.info(f"Cleanup completed. Removed {cleaned_count} items.")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "base_directory": str(self.base_dir),
            "directories": {},
            "total_size_bytes": 0,
            "file_counts": {}
        }
        
        try:
            for subdir_name in ["images", "results", "reports", "datasets", "exports"]:
                subdir = self.base_dir / subdir_name
                
                if subdir.exists():
                    size = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
                    count = len(list(subdir.rglob('*')))
                    
                    stats["directories"][subdir_name] = {
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "file_count": count
                    }
                    
                    stats["total_size_bytes"] += size
                    stats["file_counts"][subdir_name] = count
            
            stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
            
        except Exception as e:
            logger.error(f"Error calculating storage stats: {e}")
            stats["error"] = str(e)
        
        return stats