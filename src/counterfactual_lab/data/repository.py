"""Repository classes for data persistence."""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualRecord:
    """Data class for counterfactual records."""
    id: str
    method: str
    original_text: str
    generated_text: str
    target_attributes: Dict[str, str]
    confidence: float
    generation_time: float
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class EvaluationRecord:
    """Data class for evaluation records."""
    id: str
    counterfactual_id: str
    metrics: Dict[str, Any]
    fairness_score: float
    evaluation_time: float
    timestamp: str
    model_info: Dict[str, str]


class BaseRepository:
    """Base repository with common database operations."""
    
    def __init__(self, db_path: str = "./counterfactual_lab.db"):
        """Initialize repository with database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
        logger.info(f"Repository initialized with database: {self.db_path}")
    
    def _initialize_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            self._create_tables(conn)
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_tables")
    
    def _execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def _execute_command(self, command: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE command."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(command, params)
            conn.commit()
            return cursor.rowcount


class CounterfactualRepository(BaseRepository):
    """Repository for storing and retrieving counterfactual data."""
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create counterfactual tables."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS counterfactuals (
                id TEXT PRIMARY KEY,
                method TEXT NOT NULL,
                original_text TEXT NOT NULL,
                generated_text TEXT NOT NULL,
                target_attributes TEXT NOT NULL,  -- JSON
                confidence REAL NOT NULL,
                generation_time REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL  -- JSON
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_counterfactuals_method 
            ON counterfactuals(method)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_counterfactuals_timestamp 
            ON counterfactuals(timestamp)
        """)
    
    def save(self, record: CounterfactualRecord) -> bool:
        """Save a counterfactual record."""
        try:
            command = """
                INSERT OR REPLACE INTO counterfactuals 
                (id, method, original_text, generated_text, target_attributes, 
                 confidence, generation_time, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                record.id,
                record.method,
                record.original_text,
                record.generated_text,
                json.dumps(record.target_attributes),
                record.confidence,
                record.generation_time,
                record.timestamp,
                json.dumps(record.metadata)
            )
            
            rows_affected = self._execute_command(command, params)
            logger.info(f"Saved counterfactual record: {record.id}")
            return rows_affected > 0
            
        except Exception as e:
            logger.error(f"Failed to save counterfactual record {record.id}: {e}")
            return False
    
    def find_by_id(self, record_id: str) -> Optional[CounterfactualRecord]:
        """Find a counterfactual record by ID."""
        try:
            query = "SELECT * FROM counterfactuals WHERE id = ?"
            results = self._execute_query(query, (record_id,))
            
            if results:
                row = results[0]
                return CounterfactualRecord(
                    id=row['id'],
                    method=row['method'],
                    original_text=row['original_text'],
                    generated_text=row['generated_text'],
                    target_attributes=json.loads(row['target_attributes']),
                    confidence=row['confidence'],
                    generation_time=row['generation_time'],
                    timestamp=row['timestamp'],
                    metadata=json.loads(row['metadata'])
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to find counterfactual record {record_id}: {e}")
            return None
    
    def find_by_method(self, method: str, limit: int = 100) -> List[CounterfactualRecord]:
        """Find counterfactual records by generation method."""
        try:
            query = """
                SELECT * FROM counterfactuals 
                WHERE method = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            results = self._execute_query(query, (method, limit))
            
            records = []
            for row in results:
                record = CounterfactualRecord(
                    id=row['id'],
                    method=row['method'],
                    original_text=row['original_text'],
                    generated_text=row['generated_text'],
                    target_attributes=json.loads(row['target_attributes']),
                    confidence=row['confidence'],
                    generation_time=row['generation_time'],
                    timestamp=row['timestamp'],
                    metadata=json.loads(row['metadata'])
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to find records by method {method}: {e}")
            return []
    
    def find_by_attributes(self, attributes: Dict[str, str], limit: int = 100) -> List[CounterfactualRecord]:
        """Find counterfactual records by target attributes."""
        try:
            # Simple approach - would need better JSON querying for production
            query = """
                SELECT * FROM counterfactuals 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            results = self._execute_query(query, (limit * 2,))  # Get more to filter
            
            records = []
            for row in results:
                target_attrs = json.loads(row['target_attributes'])
                
                # Check if all requested attributes match
                match = all(
                    target_attrs.get(key) == value 
                    for key, value in attributes.items()
                )
                
                if match:
                    record = CounterfactualRecord(
                        id=row['id'],
                        method=row['method'],
                        original_text=row['original_text'],
                        generated_text=row['generated_text'],
                        target_attributes=target_attrs,
                        confidence=row['confidence'],
                        generation_time=row['generation_time'],
                        timestamp=row['timestamp'],
                        metadata=json.loads(row['metadata'])
                    )
                    records.append(record)
                    
                    if len(records) >= limit:
                        break
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to find records by attributes {attributes}: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        try:
            stats = {}
            
            # Total count
            query = "SELECT COUNT(*) as total FROM counterfactuals"
            result = self._execute_query(query)
            stats['total_records'] = result[0]['total'] if result else 0
            
            # Count by method
            query = """
                SELECT method, COUNT(*) as count 
                FROM counterfactuals 
                GROUP BY method
            """
            results = self._execute_query(query)
            stats['by_method'] = {row['method']: row['count'] for row in results}
            
            # Average confidence
            query = "SELECT AVG(confidence) as avg_confidence FROM counterfactuals"
            result = self._execute_query(query)
            stats['avg_confidence'] = result[0]['avg_confidence'] if result else 0.0
            
            # Date range
            query = """
                SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest 
                FROM counterfactuals
            """
            result = self._execute_query(query)
            if result and result[0]['earliest']:
                stats['date_range'] = {
                    'earliest': result[0]['earliest'],
                    'latest': result[0]['latest']
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def delete_by_id(self, record_id: str) -> bool:
        """Delete a counterfactual record by ID."""
        try:
            command = "DELETE FROM counterfactuals WHERE id = ?"
            rows_affected = self._execute_command(command, (record_id,))
            
            if rows_affected > 0:
                logger.info(f"Deleted counterfactual record: {record_id}")
                return True
            else:
                logger.warning(f"No record found to delete: {record_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete record {record_id}: {e}")
            return False


class EvaluationRepository(BaseRepository):
    """Repository for storing and retrieving evaluation results."""
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create evaluation tables."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                counterfactual_id TEXT NOT NULL,
                metrics TEXT NOT NULL,  -- JSON
                fairness_score REAL NOT NULL,
                evaluation_time REAL NOT NULL,
                timestamp TEXT NOT NULL,
                model_info TEXT NOT NULL,  -- JSON
                FOREIGN KEY (counterfactual_id) REFERENCES counterfactuals (id)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_counterfactual 
            ON evaluations(counterfactual_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_fairness_score 
            ON evaluations(fairness_score)
        """)
    
    def save(self, record: EvaluationRecord) -> bool:
        """Save an evaluation record."""
        try:
            command = """
                INSERT OR REPLACE INTO evaluations 
                (id, counterfactual_id, metrics, fairness_score, 
                 evaluation_time, timestamp, model_info)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                record.id,
                record.counterfactual_id,
                json.dumps(record.metrics),
                record.fairness_score,
                record.evaluation_time,
                record.timestamp,
                json.dumps(record.model_info)
            )
            
            rows_affected = self._execute_command(command, params)
            logger.info(f"Saved evaluation record: {record.id}")
            return rows_affected > 0
            
        except Exception as e:
            logger.error(f"Failed to save evaluation record {record.id}: {e}")
            return False
    
    def find_by_counterfactual_id(self, counterfactual_id: str) -> List[EvaluationRecord]:
        """Find evaluation records by counterfactual ID."""
        try:
            query = """
                SELECT * FROM evaluations 
                WHERE counterfactual_id = ? 
                ORDER BY timestamp DESC
            """
            results = self._execute_query(query, (counterfactual_id,))
            
            records = []
            for row in results:
                record = EvaluationRecord(
                    id=row['id'],
                    counterfactual_id=row['counterfactual_id'],
                    metrics=json.loads(row['metrics']),
                    fairness_score=row['fairness_score'],
                    evaluation_time=row['evaluation_time'],
                    timestamp=row['timestamp'],
                    model_info=json.loads(row['model_info'])
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to find evaluations for counterfactual {counterfactual_id}: {e}")
            return []
    
    def find_by_fairness_score_range(
        self, 
        min_score: float, 
        max_score: float, 
        limit: int = 100
    ) -> List[EvaluationRecord]:
        """Find evaluation records by fairness score range."""
        try:
            query = """
                SELECT * FROM evaluations 
                WHERE fairness_score BETWEEN ? AND ? 
                ORDER BY fairness_score DESC 
                LIMIT ?
            """
            results = self._execute_query(query, (min_score, max_score, limit))
            
            records = []
            for row in results:
                record = EvaluationRecord(
                    id=row['id'],
                    counterfactual_id=row['counterfactual_id'],
                    metrics=json.loads(row['metrics']),
                    fairness_score=row['fairness_score'],
                    evaluation_time=row['evaluation_time'],
                    timestamp=row['timestamp'],
                    model_info=json.loads(row['model_info'])
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to find evaluations by score range [{min_score}, {max_score}]: {e}")
            return []
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        try:
            stats = {}
            
            # Total evaluations
            query = "SELECT COUNT(*) as total FROM evaluations"
            result = self._execute_query(query)
            stats['total_evaluations'] = result[0]['total'] if result else 0
            
            # Fairness score statistics
            query = """
                SELECT 
                    AVG(fairness_score) as avg_score,
                    MIN(fairness_score) as min_score,
                    MAX(fairness_score) as max_score,
                    COUNT(CASE WHEN fairness_score >= 0.8 THEN 1 END) as high_fairness_count
                FROM evaluations
            """
            result = self._execute_query(query)
            if result:
                row = result[0]
                stats['fairness_scores'] = {
                    'average': row['avg_score'] or 0.0,
                    'minimum': row['min_score'] or 0.0,
                    'maximum': row['max_score'] or 0.0,
                    'high_fairness_count': row['high_fairness_count'] or 0
                }
            
            # Average evaluation time
            query = "SELECT AVG(evaluation_time) as avg_time FROM evaluations"
            result = self._execute_query(query)
            stats['avg_evaluation_time'] = result[0]['avg_time'] if result else 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get evaluation statistics: {e}")
            return {}