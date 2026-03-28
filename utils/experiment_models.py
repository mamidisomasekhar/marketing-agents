"""Database models for experiment tracking and metrics."""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ExperimentStatus(Enum):
    """Experiment lifecycle statuses."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class Experiment:
    """Experiment definition."""
    id: str
    name: str
    description: str
    agent_name: str
    status: ExperimentStatus
    created_at: str
    updated_at: str
    config: Dict[str, Any]  # Variant definitions, evaluation criteria
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "status": self.status.value,
            "config": json.dumps(self.config)
        }


@dataclass
class ExperimentVariant:
    """Variant (model configuration) within an experiment."""
    id: str
    experiment_id: str
    name: str
    model_id: str
    provider: str
    weight: int  # Traffic allocation percentage
    config: Dict[str, Any]  # Temperature, max_tokens, etc.
    is_active: bool
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "config": json.dumps(self.config)
        }


@dataclass
class ExperimentExecution:
    """Single execution of an agent with a specific variant."""
    id: Optional[int]
    experiment_id: str
    variant_id: str
    agent_name: str
    model_id: str
    
    # Timing
    started_at: str
    completed_at: Optional[str]
    latency_ms: int
    
    # Cost
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    
    # Quality
    success: bool
    error_message: Optional[str]
    quality_score: float
    quality_metrics: Dict[str, float]  # Individual metric scores
    
    # Content (truncated for storage)
    input_query: str
    output_sample: str
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "quality_metrics": json.dumps(self.quality_metrics)
        }


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment/variant."""
    experiment_id: str
    variant_id: str
    
    # Sample stats
    total_executions: int
    successful_executions: int
    failed_executions: int
    
    # Performance
    avg_latency_ms: float
    min_latency_ms: int
    max_latency_ms: int
    p50_latency_ms: int
    p95_latency_ms: int
    p99_latency_ms: int
    
    # Cost
    total_cost_usd: float
    avg_cost_per_execution: float
    
    # Quality
    avg_quality_score: float
    min_quality_score: float
    max_quality_score: float
    
    # Success rate
    success_rate: float
    
    # Time window
    calculated_at: str
    window_start: str
    window_end: str


class ExperimentDatabase:
    """SQLite database for experiment tracking."""
    
    def __init__(self, db_path: str = "data/experiments.db"):
        self.db_path = db_path
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize database schema."""
        schema = """
        -- Experiments table
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            agent_name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            config_json TEXT
        );
        
        -- Experiment variants table
        CREATE TABLE IF NOT EXISTS experiment_variants (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            name TEXT NOT NULL,
            model_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            weight INTEGER DEFAULT 50,
            config_json TEXT,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
        
        -- Executions table (main time-series data)
        CREATE TABLE IF NOT EXISTS experiment_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            variant_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            model_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            latency_ms INTEGER,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            success BOOLEAN DEFAULT 0,
            error_message TEXT,
            quality_score REAL DEFAULT 0.0,
            quality_metrics_json TEXT,
            input_query TEXT,
            output_sample TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id),
            FOREIGN KEY (variant_id) REFERENCES experiment_variants(id)
        );
        
        -- Aggregated metrics table (for fast dashboard queries)
        CREATE TABLE IF NOT EXISTS experiment_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            variant_id TEXT NOT NULL,
            total_executions INTEGER DEFAULT 0,
            successful_executions INTEGER DEFAULT 0,
            failed_executions INTEGER DEFAULT 0,
            avg_latency_ms REAL DEFAULT 0.0,
            min_latency_ms INTEGER DEFAULT 0,
            max_latency_ms INTEGER DEFAULT 0,
            p50_latency_ms INTEGER DEFAULT 0,
            p95_latency_ms INTEGER DEFAULT 0,
            p99_latency_ms INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0.0,
            avg_cost_per_execution REAL DEFAULT 0.0,
            avg_quality_score REAL DEFAULT 0.0,
            min_quality_score REAL DEFAULT 0.0,
            max_quality_score REAL DEFAULT 0.0,
            success_rate REAL DEFAULT 0.0,
            calculated_at TEXT NOT NULL,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id),
            FOREIGN KEY (variant_id) REFERENCES experiment_variants(id)
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_executions_experiment 
            ON experiment_executions(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_executions_variant 
            ON experiment_executions(variant_id);
        CREATE INDEX IF NOT EXISTS idx_executions_timestamp 
            ON experiment_executions(started_at);
        CREATE INDEX IF NOT EXISTS idx_executions_model 
            ON experiment_executions(model_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_experiment 
            ON experiment_metrics(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_calculated 
            ON experiment_metrics(calculated_at);
        """
        
        conn = self._get_connection()
        conn.executescript(schema)
        conn.commit()
    
    def create_experiment(self, experiment: Experiment) -> str:
        """Create new experiment."""
        conn = self._get_connection()
        conn.execute(
            """INSERT INTO experiments 
               (id, name, description, agent_name, status, created_at, updated_at, config_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (experiment.id, experiment.name, experiment.description,
             experiment.agent_name, experiment.status.value,
             experiment.created_at, experiment.updated_at,
             json.dumps(experiment.config))
        )
        conn.commit()
        return experiment.id
    
    def add_variant(self, variant: ExperimentVariant) -> str:
        """Add variant to experiment."""
        conn = self._get_connection()
        conn.execute(
            """INSERT INTO experiment_variants
               (id, experiment_id, name, model_id, provider, weight, config_json, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (variant.id, variant.experiment_id, variant.name, variant.model_id,
             variant.provider, variant.weight, json.dumps(variant.config),
             variant.is_active)
        )
        conn.commit()
        return variant.id
    
    def record_execution(self, execution: ExperimentExecution) -> int:
        """Record an execution."""
        conn = self._get_connection()
        cursor = conn.execute(
            """INSERT INTO experiment_executions
               (experiment_id, variant_id, agent_name, model_id, started_at, completed_at,
                latency_ms, input_tokens, output_tokens, total_tokens, cost_usd,
                success, error_message, quality_score, quality_metrics_json,
                input_query, output_sample)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (execution.experiment_id, execution.variant_id, execution.agent_name,
             execution.model_id, execution.started_at, execution.completed_at,
             execution.latency_ms, execution.input_tokens, execution.output_tokens,
             execution.total_tokens, execution.cost_usd, execution.success,
             execution.error_message, execution.quality_score,
             json.dumps(execution.quality_metrics),
             execution.input_query[:1000], execution.output_sample[:2000])
        )
        conn.commit()
        return cursor.lastrowid
    
    def calculate_metrics(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Calculate aggregated metrics for experiment."""
        conn = self._get_connection()
        
        # Get per-variant stats
        cursor = conn.execute(
            """SELECT 
                variant_id,
                model_id,
                COUNT(*) as total_executions,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_executions,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed_executions,
                AVG(latency_ms) as avg_latency,
                MIN(latency_ms) as min_latency,
                MAX(latency_ms) as max_latency,
                AVG(cost_usd) as avg_cost,
                SUM(cost_usd) as total_cost,
                AVG(quality_score) as avg_quality,
                MIN(quality_score) as min_quality,
                MAX(quality_score) as max_quality,
                AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                MIN(started_at) as window_start,
                MAX(started_at) as window_end
            FROM experiment_executions
            WHERE experiment_id = ?
            GROUP BY variant_id, model_id""",
            (experiment_id,)
        )
        
        results = []
        for row in cursor.fetchall():
            variant_metrics = dict(row)
            
            # Calculate percentiles
            percentile_cursor = conn.execute(
                """SELECT latency_ms 
                   FROM experiment_executions
                   WHERE experiment_id = ? AND variant_id = ? AND success = 1
                   ORDER BY latency_ms""",
                (experiment_id, row[0])
            )
            latencies = [r[0] for r in percentile_cursor.fetchall() if r[0]]
            
            if latencies:
                p50_idx = int(len(latencies) * 0.5)
                p95_idx = int(len(latencies) * 0.95)
                p99_idx = int(len(latencies) * 0.99)
                
                variant_metrics['p50_latency_ms'] = latencies[min(p50_idx, len(latencies)-1)]
                variant_metrics['p95_latency_ms'] = latencies[min(p95_idx, len(latencies)-1)]
                variant_metrics['p99_latency_ms'] = latencies[min(p99_idx, len(latencies)-1)]
            else:
                variant_metrics['p50_latency_ms'] = 0
                variant_metrics['p95_latency_ms'] = 0
                variant_metrics['p99_latency_ms'] = 0
            
            results.append(variant_metrics)
        
        return results
    
    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment summary with all variant metrics."""
        conn = self._get_connection()
        
        # Get experiment info
        cursor = conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,)
        )
        exp_row = cursor.fetchone()
        
        if not exp_row:
            return None
        
        experiment = dict(exp_row)
        experiment['config'] = json.loads(experiment.get('config_json', '{}'))
        del experiment['config_json']
        
        # Get variants
        cursor = conn.execute(
            "SELECT * FROM experiment_variants WHERE experiment_id = ?",
            (experiment_id,)
        )
        variants = []
        for row in cursor.fetchall():
            variant = dict(row)
            variant['config'] = json.loads(variant.get('config_json', '{}'))
            del variant['config_json']
            variants.append(variant)
        
        # Get metrics
        metrics = self.calculate_metrics(experiment_id)
        
        return {
            "experiment": experiment,
            "variants": variants,
            "metrics": metrics
        }
    
    def list_experiments(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments with summary stats."""
        conn = self._get_connection()
        
        if agent_name:
            cursor = conn.execute(
                """SELECT e.*, 
                          COUNT(DISTINCT ev.id) as variant_count,
                          COUNT(ex.id) as execution_count
                   FROM experiments e
                   LEFT JOIN experiment_variants ev ON e.id = ev.experiment_id
                   LEFT JOIN experiment_executions ex ON e.id = ex.experiment_id
                   WHERE e.agent_name = ?
                   GROUP BY e.id
                   ORDER BY e.created_at DESC""",
                (agent_name,)
            )
        else:
            cursor = conn.execute(
                """SELECT e.*, 
                          COUNT(DISTINCT ev.id) as variant_count,
                          COUNT(ex.id) as execution_count
                   FROM experiments e
                   LEFT JOIN experiment_variants ev ON e.id = ev.experiment_id
                   LEFT JOIN experiment_executions ex ON e.id = ex.experiment_id
                   GROUP BY e.id
                   ORDER BY e.created_at DESC"""
            )
        
        results = []
        for row in cursor.fetchall():
            exp = dict(row)
            exp['config'] = json.loads(exp.get('config_json', '{}'))
            del exp['config_json']
            results.append(exp)
        
        return results
