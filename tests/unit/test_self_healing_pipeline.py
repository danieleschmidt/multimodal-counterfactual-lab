"""Tests for self-healing pipeline guard functionality."""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from counterfactual_lab.self_healing_pipeline import (
    SelfHealingPipelineGuard,
    CircuitBreaker,
    FailureEvent,
    RecoveryStrategy,
    get_global_guard,
    initialize_self_healing
)
from counterfactual_lab.monitoring import SystemHealth


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal conditions."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        @cb
        def test_function():
            return "success"
        
        # Should work normally
        result = test_function()
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_failure_counting(self):
        """Test circuit breaker failure counting."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        @cb
        def failing_function():
            raise Exception("Test failure")
        
        # First two failures should not open circuit
        for i in range(2):
            with pytest.raises(Exception):
                failing_function()
            assert cb.state == "CLOSED"
            assert cb.failure_count == i + 1
        
        # Third failure should open circuit
        with pytest.raises(Exception):
            failing_function()
        assert cb.state == "OPEN"
        assert cb.failure_count == 3
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker when open."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        @cb
        def failing_function():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                failing_function()
        
        assert cb.state == "OPEN"
        
        # Should raise exception without calling function
        with pytest.raises(Exception, match="Circuit breaker OPEN"):
            failing_function()
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        call_count = 0
        
        @cb
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Initial failures")
            return "success"
        
        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                test_function()
        
        assert cb.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to HALF_OPEN and then CLOSED on success
        result = test_function()
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0


class TestSelfHealingPipelineGuard:
    """Test self-healing pipeline guard."""
    
    @pytest.fixture
    def guard(self):
        """Create test guard instance."""
        return SelfHealingPipelineGuard(
            monitoring_interval=1,
            auto_recovery=True
        )
    
    def test_guard_initialization(self, guard):
        """Test guard initialization."""
        assert guard.monitoring_interval == 1
        assert guard.auto_recovery is True
        assert not guard.is_running
        assert guard.recovery_in_progress is False
        assert len(guard.recovery_strategies) > 0
        assert len(guard.circuit_breakers) > 0
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Test with default config
        guard = SelfHealingPipelineGuard()
        assert "failure_thresholds" in guard.config
        assert "recovery_settings" in guard.config
        
        # Test with non-existent config file
        guard = SelfHealingPipelineGuard(config_path="non_existent.json")
        assert guard.config["failure_thresholds"]["cpu_usage"] == 95.0
    
    def test_recovery_strategies_initialization(self, guard):
        """Test recovery strategies are properly initialized."""
        expected_strategies = [
            "memory_pressure",
            "gpu_memory", 
            "storage_cleanup",
            "model_recovery",
            "cache_recovery",
            "performance_optimization"
        ]
        
        for strategy_name in expected_strategies:
            assert strategy_name in guard.recovery_strategies
            strategy = guard.recovery_strategies[strategy_name]
            assert hasattr(strategy, 'recovery_function')
            assert strategy.max_attempts > 0
            assert strategy.cooldown_seconds > 0
    
    def test_circuit_breakers_initialization(self, guard):
        """Test circuit breakers are properly initialized."""
        expected_breakers = ["generation", "storage", "cache"]
        
        for breaker_name in expected_breakers:
            assert breaker_name in guard.circuit_breakers
            breaker = guard.circuit_breakers[breaker_name]
            assert isinstance(breaker, CircuitBreaker)
    
    def test_monitoring_start_stop(self, guard):
        """Test monitoring start and stop."""
        assert not guard.is_running
        
        # Start monitoring
        guard.start_monitoring()
        assert guard.is_running
        assert guard.monitor_thread is not None
        assert guard.monitor_thread.is_alive()
        
        # Stop monitoring
        guard.stop_monitoring()
        assert not guard.is_running
    
    @patch('counterfactual_lab.self_healing_pipeline.SystemDiagnostics')
    def test_system_state_analysis(self, mock_diagnostics_class, guard):
        """Test system state analysis and recovery triggering."""
        # Mock diagnostics
        mock_diagnostics = Mock()
        mock_diagnostics.run_full_diagnostics.return_value = {
            "health": {
                "overall_status": "critical",
                "cpu_usage": 95.0,
                "memory_usage": 90.0,
                "issues": ["High CPU usage: 95.0%"]
            },
            "alerts": {
                "active_alerts": [
                    {
                        "type": "cpu_high",
                        "message": "CPU usage high: 95.0%",
                        "severity": "critical",
                        "value": 95.0
                    }
                ]
            }
        }
        guard.diagnostics = mock_diagnostics
        
        # Mock recovery function
        mock_recovery = Mock(return_value=True)
        guard.recovery_strategies["memory_pressure"].recovery_function = mock_recovery
        
        # Analyze system state
        diagnostics = mock_diagnostics.run_full_diagnostics()
        guard._analyze_system_state(diagnostics)
        
        # Should have triggered recovery
        assert len(guard.failure_history) > 0
    
    def test_memory_pressure_recovery(self, guard):
        """Test memory pressure recovery."""
        alerts = [
            {
                "type": "memory_high",
                "message": "Memory usage high: 90.0%",
                "severity": "critical",
                "value": 90.0
            }
        ]
        
        # Should complete without error
        result = guard._recover_memory_pressure(alerts)
        assert isinstance(result, bool)
    
    def test_gpu_memory_recovery(self, guard):
        """Test GPU memory recovery."""
        alerts = [
            {
                "type": "gpu_memory_high", 
                "message": "GPU memory usage high: 95.0%",
                "severity": "critical",
                "value": 95.0
            }
        ]
        
        # Should complete without error
        result = guard._recover_gpu_memory(alerts)
        assert isinstance(result, bool)
    
    def test_storage_cleanup_recovery(self, guard):
        """Test storage cleanup recovery."""
        alerts = [
            {
                "type": "disk_high",
                "message": "Disk usage high: 95.0%",
                "severity": "critical", 
                "value": 95.0
            }
        ]
        
        # Should complete without error
        result = guard._recover_storage_space(alerts)
        assert isinstance(result, bool)
    
    def test_cache_recovery(self, guard):
        """Test cache system recovery."""
        alerts = [
            {
                "type": "cache_error",
                "message": "Cache system error",
                "severity": "critical",
                "value": 0
            }
        ]
        
        result = guard._recover_cache_system(alerts)
        assert isinstance(result, bool)
    
    def test_performance_recovery(self, guard):
        """Test performance optimization recovery."""
        alerts = [
            {
                "type": "generation_slow",
                "message": "Generation time slow: 180.0s",
                "severity": "critical",
                "value": 180.0
            }
        ]
        
        result = guard._recover_performance(alerts)
        assert isinstance(result, bool)
    
    def test_protected_operation_success(self, guard):
        """Test protected operation with success."""
        with guard.protected_operation("test_operation"):
            # Should complete normally
            pass
        
        # Should not have recorded failure
        test_failures = [
            f for f in guard.failure_history 
            if f.component == "test_operation"
        ]
        assert len(test_failures) == 0
    
    def test_protected_operation_failure(self, guard):
        """Test protected operation with failure."""
        with pytest.raises(Exception):
            with guard.protected_operation("test_operation"):
                raise Exception("Test failure")
        
        # Should have recorded failure
        test_failures = [
            f for f in guard.failure_history 
            if f.component == "test_operation"
        ]
        assert len(test_failures) == 1
        assert test_failures[0].error_message == "Test failure"
    
    def test_force_recovery(self, guard):
        """Test forced recovery execution."""
        # Mock recovery function
        mock_recovery = Mock(return_value=True)
        guard.recovery_strategies["memory_pressure"].recovery_function = mock_recovery
        
        # Force recovery
        result = guard.force_recovery("memory_pressure")
        assert result is True
        mock_recovery.assert_called_once()
        
        # Should have recorded event
        manual_recoveries = [
            f for f in guard.failure_history
            if f.failure_type == "manual_recovery"
        ]
        assert len(manual_recoveries) == 1
    
    def test_force_recovery_unknown_strategy(self, guard):
        """Test forced recovery with unknown strategy."""
        result = guard.force_recovery("unknown_strategy")
        assert result is False
    
    def test_failure_history_cleanup(self, guard):
        """Test failure history cleanup."""
        # Add old failure event
        old_event = FailureEvent(
            timestamp=(datetime.now() - timedelta(hours=25)).isoformat(),
            failure_type="test",
            component="test",
            error_message="Old error",
            severity="error",
            context={}
        )
        guard.failure_history.append(old_event)
        
        # Add recent failure event
        recent_event = FailureEvent(
            timestamp=datetime.now().isoformat(),
            failure_type="test",
            component="test", 
            error_message="Recent error",
            severity="error",
            context={}
        )
        guard.failure_history.append(recent_event)
        
        # Cleanup should remove old event
        guard._cleanup_failure_history()
        
        assert len(guard.failure_history) == 1
        assert guard.failure_history[0].error_message == "Recent error"
    
    def test_get_system_status(self, guard):
        """Test system status retrieval."""
        status = guard.get_system_status()
        
        assert "monitoring" in status
        assert "circuit_breakers" in status
        assert "failure_history" in status
        assert "recovery_strategies" in status
        
        assert status["monitoring"]["is_running"] == guard.is_running
        assert status["monitoring"]["auto_recovery"] == guard.auto_recovery
    
    def test_recovery_success_rate_calculation(self, guard):
        """Test recovery success rate calculation."""
        # Add some recovery events
        events = [
            FailureEvent(
                timestamp=datetime.now().isoformat(),
                failure_type="test",
                component="test",
                error_message="Error 1", 
                severity="error",
                context={},
                recovery_attempted=True,
                recovery_successful=True
            ),
            FailureEvent(
                timestamp=datetime.now().isoformat(),
                failure_type="test",
                component="test",
                error_message="Error 2",
                severity="error", 
                context={},
                recovery_attempted=True,
                recovery_successful=False
            ),
            FailureEvent(
                timestamp=datetime.now().isoformat(),
                failure_type="test",
                component="test",
                error_message="Error 3",
                severity="error",
                context={},
                recovery_attempted=False
            )
        ]
        
        guard.failure_history.extend(events)
        
        # Should be 50% (1 success out of 2 attempts)
        success_rate = guard._calculate_recovery_success_rate()
        assert success_rate == 0.5
    
    def test_export_failure_report(self, guard, tmp_path):
        """Test failure report export."""
        # Add some failure events
        event = FailureEvent(
            timestamp=datetime.now().isoformat(),
            failure_type="test_failure",
            component="test_component",
            error_message="Test error message",
            severity="error",
            context={"test": "data"}
        )
        guard.failure_history.append(event)
        
        # Export report
        report_path = tmp_path / "failure_report.json"
        guard.export_failure_report(str(report_path))
        
        assert report_path.exists()
        
        # Verify report content
        import json
        with open(report_path) as f:
            report = json.load(f)
        
        assert "export_timestamp" in report
        assert "system_status" in report
        assert "failure_events" in report
        assert "statistics" in report
        assert len(report["failure_events"]) == 1


class TestGlobalGuardFunctions:
    """Test global guard functions."""
    
    def test_get_global_guard(self):
        """Test getting global guard instance."""
        # Clear any existing global guard
        import counterfactual_lab.self_healing_pipeline as shp
        shp._global_guard = None
        
        guard1 = get_global_guard()
        guard2 = get_global_guard()
        
        # Should return same instance
        assert guard1 is guard2
        assert isinstance(guard1, SelfHealingPipelineGuard)
    
    def test_initialize_self_healing(self):
        """Test self-healing initialization."""
        # Clear any existing global guard
        import counterfactual_lab.self_healing_pipeline as shp
        shp._global_guard = None
        
        guard = initialize_self_healing(
            auto_start=False,
            monitoring_interval=30
        )
        
        assert isinstance(guard, SelfHealingPipelineGuard)
        assert guard.monitoring_interval == 30
        assert not guard.is_running  # auto_start=False
        
        # Test with auto_start=True
        shp._global_guard = None
        guard = initialize_self_healing(auto_start=True)
        assert guard.is_running
        
        # Cleanup
        guard.stop_monitoring()


class TestRecoveryStrategyCooldown:
    """Test recovery strategy cooldown behavior."""
    
    def test_cooldown_prevents_rapid_recovery_attempts(self):
        """Test that cooldown prevents rapid recovery attempts."""
        guard = SelfHealingPipelineGuard(auto_recovery=True)
        
        # Mock recovery function
        mock_recovery = Mock(return_value=True)
        guard.recovery_strategies["memory_pressure"].recovery_function = mock_recovery
        guard.recovery_strategies["memory_pressure"].cooldown_seconds = 10
        
        alerts = [
            {
                "type": "memory_high",
                "message": "Memory usage high",
                "severity": "critical",
                "value": 90.0
            }
        ]
        
        # First attempt should succeed
        guard._execute_recovery_for_alert_type("memory_high", alerts)
        assert mock_recovery.call_count == 1
        
        # Second attempt within cooldown should be skipped
        guard._execute_recovery_for_alert_type("memory_high", alerts)
        assert mock_recovery.call_count == 1  # Still 1, not 2
    
    def test_cooldown_allows_recovery_after_timeout(self):
        """Test that recovery is allowed after cooldown period."""
        guard = SelfHealingPipelineGuard(auto_recovery=True)
        
        # Mock recovery function with very short cooldown
        mock_recovery = Mock(return_value=True)
        guard.recovery_strategies["memory_pressure"].recovery_function = mock_recovery
        guard.recovery_strategies["memory_pressure"].cooldown_seconds = 0.1
        
        alerts = [
            {
                "type": "memory_high",
                "message": "Memory usage high",
                "severity": "critical", 
                "value": 90.0
            }
        ]
        
        # First attempt
        guard._execute_recovery_for_alert_type("memory_high", alerts)
        assert mock_recovery.call_count == 1
        
        # Wait for cooldown
        time.sleep(0.2)
        
        # Second attempt should now succeed
        guard._execute_recovery_for_alert_type("memory_high", alerts)
        assert mock_recovery.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])