"""
Core testing framework modules.

This package contains the core functionality for the MRI-to-CT synthesis testing framework:
- Edge case simulation
- Enhanced evaluation metrics
- Robust testing pipeline
- Data loading utilities
- Basic evaluation metrics
"""

# Basic imports that should always work
from .evaluation import calculate_metrics
from .simple_loader import SimpleTestDataset

# Optional imports that may require additional dependencies
try:
    from .edge_case_simulator import EdgeCaseSimulator, create_edge_case_test_suite
    _edge_case_available = True
except ImportError:
    _edge_case_available = False

try:
    from .enhanced_evaluation import RobustEvaluator, calculate_enhanced_metrics, HUAnalyzer
    _enhanced_eval_available = True
except ImportError:
    _enhanced_eval_available = False

try:
    from .robust_testing_pipeline import RobustTestingPipeline, RobustTestDataset
    _robust_pipeline_available = True
except ImportError:
    _robust_pipeline_available = False

# Build __all__ based on what's available
__all__ = ['calculate_metrics', 'SimpleTestDataset']

if _edge_case_available:
    __all__.extend(['EdgeCaseSimulator', 'create_edge_case_test_suite'])

if _enhanced_eval_available:
    __all__.extend(['RobustEvaluator', 'calculate_enhanced_metrics', 'HUAnalyzer'])

if _robust_pipeline_available:
    __all__.extend(['RobustTestingPipeline', 'RobustTestDataset'])
