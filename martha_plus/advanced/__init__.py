"""Advanced, dependency-free research and operations utilities.

These modules are deliberately opt-in and extend the base research toolkit.
They do not add live execution or connect to a brokerage.
"""
from .features import FeaturePipeline, FeatureRow
from .portfolio import PortfolioOptimizer, OptimizationResult
from .experiments import ExperimentRegistry, ExperimentRun
from .eventbus import EventBus, EventEnvelope
from .alerts import AlertManager, Alert, Severity
from .constraints import AllocationConstraints, ConstraintProjector
from .regimes import RegimeDetector, MarketRegime

__all__ = [
    "FeaturePipeline", "FeatureRow", "PortfolioOptimizer", "OptimizationResult",
    "ExperimentRegistry", "ExperimentRun", "EventBus", "EventEnvelope",
    "AlertManager", "Alert", "Severity", "AllocationConstraints",
    "ConstraintProjector", "RegimeDetector", "MarketRegime",
]
