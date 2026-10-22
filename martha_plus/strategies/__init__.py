"""Signal-generation strategies; all signals are advisory until risk-approved."""
from .base import Strategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .pairs import PairsZScoreStrategy
from .ensemble import WeightedEnsemble
from .breakout import BreakoutStrategy
from .volatility_target import VolatilityTargetStrategy

__all__ = ["Strategy", "MomentumStrategy", "MeanReversionStrategy", "PairsZScoreStrategy", "WeightedEnsemble", "BreakoutStrategy", "VolatilityTargetStrategy"]
