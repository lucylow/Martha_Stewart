"""Leakage-aware validation and statistical backtest diagnostics."""
from .splits import purged_kfold,walk_forward,combinatorial_purged_cv,Fold
from .statistics import probabilistic_sharpe,deflated_sharpe,reality_check,probability_of_backtest_overfitting,benjamini_hochberg
__all__=["purged_kfold","walk_forward","combinatorial_purged_cv","Fold","probabilistic_sharpe","deflated_sharpe","reality_check","probability_of_backtest_overfitting","benjamini_hochberg"]
from .stability import moving_block_bootstrap,bootstrap_sharpe,rolling_parameter_stability,regime_performance,probability_of_loss
__all__ += ["moving_block_bootstrap","bootstrap_sharpe","rolling_parameter_stability","regime_performance","probability_of_loss"]
from .feature_importance import permutation_importance,ablation_importance,time_series_permutation
__all__ += ["permutation_importance","ablation_importance","time_series_permutation"]
from .splits import assert_no_overlap
__all__ += ["assert_no_overlap"]
