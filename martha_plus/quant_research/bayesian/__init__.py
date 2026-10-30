"""Conjugate Bayesian return and factor-model estimation."""
from .conjugate import normal_normal_posterior,normal_inverse_gamma_posterior,bayesian_linear_regression,bayesian_shrinkage_mean
__all__=["normal_normal_posterior","normal_inverse_gamma_posterior","bayesian_linear_regression","bayesian_shrinkage_mean"]
from .ensemble import bayesian_model_average,log_score_weights,posterior_predictive_average
__all__ += ["bayesian_model_average","log_score_weights","posterior_predictive_average"]
