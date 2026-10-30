"""Advanced quantitative research extensions for Martha_Stewart.

This namespace is dependency-free and deliberately research-oriented. It does
not create brokerage connections, send orders, or promise investment returns.
"""
__version__ = "0.1.0"

from .core import TimeSeries

__all__ = ["TimeSeries", "__version__"]
