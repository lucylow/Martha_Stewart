"""Domain-specific errors used to keep failures explicit and recoverable."""

class MarthaPlusError(Exception):
    """Base class for expected application errors."""

class ConfigurationError(MarthaPlusError):
    """Raised for invalid or unsafe configuration."""

class DataValidationError(MarthaPlusError):
    """Raised when incoming market data violates the contract."""

class RiskRejected(MarthaPlusError):
    """Raised when risk controls reject a proposed order."""

class ExecutionError(MarthaPlusError):
    """Raised when an execution adapter cannot process an order."""

class DuplicateOrder(MarthaPlusError):
    """Raised when an idempotency key is reused for a different order."""

class StorageError(MarthaPlusError):
    """Raised when persistence fails in a known, handled manner."""

class StrategyError(MarthaPlusError):
    """Raised when a strategy receives invalid state or parameters."""
