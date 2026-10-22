"""Optional Flask API integration; Flask remains an optional dependency."""

def create_blueprint(*args, **kwargs):
    try:
        from .flask_blueprint import create_blueprint as factory
    except ImportError as exc:
        raise RuntimeError("Flask is required to mount the optional operations API") from exc
    return factory(*args, **kwargs)

__all__ = ["create_blueprint"]
