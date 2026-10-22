"""File-backed emergency halt. Presence means trading is halted; deletion is explicit."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

class KillSwitch:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = RLock()

    @property
    def active(self) -> bool:
        return self.path.exists()

    def activate(self, reason: str = "manual halt") -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            payload = f"timestamp={datetime.now(timezone.utc).isoformat()}\nreason={reason.strip() or 'unspecified'}\n"
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(self.path)

    def deactivate(self) -> None:
        with self._lock:
            self.path.unlink(missing_ok=True)

    def reason(self) -> str:
        if not self.path.is_file():
            return ""
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            return "kill switch file exists; reason unavailable"
