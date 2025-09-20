import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _TrackedDict(dict):
    def __init__(self, data: dict[str, Any], parent: "AutoSaveConfig"):
        super().__init__()
        self._parent = parent
        for k, v in data.items():
            super().__setitem__(k, self._wrap(v))

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, _TrackedDict):
            return _TrackedDict(value, self._parent)
        return value

    def _to_dict(self) -> dict[str, Any]:
        out = {}
        for k, v in self.items():
            out[k] = v._to_dict() if isinstance(v, _TrackedDict) else v
        return out

    def __setitem__(self, key, value):
        wrapped = self._wrap(value)
        if key in self and self[key] == wrapped:
            return  # no change
        super().__setitem__(key, wrapped)
        self._parent._mark_dirty()

    def __delitem__(self, key):
        if key in self:
            super().__delitem__(key)
            self._parent._mark_dirty()

    def clear(self):
        if self:
            super().clear()
            self._parent._mark_dirty()

    def pop(self, key, *args):
        if key in self:
            val = super().pop(key, *args)
            self._parent._mark_dirty()
            return val
        return super().pop(key, *args)

    def popitem(self):
        val = super().popitem()
        self._parent._mark_dirty()
        return val

    def update(self, *args, **kwargs):
        changed = False
        for k, v in dict(*args, **kwargs).items():
            wrapped = self._wrap(v)
            if k not in self or self[k] != wrapped:
                super().__setitem__(k, wrapped)
                changed = True
        if changed:
            self._parent._mark_dirty()

    def setdefault(self, key, default=None):
        if key not in self or self[key] != default:
            self[key] = default
        return self[key]


class AutoSaveConfig(_TrackedDict):
    def __init__(self, filepath: str, debounce: float = 0.5, *args, **kwargs):
        """
        debounce: time in seconds to delay writes after a change
        """
        self._path = Path(filepath).expanduser().resolve()
        self._debounce = debounce
        self._dirty = False
        self._timer: threading.Timer | None = None

        # Load existing JSON
        data: dict[str, Any] = {}
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # Merge with provided args
        data.update(dict(*args, **kwargs))

        super().__init__(data, parent=self)

        # Ensure file exists
        self._flush()

    def _mark_dirty(self):
        """Mark config dirty and schedule a debounced flush."""
        self._dirty = True
        if self._timer and self._timer.is_alive():
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self):
        if not self._dirty:
            return
        logger.debug(f"Saving config to {self._path}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, indent=2, ensure_ascii=False)
        self._dirty = False

    def save(self):
        """Force immediate save (e.g. on app exit)."""
        if self._timer and self._timer.is_alive():
            self._timer.cancel()
        self._flush()
