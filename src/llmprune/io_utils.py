"""Small I/O helpers (atomic JSON writes, provenance) with no heavy dependencies."""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

try:  # POSIX only; on other platforms file_lock() degrades to "no locking"
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj: Any, indent: int = 2) -> None:
    """Write JSON atomically (temp file + rename) so an interrupted run never leaves a
    truncated result that a later ``--skip-existing`` run would trust."""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=indent, default=_json_default)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_default(o):
    # numpy / torch scalars and arrays, Paths, sets
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(cwd: Optional[str] = None) -> Optional[str]:
    """Commit hash of the repository the code runs from (None outside a git checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return None


# Packages whose versions decide whether a re-run is comparable to the thesis (Table 10).
TRACKED_PACKAGES = ("torch", "transformers", "peft", "accelerate", "datasets", "lm_eval", "numpy", "safetensors", "tokenizers")


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> Dict[str, Optional[str]]:
    """Installed versions (None if a package is missing) without importing the packages."""
    from importlib import metadata

    out: Dict[str, Optional[str]] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def environment_info() -> Dict[str, Any]:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": package_versions()}


@contextlib.contextmanager
def file_lock(path, poll: float = 5.0, timeout: Optional[float] = None) -> Iterator[None]:
    """Exclusive inter-process lock (released automatically if the process dies).

    Used so that two shards that need the same pruning recipe never compute it twice:
    the second one waits, then finds the recipe on disk.
    """
    path = Path(path)
    ensure_dir(path.parent)
    fh = open(path, "a+")
    try:
        if fcntl is not None:
            t0 = time.time()
            while True:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if timeout is not None and time.time() - t0 > timeout:
                        raise TimeoutError(f"could not acquire {path} within {timeout} s")
                    time.sleep(poll)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fh, fcntl.LOCK_UN)
            except OSError:  # pragma: no cover
                pass
        fh.close()


def progress(iterable, desc: str = "", total: Optional[int] = None, disable: bool = False):
    """tqdm if installed, otherwise the iterable unchanged."""
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total, leave=False)
    except ImportError:
        return iterable
