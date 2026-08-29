"""Versioned artifact load/save with a metadata.json manifest per run."""
"""Versioned artifact storage.

Every training run writes to its own timestamped folder and then repoints
`latest`. Nothing is ever overwritten in place.

That sounds like bureaucracy until the first time a retrain makes the feed worse
and you need yesterday's model back. Overwrite in place and it is gone. It also
means metadata.json can answer 'what data was this trained on and how did it
score', which is the question you will be asked about every number in your
paper.
"""

import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib

LATEST = "latest"
METADATA_FILE = "metadata.json"


def config_hash(config: dict) -> str:
    """Short stable digest of a config dict.

    Lets you tell at a glance whether two runs used identical settings, without
    diffing two YAML dumps by eye.
    """
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def new_version() -> str:
    """UTC timestamp, sortable as a string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def save_artifacts(
    models: dict,
    artifact_dir: str | Path,
    config: dict | None = None,
    metrics: dict | None = None,
    data_summary: dict | None = None,
    version: str | None = None,
    make_latest: bool = True,
) -> Path:
    """Write every model plus a metadata manifest into a fresh version folder.

    models: {"content": obj, "collaborative": obj, ...}. Anything joblib can
    pickle. Objects that define their own save() are still fine - joblib will
    handle them - but if you later add a FAISS index or a LightGBM booster,
    give those their own branch here, since neither pickles cleanly.
    """
    artifact_dir = Path(artifact_dir)
    version = version or new_version()
    out_dir = artifact_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, model in models.items():
        if model is None:
            continue
        path = out_dir / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        saved.append({
            "name": name,
            "file": path.name,
            "class": type(model).__name__,
            "megabytes": round(path.stat().st_size / 1_048_576, 2),
        })

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "models": saved,
        "config_hash": config_hash(config) if config else None,
        "config": config,
        "data": data_summary,
        "metrics": metrics,
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))

    if make_latest:
        _point_latest(artifact_dir, version)

    return out_dir


def _point_latest(artifact_dir: Path, version: str) -> None:
    """Make `latest` refer to `version`.

    Symlinks need Developer Mode or admin rights on Windows, so fall back to a
    plain text pointer file. Slightly less elegant, works everywhere, and
    resolve_version() below handles both.
    """
    link = artifact_dir / LATEST
    target = artifact_dir / version

    try:
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(target.name, target_is_directory=True)
    except (OSError, NotImplementedError):
        (artifact_dir / "LATEST.txt").write_text(version)


def resolve_version(artifact_dir: str | Path, version: str = LATEST) -> Path:
    """Turn a version string into a real directory.

    Accepts an explicit timestamp, or 'latest' via symlink, pointer file, or -
    failing both - the newest folder by name. That last fallback matters:
    timestamped names sort chronologically, so it degrades gracefully instead
    of erroring when the pointer is missing.
    """
    artifact_dir = Path(artifact_dir)

    if version != LATEST:
        path = artifact_dir / version
        if not path.is_dir():
            raise FileNotFoundError(f"No artifact version {version} in {artifact_dir}")
        return path

    link = artifact_dir / LATEST
    if link.is_dir():
        return link

    pointer = artifact_dir / "LATEST.txt"
    if pointer.exists():
        path = artifact_dir / pointer.read_text().strip()
        if path.is_dir():
            return path

    candidates = sorted(
        p for p in artifact_dir.iterdir()
        if p.is_dir() and p.name != LATEST
    )
    if not candidates:
        raise FileNotFoundError(
            f"No artifacts in {artifact_dir}. Run the training pipeline first."
        )
    return candidates[-1]


def load_artifacts(artifact_dir: str | Path, version: str = LATEST) -> dict:
    """Load every model in a version folder, plus its metadata.

    Returns {"content": obj, ..., "_metadata": {...}, "_version": str}.
    """
    path = resolve_version(artifact_dir, version)

    out: dict = {}
    for file in sorted(path.glob("*.joblib")):
        out[file.stem] = joblib.load(file)

    metadata_path = path / METADATA_FILE
    out["_metadata"] = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    out["_version"] = out["_metadata"].get("version", path.name)

    return out


def list_versions(artifact_dir: str | Path) -> list[dict]:
    """Every saved run, newest first, with its headline metric.

    Useful on its own: run it after a few trainings and you have the history of
    what you tried and what it scored, without keeping notes.
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir():
        return []

    rows = []
    for folder in sorted(artifact_dir.iterdir(), reverse=True):
        if not folder.is_dir() or folder.name == LATEST:
            continue
        metadata_path = folder / METADATA_FILE
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        metrics = metadata.get("metrics") or {}
        rows.append({
            "version": folder.name,
            "created_at": metadata.get("created_at"),
            "config_hash": metadata.get("config_hash"),
            "ndcg@10": metrics.get("ndcg@10"),
            "models": [m["name"] for m in metadata.get("models", [])],
        })
    return rows


def prune_versions(artifact_dir: str | Path, keep: int = 5) -> list[str]:
    """Delete all but the newest `keep` versions.

    Never touches whatever `latest` points at, even if it falls outside the
    window. Call this manually, not from the training pipeline - automatic
    deletion of the artifact you were about to roll back to is a bad afternoon.
    """
    artifact_dir = Path(artifact_dir)
    current = resolve_version(artifact_dir).name

    folders = sorted(
        (p for p in artifact_dir.iterdir() if p.is_dir() and p.name != LATEST),
        reverse=True,
    )

    removed = []
    for folder in folders[keep:]:
        if folder.name == current:
            continue
        shutil.rmtree(folder)
        removed.append(folder.name)
    return removed