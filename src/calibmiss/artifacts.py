"""Save and load trained pipeline artifacts (imputer, model, calibrator, scaler)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def save_artifacts(artifacts: dict[str, Any], path: Path, manifest: dict[str, Any]) -> Path:
    """Serialize fitted artifacts to `path` alongside a manifest of metadata.

    Args:
        artifacts: Mapping of artifact name to fitted object (imputer, model,
            calibrator, scaler, ...). Objects must be joblib-serializable.
        path: Directory to write into (created if needed).
        manifest: Metadata to persist as manifest.json (config snapshot,
            fit window, run timestamp, ...).

    Returns:
        The directory written to.
    """
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise ValueError(f"{path} exists and is not a directory")
    path.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        joblib.dump(obj, path / f"{name}.joblib")
    record = {"artifacts": sorted(artifacts), **manifest}
    record.setdefault("saved_at_utc", datetime.now(UTC).isoformat())
    (path / "manifest.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    return path


def load_artifacts(
    path: Path,
    expected_experiment: str | None = None,
) -> dict[str, Any]:
    """Load artifacts saved by :func:`save_artifacts`.

    Args:
        path: Directory written by save_artifacts.
        expected_experiment: If given, verify manifest[config][experiment_file]
            matches before loading; raises otherwise.

    Returns:
        Mapping of artifact name to deserialized object.
    """
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"No manifest.json in {path}; not a saved artifact set")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_experiment is not None:
        recorded = (manifest.get("config") or {}).get("experiment_file")
        if recorded != expected_experiment:
            raise ValueError(
                f"Manifest records experiment_file={recorded!r}, "
                f"expected {expected_experiment!r}"
            )
    artifacts = {}
    for name in manifest.get("artifacts", []):
        file = path / f"{name}.joblib"
        if not file.exists():
            raise ValueError(f"Artifact {name!r} missing expected file {file.name}")
        artifacts[name] = joblib.load(file)
    return artifacts
