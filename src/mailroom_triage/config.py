"""Config loading with paths resolved inside the project."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "base.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("data_root", "artifacts_dir", "reports_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else (ROOT / p).resolve())
    cfg["project_root"] = str(ROOT)
    return cfg


def ensure_dirs(cfg: dict) -> None:
    Path(cfg["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["reports_dir"]).mkdir(parents=True, exist_ok=True)
