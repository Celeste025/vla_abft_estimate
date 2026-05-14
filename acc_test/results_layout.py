"""results/{model_slug}_{dataset}/{run_config}/plots|csv|json layout for ACC experiments."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional


def model_slug_from_id(model_id: str) -> str:
    s = str(model_id).strip().lower().replace("/", "-")
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "model"


def dataset_slug(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", str(name).strip().lower()) or "data"


def build_run_config_segment(
    *,
    n_total: int,
    n_warmup: int,
    gamma: float,
    fault_mode: str,
    seed: int,
    max_new_tokens: Optional[int] = None,
    fault_delta: Optional[float] = None,
) -> str:
    """Path segment (no leading/trailing slashes). Examples:
    n200_wu5_g3_thr-mMg_fm-rand2pow_s2026_mnt64
    n200_wu5_g3_thr-mMg_fm-fixed_fd10000_s2026_mnt64
    """
    fm = str(fault_mode).strip().lower()
    if fm not in {"rand2pow", "fixed"}:
        raise ValueError(f"fault_mode must be rand2pow|fixed, got {fault_mode!r}")
    parts = [
        f"n{int(n_total)}",
        f"wu{int(n_warmup)}",
        f"g{float(gamma)}",
        "thr-mMg",
        f"fm-{fm}",
    ]
    if fm == "fixed":
        if fault_delta is None:
            raise ValueError("fault_delta required when fault_mode=fixed")
        fd = float(fault_delta)
        parts.append(f"fd{fd:g}".replace("+", ""))
    parts.append(f"s{int(seed)}")
    if max_new_tokens is not None:
        parts.append(f"mnt{int(max_new_tokens)}")
    return "_".join(parts)


def results_run_dir(
    results_root: Path | str,
    *,
    model_id: str,
    dataset: str,
    n_total: int,
    n_warmup: int,
    gamma: float,
    fault_mode: str,
    seed: int,
    max_new_tokens: Optional[int] = None,
    fault_delta: Optional[float] = None,
) -> Path:
    root = Path(results_root)
    slug_m = model_slug_from_id(model_id)
    slug_d = dataset_slug(dataset)
    seg = build_run_config_segment(
        n_total=n_total,
        n_warmup=n_warmup,
        gamma=gamma,
        fault_mode=fault_mode,
        seed=seed,
        max_new_tokens=max_new_tokens,
        fault_delta=fault_delta,
    )
    return root / f"{slug_m}_{slug_d}" / seg


def ensure_results_subdirs(run_dir: Path) -> Dict[str, Path]:
    plots = run_dir / "plots"
    csv = run_dir / "csv"
    json_dir = run_dir / "json"
    plots.mkdir(parents=True, exist_ok=True)
    csv.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return {"run_dir": run_dir, "plots": plots, "csv": csv, "json": json_dir}


def write_run_meta(paths: Dict[str, Path], meta: Dict[str, Any]) -> Path:
    import json

    p = paths["json"] / "run_meta.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return p
