"""Per-operator-type max ratio vs decoder layer.

For each hook site (same calibration as plot_op_stats_calibration_ratio.py), compute
r_M(t), r_m(t) over testcase index t, then:

  max_rM = max_t r_M(t),   max_rm = max_t r_m(t)

X-axis: layer index 0 .. num_layers-1.
Y-axis: one of ``--y-metric`` (default: plot **two** PNGs: max_rM and max_rm).

Eight lines per figure: q_proj, k_proj, v_proj, attn_core, o_proj, mlp_gate, mlp_up, mlp_down.

Outputs under ``<data-dir>/plots/``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_op_stats_calibration_ratio import _find_capture_json, _load, _ratios_for_site

SUFFIX_ORDER = [
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_core",
    "o_proj",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
]

_LABEL = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "attn_core": "attn_core",
    "o_proj": "o_proj",
    "mlp_gate": "mlp_gate",
    "mlp_up": "mlp_up",
    "mlp_down": "mlp_down",
}

_SITE_RE = re.compile(r"^L(\d+)_(.+)$")


def _parse_site(site_id: str) -> Tuple[int, str] | None:
    m = _SITE_RE.match(site_id)
    if not m:
        return None
    return int(m.group(1)), str(m.group(2))


def _build_per_suffix_max(
    site_ids: List[str],
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    calib_k: int,
    eps: float,
    which: str,
) -> Tuple[int, Dict[str, np.ndarray]]:
    """Return num_layers and dict suffix -> length-num_layers array of max ratio (nan if missing)."""
    max_layer = -1
    parsed: List[Tuple[int, str, str]] = []
    for sid in site_ids:
        pr = _parse_site(sid)
        if pr is None:
            continue
        layer, suf = pr
        if suf not in SUFFIX_ORDER:
            continue
        max_layer = max(max_layer, layer)
        parsed.append((layer, suf, sid))

    if max_layer < 0:
        raise ValueError("no parsable sites")
    n_layers = max_layer + 1
    out: Dict[str, np.ndarray] = {s: np.full(n_layers, np.nan, dtype=np.float64) for s in SUFFIX_ORDER}

    for layer, suf, sid in parsed:
        mins, maxs = series[sid]
        r_M, r_m, _, _ = _ratios_for_site(mins, maxs, calib_k, eps)
        if which == "M":
            v = float(np.nanmax(r_M)) if np.any(np.isfinite(r_M)) else float("nan")
        elif which == "m":
            v = float(np.nanmax(r_m)) if np.any(np.isfinite(r_m)) else float("nan")
        else:
            raise ValueError(which)
        out[suf][layer] = v

    return n_layers, out


def _plot(
    n_layers: int,
    data: Dict[str, np.ndarray],
    ylabel: str,
    title: str,
    out_path: str,
) -> None:
    x = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    for i, suf in enumerate(SUFFIX_ORDER):
        ax.plot(x, data[suf], markers[i % len(markers)] + "-", label=_LABEL[suf], markersize=4, linewidth=1.2)
    ax.set_xlabel("layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(range(0, n_layers, max(1, n_layers // 14))))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--calib-k", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    cap = _find_capture_json(data_dir)
    site_ids, n_tc, series = _load(cap)
    calib_k = int(args.calib_k)
    eps = float(args.eps)

    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    n_layers, max_M = _build_per_suffix_max(site_ids, series, calib_k, eps, "M")
    _, max_m = _build_per_suffix_max(site_ids, series, calib_k, eps, "m")

    out_M = os.path.join(
        plots_dir,
        f"ratio_max_M_all_over_M_by_layer_calib{calib_k}.png",
    )
    out_m = os.path.join(
        plots_dir,
        f"ratio_max_m_all_over_m_by_layer_calib{calib_k}.png",
    )
    _plot(
        n_layers,
        max_M,
        r"$\max_t\,(M_{\mathrm{all}}/M)$",
        f"Max over {n_tc} testcases of M_all/M vs layer (calib first {calib_k} cases)",
        out_M,
    )
    _plot(
        n_layers,
        max_m,
        r"$\max_t\,(m_{\mathrm{all}}/m)$",
        f"Max over {n_tc} testcases of m_all/m vs layer (calib first {calib_k} cases)",
        out_m,
    )

    serial = {
        "capture_json": cap,
        "calib_k": calib_k,
        "n_testcases": n_tc,
        "n_layers": n_layers,
        "max_M_all_over_M_by_layer": {s: max_M[s].tolist() for s in SUFFIX_ORDER},
        "max_m_all_over_m_by_layer": {s: max_m[s].tolist() for s in SUFFIX_ORDER},
        "plot_max_M": out_M,
        "plot_max_m": out_m,
    }
    with open(
        os.path.join(plots_dir, f"ratio_max_by_layer_meta_calib{calib_k}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(serial, f, ensure_ascii=False, indent=2)

    print(json.dumps({"plots": [out_M, out_m], "n_layers": n_layers}, ensure_ascii=False))


if __name__ == "__main__":
    main()
