"""Integration tests for ACC inject + sweep-shaped workflows (no GPU / HF model)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ACC_ROOT = Path(__file__).resolve().parents[1]
if str(_ACC_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACC_ROOT))

from inject import InjectionContext  # noqa: E402
from tests.test_inject import DummyModel, _run  # noqa: E402


def _sweep_one_site(
    model: DummyModel,
    *,
    site: str,
    n_warmup: int,
    n_formal: int,
    gamma: float,
    threshold_enable: bool,
    restore_mode: str,
    inject_enable: bool,
    fault_delta: float,
) -> tuple[dict, int]:
    xs = [torch.randn(1, 4, 8) for _ in range(n_warmup + n_formal)]
    inj = InjectionContext(
        model,
        target_site=site,
        fault_delta=fault_delta,
        seed=42,
        thr_gamma=gamma,
        threshold_enable=threshold_enable,
        restore_mode=restore_mode,
        inject_enable=inject_enable,
        fault_mode="fixed",
    )
    with inj:
        inj.reset_site_bounds()
        inj.set_warmup(True)
        for x in xs[:n_warmup]:
            _ = _run(model, x)
        inj.set_warmup(False)
        inj.reset_acc_metrics()
        for x in xs[n_warmup:]:
            _ = _run(model, x)
        n_inj = inj.inject_count
    return inj.get_acc_metrics(), n_inj


def test_sweep_warmup_then_inject_golden_catches_fault():
    torch.manual_seed(0)
    model = DummyModel(n_layers=2)
    m, n_inj = _sweep_one_site(
        model,
        site="L0_q_proj",
        n_warmup=3,
        n_formal=2,
        gamma=3.0,
        threshold_enable=True,
        restore_mode="golden",
        inject_enable=True,
        fault_delta=1e4,
    )
    assert n_inj == 2
    assert m["runs"] >= 2
    assert m["tp"] >= 1


def test_sweep_no_threshold_injects_without_tp():
    torch.manual_seed(1)
    model = DummyModel(n_layers=2)
    inj = InjectionContext(
        model,
        target_site="L1_mlp_gate",
        fault_delta=500.0,
        seed=7,
        threshold_enable=False,
        inject_enable=True,
    )
    with inj:
        _ = _run(model, torch.randn(1, 4, 8))
        n_inj = inj.inject_count
    m = inj.get_acc_metrics()
    assert n_inj == 1
    assert m["runs"] >= 1
    assert m["tp"] == 0
    assert m["fp"] == 0


def test_threshold_monitor_all_sites_metrics():
    torch.manual_seed(2)
    model = DummyModel(n_layers=1)
    inj = InjectionContext(
        model,
        target_site=None,
        threshold_enable=True,
        inject_enable=False,
        metrics_scope="all",
        fault_mode="none",
    )
    with inj:
        inj.reset_site_bounds()
        inj.set_warmup(True)
        _ = _run(model, torch.randn(1, 4, 8))
        inj.set_warmup(False)
        inj.reset_acc_metrics()
        _ = _run(model, torch.randn(1, 4, 8))
    by_site = inj.get_acc_metrics_by_site()
    assert len(by_site) == 8
    assert all("runs" in v for v in by_site.values())
