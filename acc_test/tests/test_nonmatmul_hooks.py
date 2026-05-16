"""Non-matmul hook registration on dummy Qwen-shaped model."""
from __future__ import annotations

import torch

from inject import InjectionContext, layer_site_ids, list_sites
from tests.test_inject import DummyModel, _run


def test_list_sites_nonmatmul_count():
    model = DummyModel(n_layers=3)
    assert len(list_sites(model, site_set="matmul")) == 24
    assert len(list_sites(model, site_set="nonmatmul")) == 15
    assert len(list_sites(model, site_set="all")) == 39


def test_nonmatmul_hooks_register():
    model = DummyModel(n_layers=1)
    x = torch.randn(1, 4, 8)
    with InjectionContext(model, target_site="L0_input_norm", site_set="nonmatmul") as inj:
        _ = _run(model, x)
        st = inj.collect_hook_stats()
    assert st.registered_site_count == 5
    assert st.missing_sites == []


def test_nonmatmul_inject_at_mlp_act():
    torch.manual_seed(0)
    model = DummyModel(n_layers=1)
    x = torch.randn(1, 4, 8)
    with InjectionContext(
        model,
        target_site="L0_mlp_act",
        fault_delta=1e4,
        seed=1,
        threshold_enable=False,
        site_set="nonmatmul",
    ) as inj:
        _ = _run(model, x)
    assert inj.inject_count == 1
