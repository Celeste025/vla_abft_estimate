from __future__ import annotations

import torch

from inject import InjectionContext, list_sites


class DummyAttn(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = q + k + v
        y = self.o_proj(attn_out)
        return y, None


class DummyMLP(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) + self.up_proj(x))


class DummyNorm(torch.nn.Module):
    def forward(self, x):
        return x


class DummyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.input_layernorm = DummyNorm()
        self.self_attn = DummyAttn(hidden_size)
        self.post_attention_layernorm = DummyNorm()
        self.mlp = DummyMLP(hidden_size)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        attn_out, _ = self.self_attn(x)
        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Inner(torch.nn.Module):
    def __init__(self, n_layers: int, hidden_size: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummyLayer(hidden_size) for _ in range(n_layers)])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class DummyModel(torch.nn.Module):
    def __init__(self, n_layers: int = 2, hidden_size: int = 8):
        super().__init__()
        self.config = type("cfg", (), {"num_hidden_layers": n_layers})()
        self.model = Inner(n_layers, hidden_size)

    def forward(self, input_ids=None, attention_mask=None):
        x = input_ids.float()
        return type("out", (), {"logits": self.model(x)})


def _run(model: DummyModel, x: torch.Tensor):
    return model(input_ids=x).logits


def test_site_count_matches_structure():
    model = DummyModel(n_layers=3)
    x = torch.randn(1, 4, 8)
    with InjectionContext(model, target_site=None) as inj:
        _ = _run(model, x)
        st = inj.collect_hook_stats()
    assert st.expected_site_count == 24
    assert st.registered_site_count == 24
    assert st.missing_sites == []
    assert len(list_sites(model)) == 24


def test_inject_only_target_site_changes_output():
    torch.manual_seed(0)
    model = DummyModel(n_layers=1)
    x = torch.randn(1, 4, 8)
    base = _run(model, x)
    captured_base = {}

    def _capture_q(_m, _in, out):
        captured_base["q"] = out.detach().clone()

    h0 = model.model.layers[0].self_attn.q_proj.register_forward_hook(_capture_q)
    _ = _run(model, x)
    h0.remove()

    captured_inj = {}
    with InjectionContext(
        model,
        target_site="L0_q_proj",
        fault_delta=10000.0,
        seed=42,
        threshold_enable=False,
    ) as inj:
        def _capture_q2(_m, _in, out):
            captured_inj["q"] = out.detach().clone()

        h1 = model.model.layers[0].self_attn.q_proj.register_forward_hook(_capture_q2)
        out = _run(model, x)
        h1.remove()
        assert inj.inject_count == 1

    diff = (captured_inj["q"] - captured_base["q"]).reshape(-1)
    nz = torch.nonzero(diff != 0, as_tuple=False).reshape(-1)
    assert nz.numel() == 1
    assert torch.isclose(diff[nz[0]], torch.tensor(10000.0))

    diff = (out - base).reshape(-1)
    nz = torch.nonzero(diff != 0, as_tuple=False).reshape(-1)
    assert nz.numel() > 0


def test_warmup_collects_bounds_and_clean_forward_is_normal():
    torch.manual_seed(1)
    model = DummyModel(n_layers=1)
    x = torch.randn(1, 4, 8)
    with InjectionContext(
        model,
        target_site="L0_q_proj",
        thr_gamma=3.0,
        threshold_enable=True,
        inject_enable=False,
    ) as inj:
        inj.set_warmup(True)
        _ = _run(model, x)
        assert "L0_q_proj" in inj._site_min_max
        inj.set_warmup(False)
        inj.reset_acc_metrics()
        _ = _run(model, x)
        m = inj.get_acc_metrics()
    assert m["runs"] >= 1
    assert m["normal"] >= 1
    assert m["fp"] == 0
