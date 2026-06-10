"""HellaSwag full-sequence teacher-forcing token log-probability vector (T, 1)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from model_runner import ModelRunner


def build_hellaswag_inputs(
    runner: ModelRunner, ctx: str, ending: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    tok = runner.tokenizer
    device = runner.device
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    end_ids = tok(" " + ending, add_special_tokens=False)["input_ids"]
    full_ids = ctx_ids + end_ids
    if len(full_ids) < 2 or len(end_ids) == 0:
        raise ValueError("prompt too short for scoring forward")
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    meta = {
        "seq_len": len(full_ids),
        "ctx_token_len": len(ctx_ids),
        "ending_token_len": len(end_ids),
        "predict_len": len(full_ids) - 1,
        "ctx_predict_end": max(len(ctx_ids) - 1, 0),
        "ending_predict_start": max(len(ctx_ids) - 1, 0),
        "ending_predict_end": max(len(ctx_ids) - 1, 0) + len(end_ids),
    }
    return input_ids, attention_mask, meta


def token_logp_vector(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-position log p(x_{t+1} | x_{<=t}); shape (S-1, 1).

    Args:
        logits: (S, vocab) or (1, S, vocab)
        input_ids: (S,) or (1, S)
    """
    if logits.ndim == 3:
        logits = logits[0]
    if input_ids.ndim == 2:
        input_ids = input_ids[0]
    if logits.shape[0] < 2:
        raise ValueError(f"need seq_len >= 2, got logits {tuple(logits.shape)}")
    log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
    target = input_ids[1:].long()
    token_logp = log_probs.gather(1, target.unsqueeze(-1))
    return token_logp.detach().cpu().float()


def score_ending_loglik(runner: ModelRunner, ctx: str, ending: str) -> float:
    """HellaSwag choice score: sum of log p on ending tokens (teacher forcing)."""
    input_ids, attention_mask, tok_meta = build_hellaswag_inputs(runner, ctx, ending)
    logits = runner.forward(input_ids=input_ids, attention_mask=attention_mask)
    vec = token_logp_vector(logits, input_ids)
    return ending_logp_sum(vec, tok_meta)


def score_four_choices(runner: ModelRunner, ctx: str, endings: Sequence[str]) -> List[float]:
    return [score_ending_loglik(runner, ctx, ending) for ending in endings]


def softmax_scores(scores: Sequence[float]) -> List[float]:
    """Numerically stable softmax over choice log-likelihoods."""
    if not scores:
        return []
    t = torch.tensor(list(scores), dtype=torch.float64)
    t = t - t.max()
    p = torch.exp(t)
    p = p / p.sum()
    return [float(x) for x in p.tolist()]


def ending_logp_sum(token_logp: torch.Tensor, tok_meta: Dict[str, Any]) -> float:
    """Sum log p over ending token positions (HellaSwag score)."""
    start = int(tok_meta["ending_predict_start"])
    end = int(tok_meta["ending_predict_end"])
    v = token_logp.reshape(-1)
    if end > start:
        return float(v[start:end].sum().item())
    return float("nan")


def save_token_logp(path: Path, token_logp: torch.Tensor, extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_logp": token_logp,
        "shape": list(token_logp.shape),
        **extra,
    }
    torch.save(payload, path)
