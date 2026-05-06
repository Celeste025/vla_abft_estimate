from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from inject import InjectionContext


class LatencyHook:
    """Wrap a model with forward pre/post hooks.

    First forward call is treated as prefill (full prompt forward), every
    subsequent forward call is one decode step (1 new token + KV cache).
    Each call is fenced with cuda.synchronize() so we measure GPU wall time,
    not CUDA launch time.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.prefill_ns: Optional[int] = None
        self.decode_ns_list: List[int] = []
        self._t0_ns: Optional[int] = None
        self._is_first: bool = True
        self._h_pre = model.register_forward_pre_hook(self._pre)
        self._h_post = model.register_forward_hook(self._post)

    def _pre(self, _module: torch.nn.Module, _args) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0_ns = time.perf_counter_ns()

    def _post(self, _module: torch.nn.Module, _args, _output) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._t0_ns is None:
            return
        dt = time.perf_counter_ns() - self._t0_ns
        self._t0_ns = None
        if self._is_first:
            self.prefill_ns = int(dt)
            self._is_first = False
        else:
            self.decode_ns_list.append(int(dt))

    def reset(self) -> None:
        self.prefill_ns = None
        self.decode_ns_list = []
        self._t0_ns = None
        self._is_first = True

    def close(self) -> None:
        self._h_pre.remove()
        self._h_post.remove()

    def __enter__(self) -> "LatencyHook":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


def get_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


@dataclass
class RunMeta:
    model_id: str
    device: str
    dtype: str
    attn_implementation: Optional[str]
    inject_site: Optional[str]
    fault_delta: float
    inject_count: int
    expected_site_count: int
    registered_site_count: int
    missing_sites: list[str]
    injected_forward_count: int
    bad_forward_count: int
    errors_total: int
    warning_printed: int
    decode_problem_count: int
    decode_injected_problem_count: int


class ModelRunner:
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.dtype = get_dtype(dtype)
        self.attn_implementation = attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {"torch_dtype": self.dtype}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.eval().to(device)
        self._active_injector: Optional[InjectionContext] = None

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        tok = self.tokenizer
        inputs = tok(prompt, return_tensors="pt").to(self.device)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": temperature > 1e-9,
            "temperature": float(temperature) if temperature > 1e-9 else None,
            "top_p": float(top_p),
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        # Remove None to avoid warnings in some transformers versions.
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        injector = self._active_injector
        if injector is not None and injector.decode_step_inject_enable:
            injector.begin_decode(step_max=injector.decode_step_max)
            gen_kwargs["logits_processor"] = LogitsProcessorList([_DecodeStepCounter(injector)])

        out_ids = self.model.generate(**inputs, **gen_kwargs)
        if injector is not None and injector.decode_step_inject_enable:
            injector.end_decode()
        # Return only the newly generated part.
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_ids = out_ids[0, prompt_len:]
        return tok.decode(gen_ids, skip_special_tokens=True)

    def run_task(
        self,
        task,
        inject_site: Optional[str] = None,
        fault_delta: float = 10000.0,
        seed: int = 2026,
        fault_index_mode: str = "random",
        clear_exceptions: bool = False,
        clear_threshold_mul: float = 0.5,
        decode_step_inject_enable: bool = False,
        decode_step_max: int = 150,
    ) -> Dict[str, Any]:
        with InjectionContext(
            model=self.model,
            target_site=inject_site,
            fault_delta=fault_delta,
            seed=seed,
            fault_index_mode=fault_index_mode,
            clear_exceptions=clear_exceptions,
            clear_threshold_mul=clear_threshold_mul,
            decode_step_inject_enable=decode_step_inject_enable,
            decode_step_max=decode_step_max,
        ) as inj:
            self._active_injector = inj
            result = task.run(self)
            self._active_injector = None
            stats = inj.collect_hook_stats()
            meta = RunMeta(
                model_id=self.model_id,
                device=self.device,
                dtype=self.dtype_name,
                attn_implementation=self.attn_implementation,
                inject_site=inject_site,
                fault_delta=fault_delta,
                inject_count=inj.inject_count,
                expected_site_count=stats.expected_site_count,
                registered_site_count=stats.registered_site_count,
                missing_sites=stats.missing_sites,
                injected_forward_count=stats.injected_forward_count,
                bad_forward_count=stats.bad_forward_count,
                errors_total=stats.errors_total,
                warning_printed=stats.warning_printed,
                decode_problem_count=stats.decode_problem_count,
                decode_injected_problem_count=stats.decode_injected_problem_count,
            )
            result["run_meta"] = asdict(meta)
            return result


class _DecodeStepCounter(LogitsProcessor):
    def __init__(self, injector: InjectionContext):
        super().__init__()
        self.injector = injector
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Called once per generation step in HF generate loop.
        self.injector.set_decode_step(self.step)
        self.step += 1
        return scores
