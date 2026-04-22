#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abft import AbftConfig, AbftInjector, AbftStatsCollector, CheapSumChecker
from abft.stats import summarize_ms


def parse_int_csv(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Qwen2.5-1.5B PyTorch ABFT benchmark")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--prompt-lens", default="128,512,1024")
    ap.add_argument("--gen-lens", default="32,128")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup-prefill", type=int, default=100)
    ap.add_argument("--warmup-decode", type=int, default=100)
    ap.add_argument("--iters-prefill", type=int, default=300)
    ap.add_argument("--iters-decode", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--abft-enable", action="store_true")
    ap.add_argument("--abft-check-disable", action="store_true")
    ap.add_argument("--abft-record-disable", action="store_true")
    ap.add_argument("--abft-sample-rate", type=float, default=1.0)
    ap.add_argument("--abft-phase", default="all", choices=["all", "prefill", "decode"])
    ap.add_argument("--abft-tol-abs", type=float, default=1e-2)
    ap.add_argument("--abft-tol-rel", type=float, default=1e-2)
    ap.add_argument("--abft-disable-linear", action="store_true")
    ap.add_argument("--abft-disable-matmul", action="store_true")
    ap.add_argument("--abft-disable-bmm", action="store_true")
    ap.add_argument("--inject-fault", action="store_true")
    ap.add_argument("--inject-phase", default="all", choices=["all", "prefill", "decode"])
    ap.add_argument("--inject-probability", type=float, default=0.0)
    ap.add_argument("--inject-magnitude", type=float, default=1.0)
    ap.add_argument("--dump-shape-key", default="", help="匹配到该shape_key时导出A/B/C/bias")
    ap.add_argument("--dump-phase", default="all", choices=["all", "prefill", "decode"])
    ap.add_argument("--dump-file", default="", help="导出文件路径(.pt)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--prompt-text",
        default=(
            "你是一个有帮助的中文助手。请阅读用户问题，"
            "先给出简洁结论，再给出关键依据与可执行建议。"
        ),
        help="构造输入时使用的基础提示词文本",
    )
    ap.add_argument("--out-json", default="qwen_abft_benchmark_out.json")
    ap.add_argument("--verbose", action="store_true", help="在终端打印每次迭代输入输出与时延")
    ap.add_argument("--verbose-max-tokens", type=int, default=32, help="verbose模式下最多打印多少token")
    return ap.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def build_model_and_tokenizer(model_id: str, dtype: torch.dtype, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()
    model.to(device)
    return model, tok


def make_prompt(tokenizer: AutoTokenizer, token_len: int, base_prompt: str) -> str:
    # 使用通用 LLM 风格文本，避免单短语机械重复。
    detail_chunks = [
        "请将回答拆成：结论、分析、建议三部分。",
        "分析时列出前提假设，并指出可能的不确定性来源。",
        "建议部分请按优先级排序，并说明每条建议的预期收益。",
        "如果信息不足，请明确需要补充的上下文信息。",
    ]
    text = base_prompt + " " + " ".join(detail_chunks)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < token_len:
        pad_text = " 请继续补充更完整的解释、示例和边界条件说明。"
        pad_ids = tokenizer(pad_text, add_special_tokens=False)["input_ids"]
        while len(ids) < token_len:
            ids.extend(pad_ids)
    if len(ids) < token_len:
        ids = ids + [tokenizer.eos_token_id] * (token_len - len(ids))
    else:
        ids = ids[:token_len]
    return tokenizer.decode(ids, skip_special_tokens=False)


def _elapsed_ms(st: torch.cuda.Event, ed: torch.cuda.Event) -> float:
    return float(st.elapsed_time(ed))


def _tokens_per_s(tokens: int, latency_ms: float) -> float:
    if latency_ms <= 1e-9:
        return 0.0
    return float(tokens) * 1000.0 / float(latency_ms)


@torch.no_grad()
def run_one_prefill(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    stats: AbftStatsCollector,
    prompt_len: int,
) -> Tuple[Any, float]:
    stats.set_phase("prefill")
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    st.record()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    ed.record()
    torch.cuda.synchronize()
    ms = _elapsed_ms(st, ed)
    stats.record_prefill_latency(prompt_len=prompt_len, ms=ms)
    return out.past_key_values, ms


@torch.no_grad()
def run_decode_steps(
    model: AutoModelForCausalLM,
    past_key_values: Any,
    start_token: int,
    decode_len: int,
    stats: AbftStatsCollector,
    prompt_len: int,
    device: str,
    collect_tokens: bool = False,
) -> Tuple[float, List[int]]:
    token = torch.tensor([[start_token]], device=device, dtype=torch.long)
    st_total = torch.cuda.Event(enable_timing=True)
    ed_total = torch.cuda.Event(enable_timing=True)
    st_total.record()
    generated_tokens: List[int] = []
    for _ in range(decode_len):
        stats.set_phase("decode")
        out = model(
            input_ids=token,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]
        token = torch.argmax(logits, dim=-1, keepdim=True)
        if collect_tokens:
            generated_tokens.append(int(token[0, 0].item()))
        past_key_values = out.past_key_values
    ed_total.record()
    torch.cuda.synchronize()
    total_ms = _elapsed_ms(st_total, ed_total)
    step_ms = total_ms / float(max(decode_len, 1))
    stats.record_decode_step_latency(prompt_len=prompt_len, ms=step_ms)
    return total_ms, generated_tokens


def run_suite(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    dtype = get_dtype(args.dtype)
    model, tokenizer = build_model_and_tokenizer(args.model_id, dtype=dtype, device=device)
    prompt_lens = parse_int_csv(args.prompt_lens)
    gen_lens = parse_int_csv(args.gen_lens)

    cfg = AbftConfig(
        enable=args.abft_enable,
        check_enable=not args.abft_check_disable,
        record_enable=not args.abft_record_disable,
        enable_linear=not args.abft_disable_linear,
        enable_matmul=not args.abft_disable_matmul,
        enable_bmm=not args.abft_disable_bmm,
        phase=args.abft_phase,
        sample_rate=args.abft_sample_rate,
        tol_abs=args.abft_tol_abs,
        tol_rel=args.abft_tol_rel,
        seed=args.seed,
        inject_fault=args.inject_fault,
        inject_phase=args.inject_phase,
        inject_probability=args.inject_probability,
        inject_magnitude=args.inject_magnitude,
        dump_shape_key=args.dump_shape_key,
        dump_phase=args.dump_phase,
        dump_file=args.dump_file,
    )
    stats = AbftStatsCollector()
    checker = CheapSumChecker(cfg)

    prefill_global: List[float] = []
    decode_step_global: List[float] = []
    request_global: List[float] = []
    per_case: Dict[str, Dict[str, Dict[str, float]]] = {}

    with AbftInjector(cfg=cfg, checker=checker, stats=stats):
        for prompt_len in prompt_lens:
            prompt = make_prompt(tokenizer, prompt_len, args.prompt_text)
            enc = tokenizer(
                [prompt] * args.batch_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            start_token = int(tokenizer.eos_token_id)
            case_key = f"prompt_{prompt_len}"
            per_case[case_key] = {}

            for _ in range(args.warmup_prefill):
                run_one_prefill(model, input_ids, attention_mask, stats, prompt_len)
            for _ in range(args.warmup_decode):
                pkv, _ = run_one_prefill(model, input_ids, attention_mask, stats, prompt_len)
                _ = run_decode_steps(
                    model=model,
                    past_key_values=pkv,
                    start_token=start_token,
                    decode_len=min(gen_lens),
                    stats=stats,
                    prompt_len=prompt_len,
                    device=device,
                    collect_tokens=False,
                )

            for gen_len in gen_lens:
                prefill_samples: List[float] = []
                decode_step_samples: List[float] = []
                e2e_samples: List[float] = []
                for _ in range(args.iters_prefill):
                    iter_idx = len(prefill_samples) + 1
                    pkv, prefill_ms = run_one_prefill(model, input_ids, attention_mask, stats, prompt_len)
                    prefill_samples.append(prefill_ms)
                    prefill_global.append(prefill_ms)

                    decode_total, gen_token_ids = run_decode_steps(
                        model=model,
                        past_key_values=pkv,
                        start_token=start_token,
                        decode_len=gen_len,
                        stats=stats,
                        prompt_len=prompt_len,
                        device=device,
                        collect_tokens=args.verbose,
                    )
                    decode_step = decode_total / float(max(gen_len, 1))
                    decode_step_samples.append(decode_step)
                    decode_step_global.append(decode_step)
                    request_ms = prefill_ms + decode_total
                    e2e_samples.append(request_ms)
                    request_global.append(request_ms)
                    stats.record_e2e_latency(prompt_len=prompt_len, ms=request_ms)
                    if args.verbose:
                        prefill_tps = _tokens_per_s(prompt_len * args.batch_size, prefill_ms)
                        decode_tps = _tokens_per_s(gen_len * args.batch_size, decode_total)
                        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                        output_text = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
                        print(
                            "[iter]"
                            f" prompt_len={prompt_len}"
                            f" gen_len={gen_len}"
                            f" idx={iter_idx}/{args.iters_prefill}"
                        )
                        print(f"  input_text={input_text!r}")
                        print(f"  output_text={output_text!r}")
                        print(
                            f"  result_ms prefill={prefill_ms:.3f} "
                            f"decode_total={decode_total:.3f} "
                            f"decode_step={decode_step:.3f} "
                            f"e2e={request_ms:.3f}"
                        )
                        print(
                            f"  speed_tps prefill={prefill_tps:.2f} token/s "
                            f"decode={decode_tps:.2f} token/s"
                        )

                per_case[case_key][f"gen_{gen_len}"] = {
                    "prefill": summarize_ms(prefill_samples),
                    "decode_step": summarize_ms(decode_step_samples),
                    "e2e_request": summarize_ms(e2e_samples),
                    "throughput_tps": {
                        "prefill_mean_tps": _tokens_per_s(
                            prompt_len * args.batch_size,
                            summarize_ms(prefill_samples)["mean_ms"],
                        ),
                        "decode_mean_tps": _tokens_per_s(
                            gen_len * args.batch_size,
                            summarize_ms(decode_step_samples)["mean_ms"] * gen_len,
                        ),
                    },
                }

    prefill_summary = summarize_ms(prefill_global)
    decode_summary = summarize_ms(decode_step_global)
    e2e_summary = summarize_ms(request_global)

    out = {
        "config": {
            "model_id": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "prompt_lens": prompt_lens,
            "gen_lens": gen_lens,
            "batch_size": args.batch_size,
            "warmup_prefill": args.warmup_prefill,
            "warmup_decode": args.warmup_decode,
            "iters_prefill": args.iters_prefill,
            "iters_decode": args.iters_decode,
            "abft": asdict(cfg),
        },
        "global_latency": {
            "prefill": prefill_summary,
            "decode_step": decode_summary,
            "e2e_request": e2e_summary,
        },
        "global_throughput_tps": {
            "prefill_mean_tps_by_prompt_len": {
                str(pl): _tokens_per_s(pl * args.batch_size, prefill_summary["mean_ms"])
                for pl in prompt_lens
            },
            "decode_mean_tps": _tokens_per_s(1, decode_summary["mean_ms"]),
        },
        "by_prompt_and_gen": per_case,
        "stats": stats.export_summary(),
    }
    return out


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境来执行该基准脚本。")

    st = time.time()
    out = run_suite(args)
    out["run_seconds"] = time.time() - st

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Benchmark done. JSON written to {args.out_json}")


if __name__ == "__main__":
    main()
