from __future__ import annotations

from dataclasses import dataclass
import gc
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypeVar

import torch
from datasets import load_dataset, load_from_disk

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

T = TypeVar("T")


# Hub 上完整路径为 ILSVRC/imagenet-1k（与短名 imagenet-1k 可能指向同一资源，此处显式使用组织路径）。
HF_IMAGENET_DATASET_ID = "ILSVRC/imagenet-1k"

# torchvision published accuracies (ImageNet-1k val), for delta reporting.
COMMUNITY_TOP1_TOP5 = {
    "IMAGENET1K_V1": (0.7613, 0.9286),
    "IMAGENET1K_V2": (0.8086, 0.9543),
}


def _hf_auth_hint(exc: BaseException) -> str:
    return (
        f"加载 HuggingFace `{HF_IMAGENET_DATASET_ID}` 失败。\n"
        "注意：登录账号 ≠ 自动能下 gated 数据集。除 `hf auth login` 外还必须：\n"
        f"  1) 浏览器打开 https://huggingface.co/datasets/{HF_IMAGENET_DATASET_ID}\n"
        "  2) 在页面里点击「Agree and access repository」/「Request access」并接受条款；\n"
        "     若需人工审核，批准后再重试 load_dataset。\n"
        "  3) Token 需为 read 及以上（你终端里已是 read 即可）。\n"
        "可选：设置环境变量 HF_TOKEN 指向同一 token，避免非交互环境读不到 ~/.cache。\n"
        f"原始错误: {type(exc).__name__}: {exc}"
    )


def _wrap_progress(
    iterable: Iterable[T],
    *,
    total: Optional[int],
    desc: str,
    unit: str,
    enable: bool,
) -> Iterable[T]:
    if not enable:
        return iterable
    if tqdm is None:
        print(
            "[imagenet_task] 未安装 tqdm，无进度条。可执行: pip install tqdm",
            file=sys.stderr,
            flush=True,
        )
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.3,
        disable=False,  # 非 TTY（IDE/重定向）下也显示，避免「完全不动」
        miniters=1,
    )


@dataclass
class ImagenetTask:
    split: str = "validation"
    max_samples: int = 2000
    batch_size: int = 64
    seed: int = 2026
    streaming: bool = True
    weights_name: str = "IMAGENET1K_V2"
    shuffle_buffer_size: int = 10000
    # If True: skip HF; random N(0,1) RGB in official 224 layout (no extra transforms).
    # For pipeline / fault-injection tests only (accuracy meaningless vs ImageNet).
    synthetic: bool = False
    # HuggingFace datasets 仓库 id（默认 ILSVRC/imagenet-1k）。
    hf_dataset_id: str = HF_IMAGENET_DATASET_ID
    # 若设置：由 download_imagenet_val.py save_to_disk 的目录；完全不走路由 Hub 流式，避免断线与退出崩溃。
    local_dataset_dir: Optional[str] = None
    # tqdm 进度条；若无 tqdm 则退化为无进度输出。
    show_progress: bool = True

    def run(self, runner) -> Dict[str, Any]:
        if not hasattr(runner, "transforms"):
            raise TypeError("runner must be VisionRunner (needs .transforms)")

        community = COMMUNITY_TOP1_TOP5.get(
            self.weights_name, COMMUNITY_TOP1_TOP5["IMAGENET1K_V2"]
        )
        comm_top1, comm_top5 = community

        device = getattr(runner, "device", "cuda")
        dtype = getattr(runner, "dtype", torch.float32)

        top1_correct = 0
        top5_correct = 0
        total = 0
        per_example: List[Dict[str, Any]] = []

        batch_imgs: List[torch.Tensor] = []
        batch_labels: List[int] = []
        global_idx = 0

        def flush_batch():
            nonlocal batch_imgs, batch_labels, top1_correct, top5_correct, total, global_idx
            if not batch_imgs:
                return
            images = torch.stack(batch_imgs, dim=0).to(device=device, dtype=dtype)
            labels_t = torch.tensor(batch_labels, device=device, dtype=torch.long)
            logits = runner.forward(images)
            pred = logits.argmax(dim=-1)
            _, idx5 = logits.topk(5, dim=-1)
            for i in range(images.shape[0]):
                y = int(labels_t[i].item())
                p1 = int(pred[i].item())
                in5 = int((idx5[i] == y).any().item())
                top1_correct += int(p1 == y)
                top5_correct += in5
                per_example.append(
                    {
                        "index": global_idx,
                        "label": y,
                        "pred1": p1,
                        "in_top5": in5,
                        "correct_top1": int(p1 == y),
                    }
                )
                global_idx += 1
            total += images.shape[0]
            batch_imgs = []
            batch_labels = []

        if self.synthetic:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(self.seed))
            n = int(self.max_samples)
            bs = int(self.batch_size)
            n_batch = (n + bs - 1) // bs if bs > 0 else 0
            batch_starts = range(0, n, bs)
            batch_starts = _wrap_progress(
                batch_starts,
                total=n_batch,
                desc="synthetic",
                unit="batch",
                enable=self.show_progress,
            )
            for start in batch_starts:
                bsz = min(bs, n - start)
                # Standardized-like input (same spatial size as val); not identical to HF val pipeline.
                noise = torch.randn(bsz, 3, 224, 224, generator=g, dtype=torch.float32)
                labels = torch.randint(0, 1000, (bsz,), generator=g)
                for i in range(bsz):
                    batch_imgs.append(noise[i])
                    batch_labels.append(int(labels[i].item()))
                flush_batch()
        elif self.local_dataset_dir:
            root = Path(self.local_dataset_dir).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"local_dataset_dir 不存在或不是目录: {root}")
            ds = load_from_disk(str(root))
            ds = ds.shuffle(seed=self.seed)
            if self.max_samples > 0:
                ds = ds.select(range(min(int(self.max_samples), len(ds))))
            sample_total = len(ds)
            if self.show_progress:
                print(
                    f"[imagenet_task] 本地数据集 {root}（{sample_total} 条），无 Hub 流式。",
                    file=sys.stderr,
                    flush=True,
                )
            pbar = _wrap_progress(
                ds,
                total=sample_total,
                desc="imagenet:local",
                unit="img",
                enable=self.show_progress,
            )
            try:
                for ex in pbar:
                    img = ex["image"]
                    if hasattr(img, "mode") and img.mode != "RGB":
                        img = img.convert("RGB")
                    label = int(ex["label"])
                    t = runner.transforms(img)
                    if not isinstance(t, torch.Tensor):
                        raise TypeError("weights.transforms() must return a Tensor for batching")
                    batch_imgs.append(t)
                    batch_labels.append(label)
                    if len(batch_imgs) >= self.batch_size:
                        flush_batch()
                flush_batch()
            finally:
                close_fn = getattr(pbar, "close", None)
                if callable(close_fn):
                    close_fn()
                del pbar
                del ds
                gc.collect()
        else:
            try:
                ds = load_dataset(
                    self.hf_dataset_id,
                    split=self.split,
                    streaming=self.streaming,
                )
            except Exception as e:
                raise RuntimeError(_hf_auth_hint(e)) from e

            if self.streaming:
                # 流式 shuffle 会先攒满 buffer 再吐样本；buffer 过大时首条要等很久（尤其网络慢）。
                buf = min(int(self.shuffle_buffer_size), max(2, int(self.max_samples)), 2048)
                ds = ds.shuffle(seed=self.seed, buffer_size=buf)
                ds = ds.take(int(self.max_samples))
                sample_total = int(self.max_samples)
            else:
                ds = ds.shuffle(seed=self.seed)
                if self.max_samples > 0:
                    ds = ds.select(range(min(self.max_samples, len(ds))))
                sample_total = len(ds)

            if self.show_progress:
                print(
                    "[imagenet_task] 正在拉取首条样本（Hub Parquet/图片，依赖网络；慢时可能卡住数分钟）。"
                    " 大陆可试: export HF_ENDPOINT=https://hf-mirror.com",
                    file=sys.stderr,
                    flush=True,
                )
            pbar = _wrap_progress(
                ds,
                total=sample_total,
                desc=f"imagenet:{self.split}",
                unit="img",
                enable=self.show_progress,
            )
            try:
                for ex in pbar:
                    img = ex["image"]
                    if hasattr(img, "mode") and img.mode != "RGB":
                        img = img.convert("RGB")
                    label = int(ex["label"])
                    t = runner.transforms(img)
                    if not isinstance(t, torch.Tensor):
                        raise TypeError("weights.transforms() must return a Tensor for batching")
                    batch_imgs.append(t)
                    batch_labels.append(label)
                    if len(batch_imgs) >= self.batch_size:
                        flush_batch()
                flush_batch()
            finally:
                # 尽早释放流式迭代器，减轻解释器退出时 HF/pyarrow 后台线程与 GIL 冲突（偶发 core dump）。
                close_fn = getattr(pbar, "close", None)
                if callable(close_fn):
                    close_fn()
                del pbar
                gc.collect()

        top1_acc = float(top1_correct) / float(total) if total else 0.0
        top5_acc = float(top5_correct) / float(total) if total else 0.0

        return {
            "benchmark": "imagenet-1k",
            "mode": "classification",
            "summary": {
                "hf_dataset_id": (
                    None
                    if self.synthetic or self.local_dataset_dir
                    else self.hf_dataset_id
                ),
                "local_dataset_dir": (
                    str(Path(self.local_dataset_dir).expanduser().resolve())
                    if self.local_dataset_dir
                    else None
                ),
                "total": total,
                "top1_correct": top1_correct,
                "top5_correct": top5_correct,
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "community_top1": comm_top1,
                "community_top5": comm_top5,
                "delta_top1": top1_acc - comm_top1,
                "delta_top5": top5_acc - comm_top5,
                "weights": self.weights_name,
                "synthetic": bool(self.synthetic),
            },
            "per_example": per_example,
        }
