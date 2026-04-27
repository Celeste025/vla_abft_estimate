import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_PROVIDERS = [
    "triton",
    "cublas",
    "triton_abft_kernel",
    "triton_abft_full",
    "triton_abft_naive_kernel",
    "triton_abft_naive_full",
]

COMPONENT_COMPARE_PROVIDERS = [
    "triton",
    "triton_abft_sum_a_only_kernel",
    "triton_abft_sum_b_only_kernel",
    "triton_abft_sum_c_only_kernel",
    "triton_abft_kernel",
]

# CSV 里名称是 triton_abft_partial_reduce_fast_torch（无 partial_reducec）
PARTIAL_REDUCE_COMPARE_PROVIDERS = [
    "triton",
    "triton_abft_kernel",
    "triton_abft_partial_reduce_fast_torch",
    "triton_abft_partial_reduce_abft_like",
]

ABLATION_COMPARE_PROVIDERS = [
    "triton",
    "triton_abft_ablate_no_sum_store0",
    "triton_abft_full",
    "triton_abft_ablate_sum_no_partial_store",
]

TWO_STAGE_COMPARE_PROVIDERS = [
    "triton",
    "triton_abft_kernel",
    "triton_abft_full",
    "triton_abft_two_stage_kernel",
    "triton_abft_two_stage_full",
]

ISOLATION_COMPARE_PROVIDERS = [
    "triton_iso_mem_skeleton",
    "triton_iso_mm_only",
    "triton_iso_ab_reduce_only",
]


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            provider = row["provider"].strip()
            if not provider:
                continue
            try:
                m = int(row["M"])
                n = int(row["N"])
                k = int(row["K"])
                tflops = float(row["TFLOPS"])
            except (ValueError, TypeError):
                continue
            rows.append({"provider": provider, "M": m, "N": n, "K": k, "TFLOPS": tflops})
    return rows


def plot_tflops_vs_dim(rows, providers, output_path, title=None, y_max=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for provider in providers:
        points = sorted(
            (r for r in rows if r["provider"] == provider),
            key=lambda x: x["M"],
        )
        if not points:
            print(f"warning: provider '{provider}' not found in csv, skip.")
            continue
        xs = [p["M"] for p in points]
        ys = [p["TFLOPS"] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=provider)

    ax.set_xlabel("Matrix Dimension (M=N=K)")
    ax.set_ylabel("TFLOPS")
    ax.set_title(title if title else "TFLOPS vs Matrix Dimension")
    if y_max is not None:
        ax.set_ylim(0, y_max)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"plot_saved,{output_path}")


def _ms_from_row(row):
    # In benchmark script: TFLOPS = 2*M*N*K*1e-12 / (ms*1e-3)
    return 2.0 * row["M"] * row["N"] * row["K"] * 1e-9 / row["TFLOPS"]


def plot_time_vs_dim(rows, providers, output_path, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for provider in providers:
        points = sorted(
            (r for r in rows if r["provider"] == provider and r["TFLOPS"] > 0),
            key=lambda x: x["M"],
        )
        if not points:
            print(f"warning: provider '{provider}' not found in csv, skip.")
            continue
        xs = [p["M"] for p in points]
        ys = [_ms_from_row(p) for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=provider)

    ax.set_xlabel("Matrix Dimension (M=N=K)")
    ax.set_ylabel("Time t (ms)")
    ax.set_title(title if title else "Time vs Matrix Dimension")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"plot_saved,{output_path}")


def parse_providers(provider_arg):
    if not provider_arg:
        return DEFAULT_PROVIDERS
    return [p.strip() for p in provider_arg.split(",") if p.strip()]


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark TFLOPS curves from CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="benchmark_results.csv",
        help="Benchmark CSV path.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default=",".join(DEFAULT_PROVIDERS),
        help="Comma-separated provider list to compare.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/tflops_compare_main.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / csv_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    providers = parse_providers(args.providers)
    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No valid rows loaded from csv: {csv_path}")
    plot_tflops_vs_dim(rows, providers, output_path)

    component_output = Path(__file__).resolve().parent / "plots/tflops_compare_abft_components.png"
    plot_tflops_vs_dim(rows, COMPONENT_COMPARE_PROVIDERS, component_output)

    partial_reduce_output = Path(__file__).resolve().parent / "plots/tflops_compare_partial_reduce.png"
    plot_tflops_vs_dim(
        rows,
        PARTIAL_REDUCE_COMPARE_PROVIDERS,
        partial_reduce_output,
        title="TFLOPS vs Matrix Dimension (matmul vs partial-reduce baselines)",
    )

    ablation_output = Path(__file__).resolve().parent / "plots/tflops_compare_ablation.png"
    plot_tflops_vs_dim(
        rows,
        ABLATION_COMPARE_PROVIDERS,
        ablation_output,
        title="TFLOPS vs Matrix Dimension (ablation comparison)",
    )

    two_stage_output = Path(__file__).resolve().parent / "plots/tflops_compare_two_stage.png"
    plot_tflops_vs_dim(
        rows,
        TWO_STAGE_COMPARE_PROVIDERS,
        two_stage_output,
        title="TFLOPS vs Matrix Dimension (two-stage comparison)",
    )

    isolation_output = Path(__file__).resolve().parent / "plots/tflops_compare_isolation_three_curves.png"
    plot_time_vs_dim(
        rows,
        ISOLATION_COMPARE_PROVIDERS,
        isolation_output,
        title="Time t (ms) vs Matrix Dimension (compute-isolation kernels)",
    )


if __name__ == "__main__":
    main()
