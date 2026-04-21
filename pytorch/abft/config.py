from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AbftConfig:
    enable: bool = False
    enable_linear: bool = True
    enable_matmul: bool = True
    enable_bmm: bool = True
    phase: str = "all"  # all | prefill | decode
    sample_rate: float = 1.0
    tol_abs: float = 1e-2
    tol_rel: float = 1e-2
    seed: int = 2026
    force_fp32_check: bool = True
    detect_only: bool = True

    # fault injection (optional)
    inject_fault: bool = False
    inject_phase: str = "all"  # all | prefill | decode
    inject_probability: float = 0.0
    inject_magnitude: float = 1.0

    def phase_enabled(self, current_phase: str) -> bool:
        if self.phase == "all":
            return True
        return self.phase == current_phase

    def inject_phase_enabled(self, current_phase: str) -> bool:
        if self.inject_phase == "all":
            return True
        return self.inject_phase == current_phase
