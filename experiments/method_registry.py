"""Unified method registry for RL policy and baselines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    latex_name: str
    is_baseline: bool


METHOD_SPECS: dict[str, MethodSpec] = {
    "ours": MethodSpec(
        method_id="ours",
        display_name="RL (Ours)",
        latex_name="本文方法",
        is_baseline=False,
    ),
    "fifo": MethodSpec(
        method_id="fifo",
        display_name="FIFO",
        latex_name="FIFO",
        is_baseline=True,
    ),
    "gas": MethodSpec(
        method_id="gas",
        display_name="Gas Priority",
        latex_name="Gas 优先",
        is_baseline=True,
    ),
    "heuristic": MethodSpec(
        method_id="heuristic",
        display_name="Heuristic Risk-Aware",
        latex_name="Heuristic",
        is_baseline=True,
    ),
    "fee_risk_linear": MethodSpec(
        method_id="fee_risk_linear",
        display_name="Fee-Risk Linear",
        latex_name="Fee-Risk Linear",
        is_baseline=True,
    ),
    "fair_fee": MethodSpec(
        method_id="fair_fee",
        display_name="Fair-Fee Greedy",
        latex_name="Fair-Fee Greedy",
        is_baseline=True,
    ),
    "center_aware": MethodSpec(
        method_id="center_aware",
        display_name="Dynamic Tri-Objective Greedy",
        latex_name="Dynamic Tri-Objective Greedy",
        is_baseline=True,
    ),
}

LEGACY_METHOD_ALIASES: dict[str, str] = {
    "RL (Ours)": "ours",
    "RL": "ours",
    "FIFO": "fifo",
    "GAS": "gas",
    "HEURISTIC": "heuristic",
    "FeeRiskLinear": "fee_risk_linear",
    "FairFee": "fair_fee",
    "CenterAwareGreedy": "center_aware",
}

DEFAULT_BASELINE_METHOD_IDS = ["fifo", "gas", "heuristic", "fee_risk_linear", "fair_fee"]
STRONG_BASELINE_METHOD_IDS = ["center_aware"]
MAIN_METHOD_ORDER = [*DEFAULT_BASELINE_METHOD_IDS, *STRONG_BASELINE_METHOD_IDS, "ours"]
BASELINE_METHOD_IDS = [*DEFAULT_BASELINE_METHOD_IDS, *STRONG_BASELINE_METHOD_IDS]


def get_baseline_method_ids(include_strong_baseline: bool = False) -> list[str]:
    baselines = list(DEFAULT_BASELINE_METHOD_IDS)
    if include_strong_baseline:
        baselines.extend(STRONG_BASELINE_METHOD_IDS)
    return baselines


def normalize_method_id(method_id: str) -> str:
    if method_id in METHOD_SPECS:
        return method_id
    if method_id in LEGACY_METHOD_ALIASES:
        return LEGACY_METHOD_ALIASES[method_id]
    raise KeyError(f"Unknown method id: {method_id}")


def display_name(method_id: str) -> str:
    return METHOD_SPECS[normalize_method_id(method_id)].display_name


def latex_name(method_id: str) -> str:
    return METHOD_SPECS[normalize_method_id(method_id)].latex_name
