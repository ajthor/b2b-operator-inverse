"""
Utility helpers for formatting evaluation reports.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional


def _is_valid_number(value: Optional[float]) -> bool:
    return value is not None and not math.isnan(value) and not math.isinf(value)


def format_scientific(value: Optional[float], precision: int = 3) -> str:
    """Return a consistent scientific-notation string or an em dash for blanks."""
    if not _is_valid_number(value):
        return "—"
    return f"{value:.{precision}e}"


def format_mean_std(
    mean: Optional[float],
    std: Optional[float],
    precision: int = 2,
) -> str:
    """
    Format mean ± std in the requested ``X.XX ± Y.YY eZZ`` style.

    The exponent is shared so that the expression is easy to transcribe.
    """
    if not _is_valid_number(mean) and not _is_valid_number(std):
        return "—"

    reference = 0.0
    for candidate in (mean, std):
        if _is_valid_number(candidate) and candidate != 0.0:
            reference = abs(candidate)
            break

    exponent = 0
    if reference != 0.0:
        exponent = int(math.floor(math.log10(reference)))

    scale = 10 ** exponent

    def _scaled(value: Optional[float]) -> float:
        if not _is_valid_number(value):
            return 0.0
        return value / scale if scale != 0 else value

    mean_scaled = _scaled(mean)
    std_scaled = _scaled(std)
    exp_str = f"{exponent:+02d}"
    return f"{mean_scaled:.{precision}f} ± {std_scaled:.{precision}f} e^{exp_str}"


def build_summary_rows(
    component_stats: Dict[str, Dict[str, float]],
    precision: int = 3,
) -> Iterable[Iterable[str]]:
    """
    Yield table rows of the form
    [component, mean±std, median, min, max] using consistent formatting.
    """
    for name in sorted(component_stats.keys()):
        stats = component_stats[name]
        yield [
            name,
            format_mean_std(stats.get("mean"), stats.get("std")),
            format_scientific(stats.get("median"), precision),
            format_scientific(stats.get("min"), precision),
            format_scientific(stats.get("max"), precision),
        ]
