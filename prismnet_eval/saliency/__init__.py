"""Saliency sanity checks for PrismNet models.

This module implements model randomization tests to verify that saliency maps
are model-dependent and not artifacts of the gradient computation method.
"""

from .randomization import (
    full_randomization_test,
    cascading_randomization_test,
    compute_similarity_metrics,
)
from .visualization import (
    plot_saliency_comparison,
    plot_similarity_distributions,
    plot_cascading_results,
    plot_summary_report,
)

__all__ = [
    "full_randomization_test",
    "cascading_randomization_test",
    "compute_similarity_metrics",
    "plot_saliency_comparison",
    "plot_similarity_distributions",
    "plot_cascading_results",
    "plot_summary_report",
]
