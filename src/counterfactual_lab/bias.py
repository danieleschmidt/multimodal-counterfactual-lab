"""
BiasEvaluator: measure prediction consistency across counterfactual pairs.

Consistency ≈ fairness proxy.

If a model changes its prediction when only a protected attribute in the text
changes (and nothing semantically meaningful was altered), that inconsistency
is evidence of attribute-sensitive bias.

Works with any callable model: (text) → label or (text) → dict with "label"/"score".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from counterfactual_lab.generator import CounterfactualPair, GenerationResult

logger = logging.getLogger(__name__)

# Type alias for a text classifier
TextModel = Callable[[str], Any]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PairConsistency:
    """Consistency measurement for a single counterfactual pair."""

    pair: CounterfactualPair
    original_prediction: Any
    counterfactual_prediction: Any
    consistent: bool          # True if predictions are the same
    score_delta: Optional[float] = None  # |score_orig - score_cf| if scores available


@dataclass
class BiasReport:
    """Aggregated bias measurements over a GenerationResult."""

    original_text: str
    n_pairs: int
    n_consistent: int
    consistency_rate: float           # n_consistent / n_pairs
    attribute_rates: Dict[str, float] = field(default_factory=dict)
    strategy_rates: Dict[str, float] = field(default_factory=dict)
    pair_details: List[PairConsistency] = field(default_factory=list)
    mean_score_delta: Optional[float] = None

    @property
    def bias_detected(self) -> bool:
        """True if consistency rate is below 1.0 (any prediction flip found)."""
        return self.consistency_rate < 1.0

    def summary(self) -> str:
        lines = [
            f"BiasReport for: {self.original_text[:80]!r}",
            f"  Pairs evaluated : {self.n_pairs}",
            f"  Consistent      : {self.n_consistent}/{self.n_pairs} ({self.consistency_rate:.1%})",
            f"  Bias detected   : {self.bias_detected}",
        ]
        for attr, rate in self.attribute_rates.items():
            lines.append(f"  Attribute [{attr}] consistency: {rate:.1%}")
        for strat, rate in self.strategy_rates.items():
            lines.append(f"  Strategy  [{strat}] consistency: {rate:.1%}")
        if self.mean_score_delta is not None:
            lines.append(f"  Mean |score delta|: {self.mean_score_delta:.4f}")
        return "\n".join(lines)


@dataclass
class DatasetBiasReport:
    """Aggregated report over a full dataset (list of GenerationResults)."""

    n_inputs: int
    n_pairs_total: int
    overall_consistency_rate: float
    attribute_rates: Dict[str, float] = field(default_factory=dict)
    strategy_rates: Dict[str, float] = field(default_factory=dict)
    per_input_reports: List[BiasReport] = field(default_factory=list)
    mean_score_delta: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=== Dataset Bias Report ===",
            f"  Inputs evaluated        : {self.n_inputs}",
            f"  Total CF pairs          : {self.n_pairs_total}",
            f"  Overall consistency     : {self.overall_consistency_rate:.1%}",
        ]
        for attr, rate in self.attribute_rates.items():
            lines.append(f"  Attribute [{attr}] consistency : {rate:.1%}")
        for strat, rate in self.strategy_rates.items():
            lines.append(f"  Strategy  [{strat}] consistency : {rate:.1%}")
        if self.mean_score_delta is not None:
            lines.append(f"  Mean |score delta|          : {self.mean_score_delta:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_prediction(raw: Any) -> Tuple[str, Optional[float]]:
    """
    Normalise a model prediction to (label, score).

    Accepts:
    - str  → (str, None)
    - dict with "label" and optional "score" → (label, score)
    - list of dicts [{"label": ..., "score": ...}] (HF pipeline output) → best
    """
    if isinstance(raw, str):
        return raw, None
    if isinstance(raw, dict):
        label = str(raw.get("label", raw))
        score = raw.get("score", raw.get("confidence", None))
        return label, float(score) if score is not None else None
    if isinstance(raw, list) and raw:
        # HF pipeline: list of {"label":..., "score":...}
        best = max(raw, key=lambda x: x.get("score", 0))
        return str(best.get("label", "")), float(best.get("score", 0))
    return str(raw), None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class BiasEvaluator:
    """
    Evaluate prediction consistency (fairness proxy) for counterfactual pairs.

    Parameters
    ----------
    model : callable
        Text classifier: (str) → prediction (str | dict | list).
    """

    def __init__(self, model: TextModel):
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, result: GenerationResult) -> BiasReport:
        """Evaluate bias for a single GenerationResult."""
        details: List[PairConsistency] = []

        orig_label, orig_score = _normalise_prediction(self.model(result.original_text))

        for pair in result.pairs:
            cf_label, cf_score = _normalise_prediction(self.model(pair.counterfactual_text))
            consistent = orig_label == cf_label

            score_delta: Optional[float] = None
            if orig_score is not None and cf_score is not None:
                score_delta = abs(orig_score - cf_score)

            details.append(
                PairConsistency(
                    pair=pair,
                    original_prediction=orig_label,
                    counterfactual_prediction=cf_label,
                    consistent=consistent,
                    score_delta=score_delta,
                )
            )

        return self._aggregate(result.original_text, details)

    def evaluate_batch(self, results: List[GenerationResult]) -> DatasetBiasReport:
        """Evaluate bias over a list of GenerationResults."""
        per_input: List[BiasReport] = []
        for res in results:
            per_input.append(self.evaluate(res))

        n_pairs_total = sum(r.n_pairs for r in per_input)
        n_consistent_total = sum(r.n_consistent for r in per_input)
        overall_rate = n_consistent_total / n_pairs_total if n_pairs_total else 1.0

        # Aggregate by attribute
        attr_counts: Dict[str, List[int]] = {}
        strat_counts: Dict[str, List[int]] = {}
        all_deltas: List[float] = []

        for r in per_input:
            for attr, rate in r.attribute_rates.items():
                attr_counts.setdefault(attr, []).append(rate)
            for strat, rate in r.strategy_rates.items():
                strat_counts.setdefault(strat, []).append(rate)
            for pc in r.pair_details:
                if pc.score_delta is not None:
                    all_deltas.append(pc.score_delta)

        attr_rates = {k: sum(v) / len(v) for k, v in attr_counts.items()}
        strat_rates = {k: sum(v) / len(v) for k, v in strat_counts.items()}
        mean_delta = sum(all_deltas) / len(all_deltas) if all_deltas else None

        return DatasetBiasReport(
            n_inputs=len(results),
            n_pairs_total=n_pairs_total,
            overall_consistency_rate=overall_rate,
            attribute_rates=attr_rates,
            strategy_rates=strat_rates,
            per_input_reports=per_input,
            mean_score_delta=mean_delta,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _aggregate(
        self, original_text: str, details: List[PairConsistency]
    ) -> BiasReport:
        n = len(details)
        if n == 0:
            return BiasReport(
                original_text=original_text,
                n_pairs=0,
                n_consistent=0,
                consistency_rate=1.0,
            )

        n_consistent = sum(1 for d in details if d.consistent)

        # Per-attribute rates
        attr_groups: Dict[str, List[PairConsistency]] = {}
        strat_groups: Dict[str, List[PairConsistency]] = {}
        for d in details:
            attr = d.pair.perturbation.attribute
            strat = d.pair.perturbation.strategy
            attr_groups.setdefault(attr, []).append(d)
            strat_groups.setdefault(strat, []).append(d)

        attr_rates = {
            attr: sum(1 for d in group if d.consistent) / len(group)
            for attr, group in attr_groups.items()
        }
        strat_rates = {
            strat: sum(1 for d in group if d.consistent) / len(group)
            for strat, group in strat_groups.items()
        }

        deltas = [d.score_delta for d in details if d.score_delta is not None]
        mean_delta = sum(deltas) / len(deltas) if deltas else None

        return BiasReport(
            original_text=original_text,
            n_pairs=n,
            n_consistent=n_consistent,
            consistency_rate=n_consistent / n,
            attribute_rates=attr_rates,
            strategy_rates=strat_rates,
            pair_details=details,
            mean_score_delta=mean_delta,
        )
