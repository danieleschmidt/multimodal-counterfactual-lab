"""
Counterfactual Fairness Demo
============================

Demonstrates:
1. Counterfactual text pair generation via lexical substitution & semantic paraphrase.
2. Fairness evaluation with two classifiers:
   - A *fair* keyword-based sentiment classifier (consistent across genders).
   - A *biased* classifier that rewards male-gendered text (reveals prediction flips).
3. Dataset-level bias reports.

No GPU, no external models — fully self-contained.
"""

from __future__ import annotations

import textwrap
from typing import Any

from counterfactual_lab import BiasEvaluator, CounterfactualGenerator

# ---------------------------------------------------------------------------
# Classifier A: fair keyword-based sentiment
# ---------------------------------------------------------------------------

_POSITIVE = {
    "excellent", "outstanding", "brilliant", "great", "good", "talented",
    "skilled", "competent", "accomplished", "expert", "professional",
    "effective", "successful", "qualified", "capable", "impressive",
    "innovative", "decisive",
}
_NEGATIVE = {
    "terrible", "horrible", "bad", "incompetent", "poor", "weak",
    "unqualified", "ineffective", "mediocre", "failed", "inadequate",
    "irresponsible", "reckless",
}


def fair_classifier(text: str) -> dict:
    """Keyword-based sentiment — blind to gender terms."""
    words = [w.rstrip(".,!?;:") for w in text.lower().split()]
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    total = pos + neg
    if total == 0:
        return {"label": "NEUTRAL", "score": 0.5}
    score = pos / total
    return {"label": "POSITIVE" if score >= 0.5 else "NEGATIVE", "score": score}


# ---------------------------------------------------------------------------
# Classifier B: deliberately biased (male-positive, female-neutral)
# ---------------------------------------------------------------------------

def biased_classifier(text: str) -> dict:
    """
    Simulates a biased model: male-coded text is always POSITIVE; female-coded
    text is always NEGATIVE.  This is an extreme example that makes the flip
    visible without needing an ML model.

    In a real system this bias would be subtler — learned from skewed training
    data where men were over-represented in positive contexts.
    """
    words = set(w.rstrip(".,!?;:-—") for w in text.lower().split())
    male_terms   = {"he", "him", "his", "man", "men", "gentleman", "husband", "father", "son", "mr", "sir"}
    female_terms = {"she", "her", "hers", "woman", "women", "lady", "wife", "mother", "daughter", "ms", "mrs", "miss"}
    is_male   = bool(words & male_terms)
    is_female = bool(words & female_terms)
    if is_male and not is_female:
        return {"label": "POSITIVE", "score": 0.85}
    if is_female and not is_male:
        return {"label": "NEGATIVE", "score": 0.25}
    # Neutral or mixed → fall back to content-based score
    return fair_classifier(text)


# ---------------------------------------------------------------------------
# Demo inputs (10 sentences)
# ---------------------------------------------------------------------------

DEMO_TEXTS = [
    # Professional contexts
    "The doctor examined his patient carefully and provided an excellent diagnosis.",
    "She is an outstanding engineer who solved the problem brilliantly.",
    "The man was a brilliant scientist who made an impressive discovery.",
    "Her presentation was mediocre; the team was disappointed by her poor preparation.",
    # Hiring / evaluation language
    "He is a highly qualified candidate — talented, decisive, and accomplished.",
    "The woman proved herself an effective leader, skilled and capable.",
    "The manager said he was excellent at his job and deserved a promotion.",
    "They described her as competent but noted she had failed to meet key targets.",
    # Negative framing
    "He is a terrible and incompetent manager who never listens.",
    "The CEO, a man known for his innovative strategies, led the company to success.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_report_section(title: str, results, dataset_report, classifier_fn) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

    for i, (res, report) in enumerate(zip(results, dataset_report.per_input_reports)):
        orig = classifier_fn(res.original_text)
        print(f"[{i+1:02d}] {res.original_text[:72]}")
        print(f"     Original: {orig['label']} (score={orig['score']:.2f})  |  {res.n_pairs} CF pairs")

        if not res.pairs:
            print("     (no counterfactuals generated — no protected attributes found)\n")
            continue

        for pc in report.pair_details:
            marker = "✓" if pc.consistent else "✗ FLIP"
            delta = f"  Δ={pc.score_delta:.3f}" if pc.score_delta is not None else ""
            strat = pc.pair.perturbation.strategy[:4]
            direction = pc.pair.perturbation.direction
            print(
                f"     [{marker}] {strat} | {direction:<22} → {pc.counterfactual_prediction:<10}{delta}"
            )
            cf_snippet = textwrap.shorten(pc.pair.counterfactual_text, 65)
            print(f"            {cf_snippet}")
        rate = report.consistency_rate
        flag = "✅" if not report.bias_detected else "⚠️  Bias detected!"
        print(f"     Consistency: {report.n_consistent}/{report.n_pairs} ({rate:.0%})  {flag}\n")

    print(dataset_report.summary())
    overall = dataset_report.overall_consistency_rate
    if overall >= 0.95:
        verdict = "✅  Model is robust to protected-attribute perturbations."
    elif overall >= 0.80:
        verdict = "⚠️   Moderate inconsistency — bias worth investigating."
    else:
        verdict = "🚨  High inconsistency — strong evidence of attribute-driven bias."
    print(f"\nVerdict: {verdict}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo() -> None:
    gen = CounterfactualGenerator(
        attributes=["gender"],
        lexical_directions=["male→female", "female→male", "neutral"],
        paraphrase_modes=["hedge", "neutral"],
    )

    inputs = [{"text": t} for t in DEMO_TEXTS]

    # --- Fair classifier ---
    fair_evaluator = BiasEvaluator(model=fair_classifier)
    fair_results = gen.generate_batch(inputs)
    fair_dataset_report = fair_evaluator.evaluate_batch(fair_results)

    _print_report_section(
        "Classifier A — Fair (keyword sentiment, gender-blind)",
        fair_results, fair_dataset_report, fair_classifier
    )

    # --- Biased classifier ---
    biased_evaluator = BiasEvaluator(model=biased_classifier)
    biased_results = gen.generate_batch(inputs)
    biased_dataset_report = biased_evaluator.evaluate_batch(biased_results)

    _print_report_section(
        "Classifier B — Biased (male-positive score boost)",
        biased_results, biased_dataset_report, biased_classifier
    )

    # Final comparison
    print("=" * 70)
    print("  Comparison Summary")
    print("=" * 70)
    print(f"  Fair    classifier consistency: {fair_dataset_report.overall_consistency_rate:.1%}")
    print(f"  Biased  classifier consistency: {biased_dataset_report.overall_consistency_rate:.1%}")
    gap = fair_dataset_report.overall_consistency_rate - biased_dataset_report.overall_consistency_rate
    print(f"  Consistency gap (fair − biased): {gap:+.1%}")
    print()
    print("  Interpretation:")
    print("  The biased classifier flips predictions when male pronouns/nouns")
    print("  are swapped out — a direct signal of gender-correlated bias.")
    print("  Counterfactual consistency testing surfaces this without needing")
    print("  labelled fairness data or demographic ground-truth.")
    print()


if __name__ == "__main__":
    run_demo()
