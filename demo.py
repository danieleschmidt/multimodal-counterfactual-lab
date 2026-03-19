"""
Counterfactual Fairness Demo
============================

Demonstrates:
1. Counterfactual text pair generation via lexical substitution & semantic paraphrase.
2. Fairness evaluation with two classifiers:
   - A *fair* keyword-based sentiment classifier (consistent across genders).
   - A *biased* classifier that labels male-gendered text as POSITIVE and
     female-gendered text as NEGATIVE regardless of actual content.
3. Dataset-level bias reports comparing both classifiers.

No GPU, no external models — fully self-contained.
"""

from __future__ import annotations

import textwrap

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
    tokens = [w.strip(".,!?;:—-\"'") for w in text.lower().split()]
    pos = sum(1 for w in tokens if w in _POSITIVE)
    neg = sum(1 for w in tokens if w in _NEGATIVE)
    total = pos + neg
    if total == 0:
        return {"label": "NEUTRAL", "score": 0.5}
    score = pos / total
    return {"label": "POSITIVE" if score >= 0.5 else "NEGATIVE", "score": round(score, 3)}


# ---------------------------------------------------------------------------
# Classifier B: deliberately biased (male → POSITIVE, female → NEGATIVE)
# ---------------------------------------------------------------------------

_MALE_TERMS   = {"he", "him", "his", "himself", "man", "men", "boy", "boys",
                 "gentleman", "husband", "father", "son", "brother", "mr", "sir"}
_FEMALE_TERMS = {"she", "her", "hers", "herself", "woman", "women", "girl", "girls",
                 "lady", "wife", "mother", "daughter", "sister", "ms", "mrs", "miss"}
_NEUTRAL_TERMS = {"they", "them", "their", "themselves", "person", "people", "child",
                  "children", "individual", "sibling", "parent", "spouse", "partner", "mx"}


def biased_classifier(text: str) -> dict:
    """
    Simulates a gender-biased model:
    - Texts with male-gendered terms → POSITIVE (0.85)
    - Texts with female-gendered terms → NEGATIVE (0.15)
    - Neutral/mixed → keyword-based fair score

    In real systems this bias is subtler — learned from skewed training data.
    This extreme version makes prediction flips clearly visible.
    """
    tokens = {w.strip(".,!?;:—-\"'") for w in text.lower().split()}
    is_male   = bool(tokens & _MALE_TERMS)
    is_female = bool(tokens & _FEMALE_TERMS)
    is_neutral = bool(tokens & _NEUTRAL_TERMS)

    if is_male and not is_female and not is_neutral:
        return {"label": "POSITIVE", "score": 0.85}
    if is_female and not is_male and not is_neutral:
        return {"label": "NEGATIVE", "score": 0.15}
    # Mixed or neutral: fall back to content
    return fair_classifier(text)


# ---------------------------------------------------------------------------
# Demo inputs
# ---------------------------------------------------------------------------

DEMO_TEXTS = [
    # Clearly male-coded → biased classifier will call POSITIVE regardless of quality
    "He is a terrible and incompetent manager who never listens.",
    "The man was frequently late and failed to complete his assignments.",
    "His work was described as poor and below expectations.",
    # Clearly female-coded → biased classifier will call NEGATIVE regardless of merit
    "She is an outstanding engineer who solved the problem brilliantly.",
    "The woman proved herself an excellent leader, skilled and capable.",
    "Her research was brilliant and made an impressive contribution.",
    # Professional / neutral contexts
    "The doctor examined his patient carefully and provided an excellent diagnosis.",
    "The manager said she was excellent at her job and deserved a promotion.",
    "They described her as competent but noted he had failed to meet targets.",
    "A qualified candidate was selected based on their skills and experience.",
]


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _print_section(title: str, results, dataset_report, orig_fn) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"{sep}\n")

    for i, (res, report) in enumerate(zip(results, dataset_report.per_input_reports)):
        orig = orig_fn(res.original_text)
        print(f"[{i+1:02d}] {res.original_text[:72]}")
        print(f"     Original: {orig['label']} (score={orig['score']:.2f})  |  {res.n_pairs} CF pairs")

        if not res.pairs:
            print("     (no counterfactuals generated)\n")
            continue

        for pc in report.pair_details:
            marker = "✓" if pc.consistent else "✗ FLIP"
            delta = f"  Δ={pc.score_delta:.3f}" if pc.score_delta is not None else ""
            strat = pc.pair.perturbation.strategy[:4]
            direction = pc.pair.perturbation.direction
            print(
                f"     [{marker}] {strat} | {direction:<22} "
                f"→ {pc.counterfactual_prediction:<10}{delta}"
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
        verdict = "⚠️   Moderate inconsistency — worth investigating."
    else:
        verdict = "🚨  High inconsistency — strong evidence of gender-driven bias."
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

    # Fair classifier
    fair_eval    = BiasEvaluator(model=fair_classifier)
    fair_results = gen.generate_batch(inputs)
    fair_report  = fair_eval.evaluate_batch(fair_results)
    _print_section("Classifier A — Fair (keyword sentiment, gender-blind)",
                   fair_results, fair_report, fair_classifier)

    # Biased classifier
    biased_eval    = BiasEvaluator(model=biased_classifier)
    biased_results = gen.generate_batch(inputs)
    biased_report  = biased_eval.evaluate_batch(biased_results)
    _print_section("Classifier B — Biased (male→POSITIVE, female→NEGATIVE)",
                   biased_results, biased_report, biased_classifier)

    # Comparison
    print("=" * 70)
    print("  Comparison Summary")
    print("=" * 70)
    f_rate = fair_report.overall_consistency_rate
    b_rate = biased_report.overall_consistency_rate
    gap    = f_rate - b_rate
    print(f"  Fair    classifier consistency: {f_rate:.1%}")
    print(f"  Biased  classifier consistency: {b_rate:.1%}")
    print(f"  Consistency gap (fair − biased): {gap:+.1%}")
    print()
    print("  Key insight:")
    print("  The biased classifier flips labels when male ↔ female terms are")
    print("  swapped — surfacing gender-correlated bias via counterfactual testing.")
    print("  Counterfactual consistency is a fairness proxy that requires no")
    print("  demographic ground-truth labels — only paired perturbations.")
    print()


if __name__ == "__main__":
    run_demo()
