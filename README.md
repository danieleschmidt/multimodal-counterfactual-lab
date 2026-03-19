# multimodal-counterfactual-lab

A toolkit for generating **counterfactual image-text pairs** to support
fairness and robustness research in vision-language and text classification models.

> **Core idea:** if a model changes its prediction when only a protected attribute
> (gender, age, etc.) changes in the input text — and the actual semantics are
> preserved — that inconsistency is evidence of attribute-sensitive bias.
> No demographic ground-truth labels needed: just counterfactual pairs.

---

## Features

| Feature | Description |
|---|---|
| **LexicalSubstitution** | Direct token-level swaps of protected-attribute terms (male↔female, gender-neutral) |
| **SemanticParaphrase** | Minimal meaning-preserving rewrites (hedge absolutes, neutralise gender, voice changes) |
| **CounterfactualGenerator** | Applies both strategies to produce `CounterfactualPair` collections |
| **BiasEvaluator** | Measures prediction consistency across pairs — per-attribute, per-strategy, dataset-level |
| **Text-only mode** | Works without any vision model; image is carried along unchanged |
| **HF-compatible** | `BiasEvaluator` accepts HuggingFace pipeline output out-of-the-box |

---

## Quickstart

```bash
git clone https://github.com/danieleschmidt/multimodal-counterfactual-lab
cd multimodal-counterfactual-lab
pip install -e .
python demo.py
```

### Generate counterfactuals

```python
from counterfactual_lab import CounterfactualGenerator

gen = CounterfactualGenerator(attributes=["gender"])
result = gen.generate("The doctor examined his patient carefully.")

for pair in result.pairs:
    print(pair.perturbation.direction, "→", pair.counterfactual_text)
# male→female  → The doctor examined her patient carefully.
# neutral      → The doctor examined their patient carefully.
# hedge        → The doctor examined his patient carefully.   (if absolute words present)
```

### Measure fairness

```python
from counterfactual_lab import BiasEvaluator
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
evaluator  = BiasEvaluator(model=classifier)

report = evaluator.evaluate(result)
print(report.summary())
# BiasReport for: 'The doctor examined his patient carefully.'
#   Pairs evaluated : 3
#   Consistent      : 3/3 (100%)
#   Bias detected   : False
```

### Dataset-level analysis

```python
inputs  = [{"text": t} for t in my_corpus]
results = gen.generate_batch(inputs)
dataset_report = evaluator.evaluate_batch(results)
print(dataset_report.summary())
```

### CLI

```bash
counterfactual-lab generate --text "She is an excellent engineer." --attributes gender
```

---

## Perturbation strategies

### 1. Lexical Substitution

Direct whole-word replacement of protected-attribute terms.

| Direction | Example |
|---|---|
| `male→female` | "he examined **his** patient" → "she examined **her** patient" |
| `female→male` | "the **woman** presented" → "the **man** presented" |
| `neutral`     | "**he** said **his** view" → "**they** said **their** view" |

Covers: pronouns, nouns, family roles, honorifics.

### 2. Semantic Paraphrase

Minimal rewrites that change surface form without (ideally) changing ground truth:

| Mode | What it does | Example |
|---|---|---|
| `hedge` | Soften absolute sentiment words | "always" → "often", "excellent" → "good" |
| `neutral` | Remove gendered terms (same lexicon, framed as paraphrase) | "mother" → "parent" |
| `active-passive` | Voice transformation | "he saved her" → "she was saved by him" |

---

## Bias measurement

`BiasEvaluator` wraps any text classifier `(str) → label | dict | HF list` and computes:

- **Consistency rate** — fraction of CF pairs where prediction doesn't change
- **Per-attribute rates** — consistency broken down by protected attribute
- **Per-strategy rates** — consistency broken down by perturbation strategy  
- **Score delta** — mean absolute change in confidence score (sensitivity proxy)

A consistency rate < 1.0 indicates attribute-sensitive predictions — a fairness concern.

---

## Demo results

Running `demo.py` compares a fair vs. a deliberately biased classifier:

```
=== Dataset Bias Report ===
  Inputs evaluated        : 10
  Total CF pairs          : 34

Fair  classifier consistency:  100.0%   ✅
Biased classifier consistency:  35.3%   🚨

Consistency gap (fair − biased): +64.7%
```

The biased classifier flips labels on `male→female` swaps even when the
actual text quality is identical — surfaced cleanly by counterfactual consistency.

---

## Extending with real models

```python
from transformers import pipeline
from counterfactual_lab import CounterfactualGenerator, BiasEvaluator

# Any HF pipeline or custom callable works
clf = pipeline("text-classification", model="your-model")
gen = CounterfactualGenerator(attributes=["gender", "age"])

evaluator = BiasEvaluator(model=clf)
results   = gen.generate_batch([{"text": t} for t in texts])
report    = evaluator.evaluate_batch(results)
```

Vision support: pass `image=` to `gen.generate()`; the image is carried through
all text-side counterfactuals unchanged. Image-side perturbations (style transfer,
masking) are a planned extension.

---

## Tests

```bash
pytest tests/ -v
# 61 passed in 0.04s
```

---

## Project structure

```
src/counterfactual_lab/
    __init__.py         # public API
    perturbations.py    # LexicalSubstitution, SemanticParaphrase, Perturbation
    generator.py        # CounterfactualGenerator, CounterfactualPair, GenerationResult
    bias.py             # BiasEvaluator, BiasReport, DatasetBiasReport
    cli.py              # CLI entry point
demo.py                 # end-to-end demo with fair vs. biased classifiers
tests/                  # 61 pytest tests
```

---

## Background

Counterfactual data augmentation for fairness evaluation draws on:

- **Counterfactual Data Augmentation** (Zmigrod et al., 2019) — gender-neutral rewrites
- **WinoBias / WinoGender** — coreference resolution bias benchmarks
- **Eq. Odds / Demographic Parity** — classical fairness criteria reframed as consistency

This toolkit operationalises those ideas as a reusable Python library.

---

## License

MIT
