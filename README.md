# Business Entity Resolution

An end-to-end Machine Learning pipeline for resolving noisy business entity records across independent data sources without shared unique identifiers.

---

## 1. Problem Overview

In large-scale commercial platforms, business entity records arrive from multiple independent data sources (Source 1, Source 2, Source 3) with varying formatting, noise, missing fields, and typos. Matching must be performed using free-text business names, addresses, and country labels.

The goal is to map each entity in **Source 1** (the deduplicated reference source) to zero, one, or many matching records in **Source 2** and **Source 3**.

---

## 2. Fixed Pipeline Architecture

The solution follows a strict 8-stage architecture:

1. **Stage 1 — Normalization**: Multi-representation feature derivation (lowercasing, transliteration, legal suffix removal, Double Metaphone phonetics, character $n$-grams, postal code/city/street number extraction, missingness flags).
2. **Stage 2 — Deterministic Golden-Match Layer**: High-precision deterministic rule engine to resolve obvious matches prior to candidate generation.
3. **Stage 3 — Multi-Channel Blocking**: Candidate generation via token inverted index, character $n$-gram MinHash/LSH, coarse address/city blocker, and phonetic lookup.
4. **Stage 4 — Candidate Recall Measurement**: Validation gate to measure candidate recall ceiling (targeting $\ge 99\%$).
5. **Stage 5 — Cheap Pre-Ranking / Pruning**: Bound candidate list per Source 1 entity (outputs `candidate_pairs.tsv`).
6. **Stage 6 — Feature Engineering**: Pairwise similarity vectors (Jaccard, Levenshtein, Jaro-Winkler, TF-IDF cosine, acronym match, address overlap, missingness interaction flags).
7. **Stage 7 — Pairwise Classifier**: LightGBM binary classifier trained with iterative hard-negative mining and probability calibration.
8. **Stage 8 — Entity-Level Decision Layer**: Per-entity thresholding, margin/abstention logic, singleton handling, and output generation (`matching_results.tsv`).

---

## 3. Team Responsibilities

- **Person A (Data & Normalization Lead)**:
  - Stage 1: Normalization (`normalize_record`)
  - Stage 2: Deterministic Golden Match (`golden_match`)
  - Stage 12: Validation & Scoring Harness (`score_macro_f_beta`)
  - GroupKFold Validation Split Utility (`get_group_kfold_splits`)
  - Initial Dataset EDA (`eda.py`, `code/eda_report.md`)
  - Test Suite (`test_person_a.py`)

- **Person B (Retrieval & Blocking Lead)**:
  - Stage 3: Multi-Channel Blocking (`get_candidates`)
  - Stage 4: Candidate Recall Measurement
  - Stage 5: Cheap Pre-Ranking & Pruning (`candidate_pairs.tsv`)

- **Person C (Matching Model Lead)**:
  - Stage 6: Pairwise Feature Engineering (`build_features`)
  - Stage 7: Pairwise Classifier & Calibration (LightGBM)
  - Stage 8: Entity-Level Decision Layer (`matching_results.tsv`)

---

## 4. Tech Stack

- **Language**: Python 3.10+
- **Data Orchestration**: `pandas`, `pyarrow`, `numpy`
- **String Normalization & Phonetics**: `unidecode`, `jellyfish`, `metaphone`, `rapidfuzz`
- **Machine Learning & Validation**: `scikit-learn`, `lightgbm`, `datasketch`
- **Testing**: `pytest`

---

## 5. Repository Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── Documentation_template.md
├── dataset/                    # Local raw data (git-ignored)
│   ├── train/
│   └── test/
├── utils/
│   └── validate_submission.py
└── code/
    ├── eda_report.md
    └── business_entity_resolution/
        └── src/
            ├── __init__.py
            ├── schema.py
            ├── normalization.py
            ├── golden_match.py
            ├── scoring.py
            ├── validation.py
            ├── eda.py
            └── test_person_a.py
```

---

## 6. Dataset Instructions

The dataset files are large and are **NOT** stored in the GitHub repository.

Place the dataset files locally under the `dataset/` directory following this layout:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## 7. Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/business-entity-resolution.git
   cd business-entity-resolution
   ```

2. **Create and activate a Python virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # On Linux/macOS
   # Or on Windows PowerShell:
   # .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Person A's test suite**:
   ```bash
   python -m pytest code/business_entity_resolution/src/test_person_a.py
   ```
