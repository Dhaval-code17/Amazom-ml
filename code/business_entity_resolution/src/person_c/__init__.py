# Person C — Matching Model
# Owns: Stage 6 (Feature Engineering), Stage 7 (Pairwise Classifier), Stage 8 (Entity-Level Decision Layer)
#
# Deliverables:
#   - build_features(s1_record, candidate_record) -> dict
#   - Trained LightGBM classifier + calibration + hard-negative mining loop
#   - Threshold / margin / abstention decision logic
#
# Interface contracts consumed from upstream:
#   - Person A: NormalizedRecord schema  (code/business_entity_resolution/src/schema.py)
#               score_macro_f_beta()     (code/business_entity_resolution/src/scoring.py)
#               golden_match()           (code/business_entity_resolution/src/golden_match.py)
#               normalize_record()       (code/business_entity_resolution/src/normalization.py)
#               GroupKFold split util    (code/business_entity_resolution/src/validation.py)
#   - Person B: get_candidates(s1_id) -> set[entity_id]   (person_b/blocking.py)
#               candidate_pairs.tsv produced by person_b/pruning.py
#
# Implementation files (to be added):
#   features.py      — build_features() per (S1, candidate) pair feature vector (Stage 6)
#   classifier.py    — LightGBM training + calibration + hard-negative mining (Stage 7)
#   decision.py      — threshold sweep + margin/abstention + golden-match merge (Stage 8)
#   inference.py     — end-to-end inference: candidates -> matching_results.tsv
#   test_person_c.py — unit tests for all deliverables
