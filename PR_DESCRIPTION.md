# PR Title
`fix(conflicts/deduplication): Fix critical bugs and add comprehensive verification for Conflict and Deduplication modules`

# PR Description

## Summary
This PR addresses critical bugs in the Conflict Resolution and Deduplication modules that prevented correct execution of advanced features. It also introduces a comprehensive suite of verification scripts and unit tests to ensure stability and correctness for these components.

## Key Changes

### 1. Conflict Resolution Module (`semantica/conflicts`)
*   **Bug Fix:** Resolved a metadata propagation issue in `ConflictDetector` (`semantica/conflicts/conflict_detector.py`).
    *   *Fix:* Added `metadata` field to source tracking to ensure timestamps and other metadata are available for temporal resolution strategies (e.g., `resolve_most_recent`).
*   **Verification:** Added `scripts/verify_conflict_resolution.py` and `scripts/verify_conflict_complete.py` to validate:
    *   Resolution strategies (Voting, Credibility, Temporal, Confidence).
    *   Advanced features: Conflict Analysis, Investigation Guides, and Custom Method Registration.

### 2. Deduplication Module (`semantica/deduplication`)
*   **Bug Fix:** Resolved a `TypeError: unhashable type: 'dict'` in `SimilarityCalculator` (`semantica/deduplication/similarity_calculator.py`).
    *   *Fix:* Implemented `_make_hashable` helper to handle dictionary and list inputs during similarity calculation.
*   **Unit Tests:** Created `tests/deduplication/test_deduplication.py` covering core classes and functions.
*   **Verification:** Added `scripts/verify_deduplication_notebook.py` to validate the logic from the `18_Deduplication.ipynb` cookbook.

## Verification
*   **Unit Tests:** All new tests in `tests/deduplication/` passed.
*   **Scripts:** 
    *   `scripts/verify_conflict_complete.py`: **PASSED** (Verified advanced conflict features).
    *   `scripts/verify_conflict_resolution.py`: **PASSED** (Verified resolution strategies).
    *   `scripts/verify_deduplication_notebook.py`: **PASSED** (Verified deduplication logic).

## Checklist
- [x] Code compiles and runs without errors.
- [x] Unit tests created and passed.
- [x] Critical bugs fixed (Conflict metadata, Deduplication hashing).
- [x] Verification scripts added and validated.
