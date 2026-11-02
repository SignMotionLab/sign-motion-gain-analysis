# Reproduction status

This is a supporting-script collection, not a verified full reproduction release.

## Included material

- Two figure-preparation scripts in `scripts/`.
- Ten related experiment, evaluation, data-preparation, and reconstruction scripts in `related_experiments/`.
- An unpinned list of visible third-party Python dependencies in `requirements.txt`.

All twelve Python files were copied without changing their contents. Their original filenames are preserved. The rearrangement keeps the related scripts together but does not resolve missing dependencies.

## Known limitations

1. The complete training and operator-gain evaluation workflow for all four manuscript architectures is not present in the supplied collection.
2. No pretrained checkpoints, per-window experiment logs, or exact dataset split manifests were supplied.
3. Figure-preparation scripts contain embedded values, interpolation, fixed-kernel simulations, and adjustments to selected reference gains or jerk values. These are not independent evaluations of trained networks.
4. Some other experiment scripts concern NSDE or dissipativity workflows and are not verified reproductions of this manuscript's tables. Some use proxy outputs or formulas with fixed offsets and random noise.
5. Referenced local modules `SDE20260912`, `hand_generated`, and `generate_3d_sign_body_hand_combined_detail` are missing.
6. Scripts contain machine-specific input and output paths on drive D. Review those paths before execution; some scripts create directories at import time.
7. Dependency versions are not pinned. Installation and end-to-end execution have not been verified.
8. A code license has not been selected. No LICENSE file is included. Third-party attribution and licensing still need review before an open-source release.

## Scope of preparation checks

The preparation checks cover file count, expected folder layout, and SHA-256 equality of the copied Python files against the supplied originals. They do not establish numerical reproducibility or validate scientific claims. No experiments were executed during packaging.

The manuscript itself is not included in this folder and was not changed during packaging. Its statement about available code, evaluation scripts, and pretrained checkpoints should be reconciled with the actual release contents before journal submission.
