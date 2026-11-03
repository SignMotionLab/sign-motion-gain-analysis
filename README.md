# Sign Motion Gain Analysis — Supporting Research Scripts

Supporting research scripts associated with the manuscript:

**Bounded Keypose Error Propagation: An Input-Output Operator Gain and Robustness Analysis of Sign Motion Generators**

## Release status

The manuscript's current reproducibility statement refers to source code, evaluation scripts, and pretrained checkpoints. This script collection alone does not yet satisfy that statement. Additional original experiment artifacts are required before claiming a full reproduction release.

## Repository layout

- `scripts/`: the two figure-preparation scripts.
- `related_experiments/`: ten related research scripts, kept together to preserve sibling-module locations.
- `docs/reproduction_status.md`: reproduction status and limitations.
- `requirements.txt`: an unpinned list of visible third-party dependencies; not a validated environment lock file.

Keep the original script filenames. This repository layout does not repair missing modules or hard-coded data paths. No LICENSE file is included because the authors have not selected a license.

## Relationship to the manuscript

### Figure preparation scripts

- `scripts/generate_fig_gain.py`: prepares Figure 2 using embedded numerical values and interpolated curves. It does not train neural refiners or execute the manuscript's frequency-sweep experiments.
- `scripts/generate_how2sign_gain_experiment_figure.py`: reads local How2Sign RGB frames and OpenPose keypoints and constructs an illustrative figure using fixed convolution kernels. Gain and jerk traces are adjusted to selected manuscript reference values. These traces are not independent measurements of trained CNN checkpoints.

### Related dissipativity experiments

- `related_experiments/dissipative_refiner.py`
- `related_experiments/run_dissipativity_paper_experiment.py`
- `related_experiments/generate_dissipative_dataset_visual_comparison.py`
- `related_experiments/download_phoenix14t_mini.py`

These implement or support a related dissipativity/ODE-refinement workflow.

### Other research scripts

- `related_experiments/evaluate_metrics.py`
- `related_experiments/run_table2_experiment.py`
- `related_experiments/run_table3_ablation_experiment.py`
- `related_experiments/verify_table2_experiment.py`
- `related_experiments/verify_table3_ablation_experiment.py`
- `related_experiments/reconstruct_and_compare_gt_hamer_tbh.py`

These concern additional NSDE, evaluation, verification, or reconstruction workflows. Their filenames must not be interpreted as a verified mapping to the tables in this manuscript. In particular, `related_experiments/run_table2_experiment.py` includes proxy baseline generation and metric formulas containing fixed offsets and random noise; these outputs must not be represented as evaluations of the corresponding published baseline models.

## Environment and execution status

The visible scripts use Python packages including NumPy, SciPy, Matplotlib, OpenCV, and, in one reconstruction workflow, PyTorch. A tested environment specification has not been supplied. Installing these packages alone does not resolve all dependencies.

Missing local modules referenced by the supplied scripts include:

- `SDE20260912`
- `hand_generated`
- `generate_3d_sign_body_hand_combined_detail`

Several scripts contain hard-coded Windows paths such as `D:/How2Sign`, `D:/PHOENIX-2014-T-mini`, and `D:/DPS202360910`. Review and configure input and output paths before execution. Some scripts create output directories at import time.

## Data and checkpoints

PHOENIX-2014T and How2Sign datasets are not included in the supplied collection. Obtain data from the dataset providers under their respective terms. No permission to redistribute third-party datasets is asserted by this repository.

Pretrained checkpoints for the manuscript's four refiner architectures are not included in the supplied collection.

## Requirements for a full reproduction release

1. Original model definitions, training entry points, and tested environment versions.
2. Dataset preparation and normalization procedures, sequence identifiers, and exact train/test splits.
3. Checkpoints and training configuration for every reported model/dataset combination.
4. Anchor-perturbation and frequency-sweep evaluation entry points.
5. Raw per-window measurements, random-seed configuration, and bootstrap computations.
6. A verified mapping from each manuscript table and figure to an executable command and its output.
7. License and attribution information for code and third-party components.

## Citation

Until a public preprint or accepted publication is available, refer to the manuscript title above. No DOI or final publication venue has been assigned in this README.

## License

A code license has not yet been selected for this draft. Add the appropriate LICENSE file and third-party notices before describing the project as open-source software.

