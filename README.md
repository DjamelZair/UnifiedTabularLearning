# UnifiedTabularLearning

A lightweight guide to set up the environment and run the full tabular learning pipeline (merge data → train autoencoder → evaluate → train MLPs).

## Environment setup
- Python: 3.13 (or compatible).
- Create and activate a virtual environment in the repo root:
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

## Scripts and order of execution
Run each from the repo root with the virtualenv activated.

1) Merge datasets (produces unified data artifacts):
   ```
   python dataset_merge_unified_v6.py
   ```
2) Train the shared autoencoder (saves encoder and AE weights, plus optional latents):
   ```
   python train_autoencoder_v6.py
   ```
3) Evaluate the autoencoder:
   ```
   python evaluate_autoencoder_v6.py
   ```
4) Train/evaluate MLPs on latents:
   - Single-head version:
     ```
     python train_eval_unified_mlp_from_latents_v6.py
     ```
   - Multi-head version:
     ```
     python train_eval_unified_mlp_multihead_v6.py
     ```

## Dependencies (requirements.txt)
- numpy
- pandas
- scikit-learn
- joblib
- torch
- matplotlib
- tqdm

### Sequence of running files

- Requirements are:
+numpy
+pandas
+scikit-learn
+joblib
+torch
+matplotlib
+tqdm

- Install dependencies, create a venv via:
`cd UnifiedTabularLearning`
`python3 -m venv .venv`      # only if needed
`source .venv/bin/activate`
`pip install -r requirements.txt`

- Merge datasets via `python dataset_merge_unified_v6.py`
- Train autoencoder via `python train_autoencoder_v6.py`
- Evaluate autoencoder via ` python evaluate_autoencoder_v6.py`
- MLPs via `python train_eval_unified_mlp_from_latens_v6.py` and `python train_eval_unified_mlp_multihead_v6.py`
- Visualise t-SNE of latent space via `python visualise_embeddings.py`

Minimal guide to go from raw Kaggle CSVs → merged unified features → AE latents → classifiers → Kaggle submissions.

## Prerequisites
- Place the Kaggle CSVs under `datasets/` with these exact names:
  - `covtype_train.csv`, `covtype_test.csv`
  - `higgs_train.csv`, `higgs_test.csv`
  - `heloc_train.csv`, `heloc_test.csv`
- Python env with the usual stack: numpy, pandas, scikit-learn, torch, xgboost, joblib, tqdm, matplotlib (for plots).

## 1) Build unified training data
Creates `merged_data/unified_data_v6.npz` + scaler/imputer metadata.
```bash
python dataset_merge_unified_v6.py
```

## 2) Train autoencoder and export latents
Trains AE on the merged design matrix (numeric_z | binary | native_mask | struct_mask) and saves:
- `models/encoder.pt`, `models/autoencoder.pt`
- `merged_data/trainval_latents_v6.npy` (latents for train+val rows)
```bash
python train_autoencoder_v6.py
```

## 3) Train classifiers on latents
- Single-head MLP (unified labels): `python train_eval_unified_mlp_from_latents_v6.py`  
  Saves checkpoint to `results/results_singlehead/unified_mlp_singlehead_best_v6.pt`.
- Multi-head MLP (separate heads per dataset): `python train_eval_unified_mlp_multihead_v6.py`  
  Saves checkpoint to `results/results_multihead/unified_mlp_multihead_best_v7.pt`.
- XGBoost baseline: `python train_eval_xgboost_from_latents_v6.py`  
  Saves model to `models/xgb_unified_latents_v6.json`.

## 4) Build Kaggle test features and latents
Aligns test CSVs to training features, applies scaler/imputer, adds masks/one-hot, then encodes to latents.
Outputs: `merged_data/test_preprocessed_v6.npz` + `merged_data/test_latents_v6.npy`.
```bash
python scripts/build_and_encode_test_v6.py
```

## 5) Run test-time inference
- Single-head MLP: `python scripts/predict_singlehead_test_v6.py`  
  Writes `results/test_predictions_v6.csv`.
- Multi-head MLP: `python scripts/predict_multihead_test_v6.py`  
  Writes `results/metrics/test_predictions_multihead_v6.csv`.
- XGBoost baseline: (add similar inference script if needed)  
  Load `merged_data/test_latents_v6.npy` + `models/xgb_unified_latents_v6.json` and write preds with columns `dataset_id`, `y_pred_unified`.

## 6) Build Kaggle submission CSVs
Takes prediction files and formats per-competition submissions.
- Single-head combined: `python scripts/make_submission_v6.py`  
  → `results/submissions/submission_singlehead_v6.csv`
- Multi-head combined: `python scripts/make_submission_multihead_v6.py`  
  → `results/metrics/submission_multihead_v6.csv` (and per-dataset splits)
- Per-dataset helpers also exist: `scripts/make_submission_covtype_v6.py`, `scripts/make_submission_heloc_v6.py`.

## Notes and pitfalls
- Do not overwrite `merged_data/test_latents_v6.npy` with internal-eval latents; `evaluate_autoencoder_v6.py` now writes its own `internal_test_latents_v6.npy`.
- AE encoder expects the no-onehot layout `[numeric_z | binary | native_mask | struct_mask]`; the test encoder script slices this automatically.
- Submission builders clip any out-of-block predictions into the valid label ranges per dataset, logging a warning.