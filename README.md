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
