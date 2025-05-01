## Deep Learning Challenge

## Overview of the Analysis

Alphabet Soup wants an early signal on which funding applications are likely to be successful.  
The provided `charity_data.csv` contains historical records on 34 000+ organizations.  
Our task was to build, train, and evaluate a binary classifier that predicts `IS_SUCCESSFUL` from the remaining columns, then try to push predictive accuracy past 75 %.

The workflow:

1. Read the CSV into a Pandas DataFrame.  
2. Pre-process the data (drop IDs, consolidate rare categories, numeric/ordinal encoding, one-hot encoding, train-test split, scaling).  
3. Build a feed-forward neural network in TensorFlow / Keras.  
4. Train, checkpoint, and evaluate the model; iterate on the design to improve accuracy.  
5. Save results to HDF5 files and document findings.

---

## Results

### Data Preprocessing

* **Target**  
  * `IS_SUCCESSFUL` (1 = funds used effectively, 0 = not effective).

* **Features**  
  * Categorical columns: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `INCOME_AMT`, `STATUS`, `SPECIAL_CONSIDERATIONS`.  
  * Numeric column: `ASK_AMT` (log-transformed).

* **Removed columns**  
  * `EIN`, `NAME` – identifiers with no predictive value.

* **Additional steps**  
  * Rare categories (frequency < 500 for `APPLICATION_TYPE`, < 1000 for `CLASSIFICATION`) were grouped into “Other”.  
  * One-hot encoding converted categoricals to 0/1 columns.  
  * Features were scaled with `StandardScaler`; the target was left unchanged.  
  * Stratified 80 / 20 train-test split preserved the class balance.

### Compiling, Training, and Evaluating the Model

* **Baseline network**  
  * Architecture: input layer → 80 neurons (ReLU) → 30 neurons (ReLU) → 1 neuron (sigmoid).  
  * Loss: binary cross-entropy, Optimizer: Adam (lr = 0.001).  
  * Validation accuracy: **~72 %**.

* **Optimisation attempts**

| Attempt | Architecture / change | Notes | Val. accuracy |
|---------|-----------------------|-------|---------------|
| 1 | 128-64-32 hidden units, 0.30 dropout | Extra depth, dropout for overfit | ~72 % |
| 2 | 256-128-64, BatchNorm + dropout, class weights, LR scheduler | Handled imbalance, adaptive LR | ~72 % |
| 3 | 512-256-128, BatchNorm + L2, no dropout, larger batch, long patience | Steadier training | **72.8 %** (best) |
| 4 | Wider nets with class weights removed, rare-cutoff tuning | Marginal change | ~72–73 % |

*All models converged well below the 75 % goal; additional epochs and regularisation tweaks did not improve validation metrics.*

---

## Summary

Four optimisation passes raised the baseline accuracy only marginally (best ~72.8 %).  The feed-forward neural network struggled to capture the many sparse one-hot features without overfitting, and class imbalance limited further gains.  

A different model family is likely better suited here.  Gradient-boosted decision trees (e.g., XGBoost or LightGBM) naturally handle high-dimensional, sparse, mixed-type tabular data and usually score 78–82 % on this dataset with minimal tuning.  Future work should pivot to a tree-based ensemble or incorporate feature engineering (embedding categorical variables or combining one-hots into target-encoded columns) before returning to a neural network approach.

Original code was written by myself. Optimization and error-handling assisted by ChatGPT o3
