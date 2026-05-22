# When Machine Learning Fails — Mini-Project ECC Spring 2026
## Diagnosing and Repairing Class Imbalance Failure on the Online Shoppers Dataset

---

## Project Overview

This repository contains a research-oriented mini-project for the course **Introduction to AI and Machine Learning — ECC Spring 2026**.

The objective is not only to train a good classifier, but to deliberately expose, measure, explain, and repair a real machine-learning failure mode: **class imbalance** on the **Online Shoppers Purchasing Intention** dataset.

The project follows the scientific structure required in the assignment:

> **research question → observed symptom → causal hypothesis → controlled experiment → targeted correction → threats to validity**

The main failure mode investigated is **class imbalance failure**. A bonus notebook investigates a second failure mode: **shortcut learning through the `Month` feature**.

---

## Research Question

Does the natural class imbalance of the Online Shoppers dataset, where most sessions do not lead to a purchase, cause non-linear classifiers to obtain acceptable global accuracy while failing to correctly detect the minority class `Revenue=True`?

More specifically:

> Can imbalance-aware corrections such as **SMOTE**, **AdaBoost**, and controlled diagnostic experiments improve the detection of purchasing sessions without relying on misleading aggregate accuracy?

---

## Dataset

| Property | Value |
|---|---|
| Dataset | Online Shoppers Purchasing Intention |
| UCI ID | 468 |
| Task | Binary classification |
| Target | `Revenue` |
| Positive class | `Revenue=True` — purchase |
| Negative class | `Revenue=False` — no purchase |
| Number of observations | 12,330 web sessions |
| Feature types | Numerical + categorical |
| Main failure mode | Class imbalance |
| Bonus failure mode | Shortcut learning / seasonal shortcut through `Month` |

The target distribution is strongly imbalanced: most sessions correspond to `Revenue=False`, while only a small minority correspond to `Revenue=True`. For this reason, **accuracy alone is not a reliable metric**.

---

## Assignment Alignment

The project satisfies the core requirements of the ECC mini-project:

| Requirement | Implementation in this repository |
|---|---|
| Use an authorized dataset | Online Shoppers, UCI 468 |
| Use non-linear models | Random Forest, MLP, LightGBM, AdaBoost-based pipelines |
| Investigate a failure mode | Class imbalance failure |
| Make the failure measurable | Recall, F1, PR-AUC, confusion matrices, degradation experiment |
| Design a controlled experiment | Artificial positive-class degradation experiment |
| Propose a targeted correction | SMOTE + AdaBoost + non-linear classifiers |
| Discuss limitations | Threats to validity are included in the notebooks and should be expanded in the final report |
| Bonus failure mode | Shortcut learning via `Month` |

---

## Repository Structure

The current project is organized around four notebooks.

| Order | Notebook | Role |
|---:|---|---|
| 1 | `Untouched_class_imbalance_NB1(1).ipynb` | Establishes the broken baseline using non-linear models without imbalance correction |
| 2 | `Correction_NB2(2).ipynb` | Applies imbalance-aware correction using SMOTE + AdaBoost + RF / MLP / LightGBM |
| 3 | `Experiment.ipynb` | Runs the controlled degradation experiment to test the causal role of imbalance |
| 4 | `Bonus_Shortcut_Learning_Month_NB3.ipynb` | Bonus investigation of shortcut learning through the `Month` feature |

Recommended clean names before submitting to GitHub:

```text
01_Untouched_Class_Imbalance_Baseline.ipynb
02_SMOTE_AdaBoost_Imbalance_Correction.ipynb
03_Controlled_Degradation_Experiment.ipynb
04_Bonus_Shortcut_Learning_Month.ipynb
```

---

## Notebook 1 — Broken Baseline Without Correction

**File:** `Untouched_class_imbalance_NB1(1).ipynb`

This notebook establishes the reference performance of non-linear classifiers trained without any class-imbalance correction.

### Main goals

- Load and inspect the Online Shoppers dataset.
- Perform exploratory data analysis.
- Analyze the target distribution and feature skewness.
- Apply anti-leakage preprocessing.
- Train several non-linear models:
  - Random Forest
  - Multi-Layer Perceptron
  - LightGBM
- Compare models on validation data.
- Evaluate the best untreated model on the test set.
- Show that accuracy can hide poor minority-class detection.

### Scientific role

This notebook corresponds to the **observed symptom** part of the project.

The expected symptom is:

> The model can reach acceptable global accuracy while still missing many `Revenue=True` cases.

This is why the project focuses on minority-class metrics rather than only accuracy.

### Main outputs

```text
notebook1_untouched_results.csv
eda_overview.png
skewness_analysis.png
notebook1_final_evaluation.png
```

---

## Notebook 2 — SMOTE + AdaBoost Correction

**File:** `Correction_NB2(2).ipynb`

This notebook implements the main correction strategy for the class imbalance problem.

### Main goals

- Reuse the same dataset and preprocessing logic as Notebook 1.
- Keep the train / validation / test separation before any resampling.
- Build imbalance-aware pipelines using `imblearn`.
- Apply **SMOTE** only on the training data.
- Combine SMOTE with AdaBoost-inspired correction strategies.
- Evaluate corrected RF, MLP, and LightGBM variants.
- Compare corrected models against untreated baselines.
- Use Stratified K-Fold cross-validation for robustness.
- Select the best corrected model using metrics relevant to the minority class.

### Scientific role

This notebook corresponds to the **correction that targets the cause**.

The cause is not simply “bad hyperparameters”; it is the fact that the training distribution is dominated by the majority class. SMOTE targets the data-level imbalance by synthesizing minority-class samples, while AdaBoost increases attention to difficult examples.

### Main outputs

```text
baseline_results_notebook2.csv
notebook2_smote_adaboost_validation_results.csv
notebook2_smote_adaboost_test_results.csv
eda_overview_notebook2.png
skewness_analysis_notebook2.png
notebook2_model_comparison.png
notebook2_final_evaluation.png
```

---

## Notebook 3 — Controlled Degradation Experiment

**File:** `Experiment.ipynb`

This notebook tests the causal hypothesis that worsening class imbalance directly damages the model's ability to detect the positive class.

### Main goals

- Start from the Online Shoppers dataset.
- Create several degraded training sets with fewer positive examples.
- Train comparable non-linear models under different imbalance ratios.
- Measure how recall, F1-score, PR-AUC, and accuracy evolve as the positive class becomes rarer.
- Visualize the relationship between imbalance severity and minority-class failure.

### Scientific role

This notebook corresponds to the **causal hypothesis and controlled test**.

The hypothesis is falsifiable:

> If class imbalance is a real cause of the failure, then artificially reducing the proportion of `Revenue=True` examples in training should reduce recall and PR-AUC on the positive class.

The control is the original or less degraded training condition. The treatment is the increasingly imbalanced training condition.

### Main outputs

```text
controlled_degradation_results.csv
controlled_degradation_experiment.png
```

---

## Notebook 4 — Bonus: Shortcut Learning Through `Month`

**File:** `Bonus_Shortcut_Learning_Month_NB3.ipynb`

This notebook investigates a second failure mode: **shortcut learning**.

The suspected shortcut is the categorical feature `Month`. If purchasing behavior is seasonally correlated, a model may exploit `Month` as a shortcut instead of learning more robust behavioral patterns.

### Main goals

- Analyze purchase rate by month.
- Train models with and without the `Month` feature.
- Compare in-distribution random split performance.
- Create an out-of-distribution evaluation using November / December as held-out months.
- Measure feature importance and permutation importance.
- Test whether permuting `Month` damages performance.
- Evaluate whether removing `Month` improves robustness on unseen months.

### Scientific role

This notebook is the **bonus second failure mode**, treated using the same four-step logic:

1. Research question
2. Observed symptom
3. Causal hypothesis and controlled experiment
4. Correction and evaluation

### Main outputs

```text
bonus_shortcut_month_random_split_results.csv
bonus_shortcut_month_ood_results.csv
bonus_shortcut_month_permutation_results.csv
bonus_shortcut_month_feature_importance.csv
bonus_month_purchase_rate.png
bonus_month_random_split_comparison.png
bonus_month_feature_importance_rf.png
bonus_month_feature_importance_lgbm.png
bonus_month_permutation_importance_mlp.png
bonus_month_ood_comparison.png
bonus_month_permutation_test.png
bonus_month_confusion_matrices.png
```

---

## Methodology

### 1. Symptom

The first symptom is that non-linear models can appear successful when measured with accuracy, while still failing to detect many buyers.

In an imbalanced classification task, a model can obtain high accuracy by mostly predicting the majority class. This is dangerous because the business-relevant class is the minority class: `Revenue=True`.

### 2. Causal hypothesis

The central hypothesis is:

> The low recall on `Revenue=True` is caused by the imbalance of the training distribution, which biases the learned decision boundary toward the majority class.

### 3. Controlled test

The controlled degradation experiment artificially reduces the number of positive examples in the training data.

If recall and PR-AUC degrade as the imbalance becomes more severe, this supports the hypothesis that class imbalance is a causal factor in the observed failure.

### 4. Correction

The correction uses:

- **SMOTE**, to generate synthetic minority examples in the training set;
- **AdaBoost**, to focus learning on difficult examples;
- **non-linear classifiers**, to respect the assignment constraint and model complex decision boundaries.

The correction is evaluated by comparing before and after results on the same validation and test protocol.

---

## Data Leakage Prevention

The notebooks follow an anti-leakage protocol.

| Rule | Implementation |
|---|---|
| Split first | Train / validation / test are created before fitting scalers, encoders, or resampling methods |
| Preprocessing | Scaler and encoder are fitted only on the training set |
| SMOTE | Applied only inside the training pipeline, never before the split |
| Validation | Used for model comparison and selection |
| Test set | Reserved for final evaluation |
| Bonus OOD test | November / December are held out for out-of-distribution evaluation |

---

## Metrics

Because the dataset is imbalanced, the most important metrics are those that evaluate the positive class.

| Metric | Why it matters |
|---|---|
| `Recall_true` | Measures how many buyers are correctly detected |
| `Precision_true` | Measures how reliable the positive predictions are |
| `F1_true` | Balances precision and recall for `Revenue=True` |
| `PR_AUC` | More informative than ROC-AUC when the positive class is rare |
| `ROC_AUC` | Useful complementary ranking metric |
| `Accuracy` | Reported, but not sufficient alone |
| Confusion matrix | Makes false negatives and false positives visible |

The main project argument should not rely on accuracy alone.

---

## Installation

Create a virtual environment, then install the requirements.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS / Linux:

```bash
source .venv/bin/activate
```

Then install the packages:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` for this project is:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
lightgbm
pyswarms
jupyter
ipykernel
```

`google.colab` is not included because it is available only inside Google Colab and should not be installed locally.

---

## How to Run the Project

### Option 1 — Google Colab

1. Open each notebook in Google Colab.
2. Upload `online_shoppers_intention.csv` when requested.
3. Run the notebooks in the recommended order:

```text
1. Untouched_class_imbalance_NB1(1).ipynb
2. Correction_NB2(2).ipynb
3. Experiment.ipynb
4. Bonus_Shortcut_Learning_Month_NB3.ipynb
```

4. Download the generated CSV files and figures if needed for the report.

### Option 2 — Local Jupyter

Place the following files in the same folder:

```text
online_shoppers_intention.csv
Untouched_class_imbalance_NB1(1).ipynb
Correction_NB2(2).ipynb
Experiment.ipynb
Bonus_Shortcut_Learning_Month_NB3.ipynb
requirements.txt
```

Then run:

```bash
jupyter notebook
```

Open and execute the notebooks in order.

---

## Expected Outputs

After running all notebooks, the repository may contain outputs such as:

```text
notebook1_untouched_results.csv
baseline_results_notebook2.csv
notebook2_smote_adaboost_validation_results.csv
notebook2_smote_adaboost_test_results.csv
controlled_degradation_results.csv
bonus_shortcut_month_random_split_results.csv
bonus_shortcut_month_ood_results.csv
bonus_shortcut_month_permutation_results.csv
bonus_shortcut_month_feature_importance.csv

eda_overview.png
skewness_analysis.png
notebook1_final_evaluation.png
eda_overview_notebook2.png
skewness_analysis_notebook2.png
notebook2_model_comparison.png
notebook2_final_evaluation.png
controlled_degradation_experiment.png
bonus_month_purchase_rate.png
bonus_month_random_split_comparison.png
bonus_month_feature_importance_rf.png
bonus_month_feature_importance_lgbm.png
bonus_month_permutation_importance_mlp.png
bonus_month_ood_comparison.png
bonus_month_permutation_test.png
bonus_month_confusion_matrices.png
```

---

## Report Structure

The written report follows this exact structure:

1. Research question and chosen dataset
2. Reference model and observed symptom
3. Causal hypothesis and controlled experiment
4. Proposed correction and evaluation
5. Threats to validity
6. Conclusion and what was learned


---

## Threats to Validity

The project discusses the following limitations:

- The measured gains may depend on the random seed.
- The test set may not be large enough to make all metric differences statistically conclusive.
- SMOTE may improve recall partly by changing the decision boundary in ways not fully explained by the causal hypothesis.
- Synthetic samples may not perfectly represent real purchasing sessions.
- Online shopping behavior can vary by season, traffic source, geography, or marketing campaigns.
- The conclusions are specific to this dataset and should not be automatically generalized to all e-commerce datasets.
- Any preprocessing or resampling performed before splitting would create leakage and invalidate the comparison.

---

## Reproducibility

All notebooks use a fixed random seed:

```python
RANDOM_STATE = 42
np.random.seed(42)
```

This makes the experiments easier to reproduce, although small differences can still occur depending on package versions and execution environment.

---


## Team

**SAAIDI Salma**  
**EL BAHTOURI Aya**

---

## Project Summary

This mini-project shows that machine-learning failure is not always visible through aggregate performance. On the Online Shoppers dataset, the class imbalance can make non-linear classifiers look acceptable while they still fail on the minority class. The project diagnoses this failure with appropriate metrics, tests the causal role of imbalance through a controlled degradation experiment, applies targeted corrections with SMOTE and AdaBoost-based pipelines, and extends the investigation with a bonus study of shortcut learning through the `Month` feature.