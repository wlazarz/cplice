```
# CPLICE

**Categorical Pseudo-Labeling with Iterative Cluster Expansion**

CPLICE is a research-oriented Python project for pseudo-labeling categorical and mixed-type datasets when only a small, representative subset of objects is initially labeled.

The central CPLICE algorithm assigns provisional labels using categorical dissimilarities, selects the most representative candidates, and iteratively expands the labeled clusters. The repository also provides competitive pseudo-labeling methods, representative-object selection strategies, optimized distance-matrix computation, and internal and external evaluation metrics.

> **Paper**
>
> W. Łazarz and A. Nowak-Brzezińska,  
> “Categorical Pseudo-Labeling with Iterative Cluster Expansion,”  
> *Procedia Computer Science*, vol. 270, pp. 937–946, 2025.  
> DOI: [10.1016/j.procs.2025.09.214](https://doi.org/10.1016/j.procs.2025.09.214)

## Features

- CPLICE pseudo-labeling for categorical data.
- Experimental mixed-data CPLICE extension.
- Reusable pairwise distance matrices.
- Block-wise and memory-mapped matrix computation.
- Categorical distances:
  - Overlap,
  - Hamming,
  - Eskin,
  - Inverse Occurrence Frequency,
  - Lin,
  - categorical Dice,
  - categorical Jaccard,
  - S2 / Morlini–Zani.
- Representative-object selection:
  - centroid contrast,
  - medoid centrality,
  - K-Modes subcluster balancing.
- Competitive pseudo-labeling methods:
  - 1-nearest-neighbor labeling,
  - model-based iterative pseudo-labeling,
  - graph-based Label Spreading.
- Internal clustering evaluation in the same distance geometry used by CPLICE.
- External classification and partition-comparison metrics.
- Support for arbitrary user-provided classification models.
- Example experiment on the UCI Mushroom dataset.

## Project structure

```text
cplice/
├── algorithms/
│   ├── competitive/
│   │   ├── __init__.py
│   │   ├── knn_labeling.py
│   │   ├── label_spreading.py
│   │   └── model_based_labeling.py
│   ├── cplice/
│   │   ├── __init__.py
│   │   ├── cplice.py
│   │   ├── find_representativeness.py
│   │   ├── mplice.py
│   │   └── object_distances.py
│   ├── __init__.py
│   └── labeling_template.py
├── evaluation/
│   ├── __init__.py
│   ├── external_metrics.py
│   └── metrics.py
├── notebooks/
│   └── mushrooms_cplice_project_walkthrough.ipynb
├── LICENSE
├── CITATION.cff
├── README.md
└── requirements.txt
```

## Installation

CPLICE requires Python 3.10 or newer.

## Quick start

### 1. Prepare categorical data

CPLICE expects a pandas `DataFrame` whose rows are objects and whose columns are categorical features.

```python
import pandas as pd

data = pd.DataFrame(
    {
        "color": ["red", "red", "blue", "blue", "green"],
        "shape": ["round", "square", "round", "square", "round"],
        "size": ["small", "small", "large", "large", "small"],
    }
)
```
Missing categorical values should be represented consistently, for example:

```python
data = (
    data.astype("object")
    .where(data.notna(), "__MISSING__")
)
```
### 2. Define initially labeled objects

Initial clusters map semantic class labels to row indices:

```python
initial_clusters = {
    "class_a": [0, 1],
    "class_b": [2, 3],
}
```
Each cluster must contain at least one object, indices must be valid, and the same object cannot belong to multiple initial clusters.

### 3. Compute a reusable distance matrix

```python
import numpy as np

from algorithms.cplice.object_distances import compute_distance_matrix

distance_matrix = compute_distance_matrix(
    data=data,
    metric="lin",
    dtype=np.float32,
)
```
The matrix can be reused by CPLICE, KNN, and internal evaluation metrics.

For a large dataset, write the matrix directly to disk:

```python
distance_matrix = compute_distance_matrix(
    data=data,
    metric="s2",
    dtype=np.float32,
    output_path="cache/s2_distances.dat",
)
```
The returned object is a `numpy.memmap` and can be passed directly to the labeling algorithms.

### 4. Run CPLICE

```python
from algorithms.cplice.cplice import CPLICELabeling

labeler = CPLICELabeling(
    df=data,
    metric="lin",
    strategy="mean",
    distance_matrix=distance_matrix,
)

predicted_labels = labeler.label_data(
    initial_clusters=initial_clusters,
    expansion_rate=0.05,
)
```
The result contains one label for every input row.

## CPLICE strategies

The refactored API uses descriptive strategy names:


| Strategy       | Cluster score                                         |
| -------------- | ----------------------------------------------------- |
| `centroid`     | distance to the mode-based cluster prototype          |
| `nearest`      | minimum distance to a labeled object in the cluster   |
| `farthest`     | maximum distance to a labeled object in the cluster   |
| `mean`         | mean distance to labeled objects in the cluster       |
| `outside_mean` | contrast against objects assigned outside the cluster |

Matrix-based strategies automatically compute a distance matrix when one is not supplied. Passing a precomputed matrix is recommended for expensive distances and repeated experiments.

## Distance measures

```python
distance_matrix = compute_distance_matrix(
    data=data,
    metric="overlap",
)
```
Supported names:

```text
overlap
hamming
dice
jaccard
eskin
iof
lin
s2
```
Frequency and probability statistics are estimated separately for every feature. Categories with identical textual values in different columns are therefore not merged into one global category.

The S2 implementation works directly with categorical data and does not require external one-hot encoding.

### Feature weights

For a pandas `DataFrame`, weights can be supplied by column name:

```python
distance_matrix = compute_distance_matrix(
    data=data,
    metric="lin",
    feature_weights={
        "color": 1.0,
        "shape": 2.0,
        "size": 0.5,
    },
)
```
## Selecting representative initial objects

### Centroid-contrast selection

Selects objects that are typical for their own class and atypical for competing classes:

```python
from algorithms.cplice.find_representativeness import (
    select_representatives_by_centroid_contrast,
)

initial_clusters = select_representatives_by_centroid_contrast(
    data=data.to_numpy(dtype=object),
    class_labels=true_labels,
    number_per_class=10,
    competing_similarity="maximum",
)
```
### Medoid-centrality selection

```python
from algorithms.cplice.find_representativeness import (
    select_representatives_by_medoid,
)

initial_clusters = select_representatives_by_medoid(
    data=data.to_numpy(dtype=object),
    class_labels=true_labels,
    number_per_class=10,
)
```
### K-Modes subcluster selection

Selects central objects in round-robin order across subclusters, promoting within-class diversity:

```python
from algorithms.cplice.find_representativeness import (
    select_representatives_by_subclustering,
)

initial_clusters = select_representatives_by_subclustering(
    data=data.to_numpy(dtype=object),
    class_labels=true_labels,
    number_per_class=10,
    maximum_clusters=10,
    random_state=42,
)
```
## Competitive methods

### KNN labeling

KNN uses the same initial objects and may reuse the same distance matrix:

```python
from algorithms.competitive.knn_labeling import KNNLabeling

knn = KNNLabeling(
    df=data,
    metric="lin",
    distance_matrix=distance_matrix,
)

knn_labels = knn.label_data(initial_clusters)
```
### Model-based pseudo-labeling

`ModelBasedLabeling` accepts arbitrary estimators exposing `fit` and `predict`. Confidence-based iterative expansion additionally uses `predict_proba`, `decision_function`, or a user-provided confidence function.

```python
from sklearn.ensemble import RandomForestClassifier

from algorithms.competitive.model_based_labeling import (
    ModelBasedLabeling,
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

labeler = ModelBasedLabeling(unlabeled_features)

labels, confidences = labeler.label_data(
    labeled_data=labeled_rows_by_class,
    model=model,
    confidence_cutoff=0.95,
    max_iterations=15,
)
```
Compatible estimators include, among others:

- logistic regression,
- random forest,
- AdaBoost,
- SVC and LinearSVC,
- XGBoost,
- CatBoost,
- other scikit-learn-compatible classifiers.

Regression estimators require an explicit transformation from continuous predictions to class labels.

### Label Spreading

```python
from algorithms.competitive.label_spreading import (
    LabelSpreadingLabeling,
)

labeler = LabelSpreadingLabeling(unlabeled_numeric_features)

labels, confidences = labeler.label_data(
    labeled_data=labeled_rows_by_class,
    kernel="knn",
    n_neighbors=15,
    alpha=0.2,
)
```
For categorical or mixed data, provide numeric encoded features or a compatible preprocessing transformer.

After fitting, the wrapper exposes:

```python
labeler.model_
labeler.classes_
labeler.predictions_
labeler.confidences_
labeler.probabilities_
labeler.n_iter_
```
## Experimental mixed-data extension

The project includes an experimental extension for datasets containing both numerical and nominal features.

```python
from algorithms.cplice.mplice import MixedConditionalCPLICELabeling

labeler = MixedConditionalCPLICELabeling(
    df=mixed_data,
    categorical_columns=["segment", "region"],
    numerical_columns=["age", "income"],
    conditional_pairs=[
        ("segment", "income"),
        ("region", "age"),
    ],
)

labels = labeler.label_data(
    initial_clusters=initial_clusters,
    expansion_rate=0.05,
)
```
The method combines robust numerical prototypes, smoothed categorical distributions, and domain-selected conditional category–number profiles. It is an experimental research extension rather than a standard established algorithm.

## Evaluation

### External metrics

External evaluation compares predicted labels with known reference classes:

```python
from evaluation.external_metrics import evaluate_external_labels

report = evaluate_external_labels(
    y_true=true_labels,
    y_pred=predicted_labels,
)

print(report.classification)
print(report.clustering)
print(report.distribution)
```
Classification metrics include:

- accuracy,
- precision,
- recall,
- F1,
- Jaccard,
- Hamming loss,
- Matthews correlation coefficient,
- Cohen’s kappa,
- ROC AUC when continuous class scores are available.

Permutation-invariant partition metrics include:

- adjusted Rand index,
- Fowlkes–Mallows index,
- normalized mutual information,
- variation of information,
- homogeneity,
- completeness,
- V-measure,
- conditional cluster entropy.

ROC AUC must be calculated from class probabilities or decision scores, not from hard predicted labels:

```python
report = evaluate_external_labels(
    y_true=true_labels,
    y_pred=predicted_labels,
    y_score=class_probabilities,
    score_labels=model.classes_,
)
```
### Internal metrics

Internal metrics should use the same distance matrix as CPLICE:

```python
from evaluation.metrics import CPLICEClusteringEvaluator

evaluator = CPLICEClusteringEvaluator(
    labels=predicted_labels,
    distance_matrix=distance_matrix,
    categorical_data=data.to_numpy(dtype=object),
)

report = evaluator.evaluate(
    include_categorical_metrics=True,
    include_experimental_metrics=False,
)

print(report.values())
```
The recommended internal report includes:

- silhouette,
- Dunn index,
- distance-based pseudo-F,
- medoid Davies–Bouldin,
- within-cluster dispersion,
- between-medoid separation,
- dispersion-to-separation ratio,
- normalized categorical entropy,
- mode mismatch rate,
- mode separation.

Experimental M1–M4-style measures are available only when explicitly enabled.

## Example notebook

The repository includes a complete Mushroom dataset walkthrough:

```text
notebooks/mushrooms_cplice_project_walkthrough.ipynb
```
It demonstrates:

- representative selection,
- matrix calculation and caching,
- CPLICE,
- KNN,
- model-based pseudo-labeling,
- Label Spreading,
- external and internal evaluation,
- confidence analysis,
- result export.

Launch JupyterLab from the repository root:

```bash
jupyter lab
```
## Reproducible experiments

For fair comparisons:

1. Use the same initially labeled objects for all methods in a run.
2. Evaluate primarily on objects that were not initially labeled.
3. Reuse the same distance matrix when comparing CPLICE and KNN.
4. Record representative-selection method, initial sample size, metric, strategy, expansion rate, random seed, runtime, and matrix dtype.
5. Repeat experiments for multiple initial samples or random seeds.
6. Report both classification metrics and permutation-invariant partition metrics.
7. Keep matrix-computation time separate from pseudo-labeling runtime.
8. Use full-precision values for aggregation and statistical testing; round only for presentation.

A typical research grid may include:

```python
representative_methods = [
    "centroid_contrast",
    "medoid",
    "subclustering",
]

initial_objects_per_class = [5, 10, 20, 50]

distance_metrics = [
    "overlap",
    "eskin",
    "iof",
    "lin",
    "s2",
]

cplice_strategies = [
    "nearest",
    "mean",
    "farthest",
    "outside_mean",
]
```
## Memory considerations

A dense distance matrix requires approximately:

```text
n_samples × n_samples × dtype_size
```
For example, 8,124 objects require approximately:

- 264 MB with `float32`,
- 528 MB with `float64`.

For larger experiments, use:

- `dtype=np.float32`,
- block-wise computation,
- `output_path` to create a memory-mapped matrix,
- a cache shared by labeling and evaluation.

## Citation

When using CPLICE in academic or scientific work, please cite the original paper:

```bibtex
@article{Lazarz2025CPLICE,
  title   = {Categorical Pseudo-Labeling with Iterative Cluster Expansion},
  author  = {Łazarz, Weronika Olga and Nowak-Brzezińska, Agnieszka Justyna},
  journal = {Procedia Computer Science},
  volume  = {270},
  pages   = {937--946},
  year    = {2025},
  doi     = {10.1016/j.procs.2025.09.214}
}
```
A machine-readable citation is provided in `CITATION.cff`.

## License and attribution

The source code is distributed under the **BSD 3-Clause License**.

You may use, modify, and redistribute the code, including in commercial work, provided that the copyright notice, license conditions, and disclaimer are retained as required by the license.

Please preserve attribution to the code author:

**Weronika Olga Łazarz**

Academic and scientific publications using CPLICE should also cite the paper listed above.

The article and the source code are separate works and may be distributed under different licenses. Refer to the publisher’s page for the article’s publication license.

## Authors

**Code and CPLICE method**

- Weronika Łazarz

**Paper**

- Weronika Łazarz
- Agnieszka Nowak-Brzezińska


```

```
