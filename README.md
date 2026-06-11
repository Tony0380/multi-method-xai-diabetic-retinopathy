# Multi-Method Rule Extraction from Deep Learning for Interpretable Diabetic Retinopathy Grading

Bachelor's Thesis Project

**Author:** Antonio Colamartino
**Email:** a.colamartino6@studenti.uniba.it
**Student ID:** 778730
**Institution:** University of Bari Aldo Moro (UniBA), Department of Computer Science

> **Paper:** a manuscript based on this work is currently in preparation.

---

## Abstract

This project proposes a hybrid system for diabetic retinopathy (DR) grading that combines the predictive performance of deep neural networks with the interpretability of extracted rules. The primary goal is a clinical decision support system that is both accurate and understandable to clinicians.

Diabetic retinopathy is one of the leading causes of blindness worldwide, and early diagnosis is essential to prevent vision loss. Deep learning models have shown excellent predictive capabilities, but their black-box nature limits adoption in clinical settings, where transparency of decisions is critical.

## Objectives

1. **Training a high-performance CNN** for multi-class DR grading
2. **Explainability analysis** through saliency mapping techniques and quantitative validation
3. **Extraction of interpretable rules** using three different methodologies
4. **Development of a hybrid system** that integrates the CNN and the rules under different operating modes

## Methodology

### Phase 1: Teacher CNN Training

| Component     | Specification                                |
| ------------- | -------------------------------------------- |
| Architecture  | EfficientNet-B5 (30M parameters)             |
| Task          | Multi-class classification (5 DR grades)     |
| Classes       | No DR, Mild, Moderate, Severe, Proliferative |
| Loss function | Focal Loss + Label Smoothing (epsilon=0.1)   |

### Phase 2: Explainability Analysis

- **Saliency maps:** Grad-CAM++ for visualising the regions relevant to the prediction
- **Quantitative validation:** Deletion/Insertion AUC, IoU against ground-truth lesions (DDR dataset)
- **Layer comparison:** multi-layer analysis to identify the optimal level of abstraction

### Phase 3: Rule Extraction

Three methods compared for the extraction of interpretable rules:

| Method                         | Output                                |
| ------------------------------ | ------------------------------------- |
| A - Decision Tree Distillation | Interpretable tree with if-then rules |
| B - LIME-based Explanations    | Local interpretable explanations      |
| C - RIPPER One-vs-Rest         | Ordered rule list (RIPPER algorithm)  |

### Phase 4: Hybrid System

| Mode                   | Strategy                                                 |
| ---------------------- | -------------------------------------------------------- |
| Post-hoc Explanation   | CNN prediction with post-hoc rule explanation via DT     |
| Rule-guided Prediction | IF DT confidence > threshold THEN use DT ELSE use CNN    |
| Weighted Ensemble      | alpha * CNN_proba + (1-alpha) * DT_proba                 |

## Datasets

| Dataset                  | Images | Role                                                  |
| ------------------------ | ------ | ----------------------------------------------------- |
| APTOS 2019               | 3,296  | Primary corpus (merged with EyePACS)                  |
| Kaggle EyePACS 2015      | 35,126 | Primary corpus (37,933 images in total after cleaning) |
| Messidor-2               | 1,744  | Earlier cross-dataset experiments                     |
| DDR (lesion annotations) | 12,522 | XAI validation (pixel-level masks)                    |

The combined APTOS + EyePACS corpus (37,933 fundus images after cleaning and merging) is partitioned into training (80%, n=30,346), validation (10%, n=3,793) and internal test (10%, n=3,794) sets using a stratified split.

### Data Directory Layout

```
Data/
├── APTOS2019/
│   ├── train_images/
│   ├── val_images/
│   ├── test_images/
│   ├── train_1.csv
│   ├── valid.csv
│   └── test.csv
├── EyePACS2015/
│   ├── train/
│   ├── test/
│   └── trainLabels.csv
├── messidor-2/
│   ├── images/
│   └── messidor_data.csv
└── DDR Dataset/
    └── DDR-dataset/
        ├── DR_grading/
        │   ├── train/
        │   ├── valid/
        │   ├── test/
        │   └── *.txt (annotations)
        ├── lesion_segmentation/
        │   ├── train/
        │   ├── valid/
        │   └── test/
        └── lesion_detection/
```

## Exploratory Data Analysis

The exploratory analysis of the datasets highlighted the following characteristics.

### Dataset Statistics

| Dataset      | Total images | Use                                     |
| ------------ | ------------ | --------------------------------------- |
| APTOS 2019   | 3,296        | Primary corpus                          |
| EyePACS 2015 | 35,126       | Primary corpus                          |
| Messidor-2   | 1,744        | Earlier cross-dataset experiments       |
| DDR          | 12,522       | XAI validation                          |

**Combined dataset:** 37,933 images (APTOS + EyePACS after cleaning), stratified 80/10/10 split

### Class Distribution

The class distribution of the combined training set shows a significant imbalance, typical of medical datasets:

| Class | Name          | Images | Share |
| ----- | ------------- | ------ | ----- |
| 0     | No DR         | 27,244 | 71.8% |
| 1     | Mild NPDR     | 2,743  | 7.2%  |
| 2     | Moderate NPDR | 6,100  | 16.1% |
| 3     | Severe NPDR   | 1,027  | 2.7%  |
| 4     | Proliferative | 819    | 2.2%  |

**Maximum imbalance ratio:** 33.3:1 (class 0 vs class 4)

### Visualisations

#### Distribution by Dataset

![Class distribution by dataset](results/class_distribution_by_dataset.png)

#### Cross-dataset Comparison

![Class distribution comparison](results/class_distribution_comparison.png)

### Note on the DDR Dataset

The DDR dataset originally includes a class 5 ("ungradable"), which was excluded from the analysis as it does not represent a DR severity grade but rather marks images that cannot be graded.

---

## Image Preprocessing

All images are preprocessed with a pipeline configurable through `config.yaml`. Each technique can be enabled or disabled individually.

### Preprocessing Pipeline

| Step | Technique         | Description                                              | Parameters                   |
| ---- | ----------------- | -------------------------------------------------------- | ---------------------------- |
| 1    | Black border crop | Automatic removal of the dark borders around the retina  | tolerance: 7                 |
| 2    | Resize            | Resizing while preserving the aspect ratio               | 456x456, interpolation: area |
| 3    | CLAHE             | Contrast Limited Adaptive Histogram Equalization         | clip_limit: 2.0, grid: 8x8   |
| 4    | Ben Graham        | Local mean subtraction for illumination normalisation    | sigma: 10                    |
| 5    | Circle crop       | Circular mask to make the images uniform                 | -                            |
| 6    | Normalize         | Normalisation with ImageNet statistics                   | mean: [0.485, 0.456, 0.406]  |

### Configuration (config.yaml)

```yaml
preprocessing:
  target_size: 456
  techniques:
    black_border_crop:
      enabled: true
      tolerance: 7
    resize:
      enabled: true
    clahe:
      enabled: true
      clip_limit: 2.0
    ben_graham:
      enabled: true
      sigma: 10
    circle_crop:
      enabled: true
    normalize:
      enabled: false
```

### Preprocessed Image Statistics

All datasets are processed with the same pipeline before entering the training and evaluation stages.

### Pipeline Visualisation

![Preprocessing pipeline](results/preprocessing_pipeline.png)

---

## Data Augmentation

To improve model generalisation and counteract overfitting, a data augmentation pipeline is applied during training.

### Applied Transformations

| Transformation      | Parameters              | Probability |
| ------------------- | ----------------------- | ----------- |
| Horizontal flip     | -                       | 50%         |
| Vertical flip       | -                       | 50%         |
| Rotation            | up to 180 degrees       | 50%         |
| Brightness/Contrast | +/- 20%                 | 30%         |
| Hue/Saturation      | hue +/- 10, sat +/- 20% | 30%         |

### Class Imbalance Handling

Given the strong disproportion between classes, class weights are used to balance the loss function:

| Class | Name          | Weight |
| ----- | ------------- | ------ |
| 0     | No DR         | 0.28   |
| 1     | Mild          | 2.77   |
| 2     | Moderate      | 1.25   |
| 3     | Severe        | 7.43   |
| 4     | Proliferative | 8.44   |

### Augmentation Examples

![Augmentation examples](results/augmentation_examples.png)

---

## Model Training

### Training Configuration

| Parameter       | Value                                          |
| --------------- | ---------------------------------------------- |
| Architecture    | EfficientNet-B5                                |
| Input size      | 456x456                                        |
| Batch size      | 16                                             |
| Optimizer       | AdamW                                          |
| Learning rate   | 1e-4                                           |
| Scheduler       | Cosine annealing with warmup (2 epochs)        |
| Loss function   | Focal Loss (gamma=2.0) + Label Smoothing (0.1) |
| Early stopping  | Patience 7 on val_kappa                        |
| Mixed precision | AMP enabled                                    |

### Training Strategy

1. **Frozen backbone** (3 epochs): only the classifier head is trained
2. **Unfreeze** (subsequent epochs): fine-tuning of the entire network
3. **Early stopping**: training stops when val_kappa does not improve for 7 epochs

### Training Results

| Set           | Accuracy | Quadratic Weighted Kappa |
| ------------- | -------- | ------------------------ |
| Validation    | 82.86%   | 0.822                    |
| Internal test | 82.95%   | 0.829                    |

The Mild NPDR class remains the most challenging across all sets (F1-score around 0.39), due to the subtlety of isolated microaneurysms and the high inter-rater variability reported in the literature.

> **Note on repository artifacts.** The `results/` directory also contains metrics and plots from earlier experimental runs (including configurations with Messidor-2 used as an external test set), which differ from the final configuration reported above.

### Training Curves

![Training history](results/training_history.png)

### Confusion Matrices

#### Validation Set (APTOS)

![Confusion matrix, validation](results/confusion_matrix_val.png)

#### Test Set (Messidor-2)

![Confusion matrix, test](results/confusion_matrix_test.png)

### Generated Artifacts

| File                             | Description                            |
| -------------------------------- | -------------------------------------- |
| `checkpoints/best_model_*.pt`    | Model with the best val_kappa          |
| `checkpoints/last_model_*.pt`    | Last trained model                     |
| `results/training_history.png`   | Loss, accuracy, kappa and LR curves    |
| `results/confusion_matrix_*.png` | Confusion matrices                     |
| `results/final_metrics_*.json`   | Final metrics                          |

---

## Explainability Analysis (Grad-CAM++)

### Approach

Grad-CAM++ is used to generate saliency maps that highlight the image regions most relevant to the model prediction. The analysis was carried out on three layers of the network to identify the optimal level of abstraction.

### Layer Comparison

Three layers of EfficientNet-B5 were compared to determine which produces the most informative activation maps:

| Layer              | Resolution | Deletion AUC | Insertion AUC | Composite score |
| ------------------ | ---------- | ------------ | ------------- | --------------- |
| blocks.2           | 57x57      | 0.388        | 0.428         | -               |
| **blocks.4**       | **29x29**  | **0.330**    | **0.454**     | **0.125**       |
| bn2                | 15x15      | 0.436        | 0.483         | -               |

**Selected layer:** `blocks.4` (29x29), chosen on the basis of the composite score (Insertion AUC - Deletion AUC), which balances the ability to identify relevant regions (high Insertion) with specificity (low Deletion).

![Layer comparison](results/gradcam/gradcam_layer_comparison.png)

### Per-class Deletion/Insertion AUC

| Class         | Deletion AUC | Insertion AUC |
| ------------- | ------------ | ------------- |
| No DR         | 0.622        | 0.606         |
| Mild          | 0.344        | 0.400         |
| Moderate      | 0.195        | 0.333         |
| Severe        | 0.145        | 0.317         |
| Proliferative | 0.341        | 0.614         |
| **Overall**   | **0.330**    | **0.454**     |

![Deletion/Insertion curves](results/gradcam/deletion_insertion_curves.png)

### Ground-truth Validation (DDR Dataset)

The quantitative validation was carried out on 100 images of the DDR dataset with pixel-level lesion annotations:

| Metric                     | Value |
| -------------------------- | ----- |
| Pointing game accuracy     | 14.0% |
| Mean IoU (threshold 0.3)   | 0.039 |
| Mean IoU (threshold 0.5)   | 0.045 |
| Mean IoU (threshold 0.7)   | 0.036 |

The low IoU values indicate that Grad-CAM++ captures broader regions than the point-wise annotations of individual lesions - an expected behaviour, since the model learns contextual patterns rather than the precise localisation of each lesion.

### Visualisations

#### Grad-CAM++ Grid by Class

![Grad-CAM grid](results/gradcam/gradcam_grid_by_class.png)

#### Mean Heatmap by Class

![Mean heatmap](results/gradcam/mean_heatmap_per_class.png)

#### Grad-CAM++ vs Ground Truth (DDR)

![Grad-CAM vs ground truth](results/gradcam_validation/gradcam_vs_ground_truth.png)

---

## Rule Extraction

### Approach

The extraction of interpretable rules proceeds in three steps: (1) feature extraction from the penultimate layer of the CNN (2,048 dimensions), (2) dimensionality reduction via PCA, (3) training of interpretable models on the CNN predictions (knowledge distillation).

### PCA: Dimensionality Reduction

| Parameter          | Value |
| ------------------ | ----- |
| Original features  | 2,048 |
| PCA components     | 12    |
| Explained variance | 95.2% |

![PCA variance](results/rule_extraction/pca_variance.png)

### Method A: Decision Tree Distillation

The Decision Tree is trained to replicate the CNN predictions (rather than the original labels), maximising fidelity.

#### Decision Tree Results (depth=12)

| Metric          | Validation | Test  |
| --------------- | ---------- | ----- |
| Fidelity        | 92.6%      | 88.2% |
| Accuracy (test) | -          | 82%   |
| F1 Macro (test) | -          | 0.62  |

| Complexity  | Value |
| ----------- | ----- |
| Total nodes | 1,909 |
| Leaves      | 955   |
| Max depth   | 12    |

#### Per-class Fidelity (Test)

| Class         | Fidelity |
| ------------- | -------- |
| No DR         | 93.9%    |
| Mild          | 83.3%    |
| Moderate      | 83.7%    |
| Severe        | 87.7%    |
| Proliferative | 84.8%    |

#### Ablation Study: DT Depth

| Depth | Nodes | Leaves | Fidelity (val) | Fidelity (test) |
| ----- | ----- | ------ | -------------- | --------------- |
| 4     | 31    | 16     | 87.6%          | 76.3%           |
| 6     | 127   | 64     | 90.9%          | 82.7%           |
| 8     | 433   | 217    | 92.0%          | 85.8%           |
| 10    | 1,091 | 546    | 92.7%          | 87.7%           |
| 12    | 1,909 | 955    | 92.6%          | 88.2%           |
| 16    | 2,763 | 1,382  | 92.2%          | 88.3%           |

Fidelity stabilises at depth 10-12, with diminishing returns for greater depths.

![DT ablation](results/rule_extraction/dt_ablation_depth.png)

### Method B: RIPPER One-vs-Rest

The RIPPER algorithm generates compact if-then rules for each class.

#### RIPPER Results

| Metric          | Test  |
| --------------- | ----- |
| Fidelity        | 63.0% |
| Accuracy        | 79%   |
| F1 Macro        | 0.51  |

| Complexity         | Value |
| ------------------ | ----- |
| Number of rules    | 48    |
| Total conditions   | 48    |
| Mean conditions    | 1.0   |

### Method C: LIME Local Explanations

LIME provides local explanations for individual predictions, complementing the global approaches of DT and RIPPER.

| Parameter               | Value         |
| ----------------------- | ------------- |
| Samples per explanation | 1,000         |
| Explained images        | 25            |
| Stability IoU (mean)    | 0.263 ± 0.125 |
| Runs for stability      | 5             |

![LIME vs Grad-CAM++](results/rule_extraction/lime_vs_gradcam.png)

### Method Comparison

| Metric           | Decision Tree (d=12) | RIPPER    |
| ---------------- | -------------------- | --------- |
| Fidelity (val)   | 92.6%                | -         |
| Fidelity (test)  | 88.2%                | 63.0%     |
| Accuracy (test)  | 82%                  | 79%       |
| F1 Macro (test)  | 0.62                 | 0.51      |
| Complexity       | 1,909 nodes          | 48 rules  |
| Interpretability | Medium               | High      |

The Decision Tree achieves the highest fidelity (92.6%), making it the ideal candidate for the post-hoc explanation of CNN predictions. RIPPER favours compactness (48 rules vs 955 leaves) at the cost of lower fidelity.

![Method comparison](results/rule_extraction/method_comparison.png)

---

## Hybrid System

### Approach

The hybrid system integrates the teacher CNN with the rule-based models through three operating strategies, each with a different trade-off between performance and interpretability.

### Strategy 1: Post-hoc Explanation

The CNN produces the prediction; the Decision Tree provides an explanation in rule form. Coverage indicates the share of CNN predictions faithfully replicated by the DT.

| Metric                      | Value |
| --------------------------- | ----- |
| Coverage (CNN-DT agreement) | 88.2% |
| Mean rule length            | 11.2  |

#### Per-class Agreement (Test)

| Class         | Agreement |
| ------------- | --------- |
| No DR         | 93.9%     |
| Mild          | 83.3%     |
| Moderate      | 83.7%     |
| Severe        | 87.7%     |
| Proliferative | 84.8%     |

![Post-hoc analysis](results/hybrid_system/posthoc_analysis.png)

### Strategy 2: Rule-Guided Classification

The Decision Tree classifies when its confidence exceeds a threshold; otherwise the CNN decides.

| Parameter         | Value |
| ----------------- | ----- |
| Optimal threshold | 0.575 |
| DT coverage       | 99.4% |
| Accuracy          | 82%   |
| F1 Macro          | 0.62  |

The high DT coverage (99.4%) indicates that the DT is almost always confident, leaving little room for the CNN to intervene.

![Rule-guided sweep](results/hybrid_system/rule_guided_sweep.png)

### Strategy 3: Weighted Ensemble

Weighted combination of the probabilities: `alpha * CNN_proba + (1-alpha) * DT_proba`.

| Parameter     | Value |
| ------------- | ----- |
| Optimal alpha | 1.0   |
| Accuracy      | 82%   |
| F1 Macro      | 0.63  |

The optimal value alpha=1.0 (pure CNN) indicates that the DT does not improve the predictive performance of the CNN. The value of the DT lies in interpretability, not in predictive improvement.

![Ensemble alpha sweep](results/hybrid_system/ensemble_alpha_sweep.png)

### Strategy Comparison

| Strategy                  | Accuracy | F1 Macro | QW Kappa | Interpretability      |
| ------------------------- | -------- | -------- | -------- | --------------------- |
| CNN-only (baseline)       | 83%      | 0.63     | 0.83     | None (black-box)      |
| DT-only (d=12)            | 82%      | 0.62     | 0.80     | High (955 leaves)     |
| RIPPER-only               | 79%      | 0.51     | 0.60     | Very high (48 rules)  |
| Post-hoc (CNN + DT expl.) | 83%      | 0.63     | 0.83     | 88.2% explained       |
| Rule-guided (t=0.575)     | 82%      | 0.62     | 0.80     | 99.4% explained       |
| Ensemble (alpha=1.0)      | 82%      | 0.63     | 0.82     | Partial               |

The **post-hoc explanation** strategy proves optimal: it preserves the CNN performance while providing rule-based explanations for 88.2% of the predictions.

![Strategy comparison](results/hybrid_system/strategy_comparison.png)

### Agreement Analysis

| Pair          | Agreement |
| ------------- | --------- |
| CNN - DT      | 88.2%     |
| CNN - RIPPER  | 63.0%     |
| DT - RIPPER   | 64.1%     |
| All unanimous | 59.0%     |

![Agreement analysis](results/hybrid_system/agreement_analysis.png)

---

## Evaluation Metrics

| Category        | Metric                             | Description                                       |
| --------------- | ---------------------------------- | ------------------------------------------------- |
| Teacher CNN     | Accuracy                           | Share of correct predictions                      |
| Teacher CNN     | Per-class F1-score                 | Class-wise balanced precision/recall              |
| Teacher CNN     | Cohen's Kappa (quadratic weighted) | Weighted agreement for ordinal classes            |
| Explainability  | Deletion AUC                       | Performance drop when removing relevant pixels    |
| Explainability  | Insertion AUC                      | Performance growth when adding relevant pixels    |
| Explainability  | IoU (Grad-CAM vs lesion masks)     | Overlap with ground-truth annotations             |
| Rule Extraction | Fidelity                           | Rule-CNN agreement (knowledge distillation)       |
| Rule Extraction | Rule complexity                    | Number of nodes/rules of the interpretable model  |
| Hybrid System   | Coverage                           | Share of explainable predictions                  |

## Project Structure

```
multi-method-xai-diabetic-retinopathy/
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Exploratory analysis of the datasets
│   ├── 02_preprocessing.ipynb          # Preprocessing pipeline
│   ├── 03_augmentation.ipynb           # Data augmentation and class balancing
│   ├── 04_analysis.ipynb               # CNN training
│   ├── 05_gradcam.ipynb                # Grad-CAM++ and quantitative validation
│   ├── 06_rule_extraction.ipynb        # Rule extraction (DT, RIPPER, LIME)
│   └── 07_hybrid_system.ipynb          # Hybrid system (3 strategies)
├── scripts/                            # Utility scripts
├── checkpoints/                        # Saved models
├── results/                            # Results and visualisations
│   ├── gradcam/                        # Saliency maps and comparisons
│   ├── gradcam_validation/             # Validation against DDR ground truth
│   ├── rule_extraction/                # Rule extraction metrics and plots
│   └── hybrid_system/                  # Hybrid system results
├── config.yaml                         # Centralised configuration
├── requirements.txt                    # Python dependencies
└── README.md
```

## Contact

**Antonio Colamartino**
Email: a.colamartino6@studenti.uniba.it
University of Bari Aldo Moro
Student ID: 778730
