# Multi-Method Rule Extraction from Deep Learning for Interpretable Diabetic Retinopathy Grading

Progetto di Tesi di Laurea

**Autore:** Antonio Colamartino
**Email:** a.colamartino6@studenti.uniba.it
**Matricola:** 778730
**Universita:** Universita degli Studi di Bari Aldo Moro (UniBA)

---

## Descrizione

Questo progetto propone un sistema ibrido per la classificazione della retinopatia diabetica (Diabetic Retinopathy, DR) che combina le elevate prestazioni delle reti neurali profonde con l'interpretabilita delle regole estratte. L'obiettivo principale e sviluppare un sistema di supporto alle decisioni cliniche che sia sia accurato che comprensibile per i medici.

La retinopatia diabetica e una delle principali cause di cecità nel mondo e la sua diagnosi precoce e fondamentale per prevenire la perdita della vista. I modelli di deep learning hanno dimostrato eccellenti capacita predittive, ma la loro natura "black-box" limita l'adozione in ambito clinico dove la trasparenza delle decisioni e cruciale.

## Obiettivi

1. **Training di un modello CNN ad alte prestazioni** per la classificazione multi-classe della DR
2. **Analisi dell'explainability** tramite tecniche di saliency mapping e validazione quantitativa
3. **Estrazione di regole interpretabili** attraverso tre metodologie differenti
4. **Sviluppo di un sistema ibrido** che integri CNN e regole in modalita operative diverse

## Architettura del Progetto

### Fase 1 - Teacher CNN Training

| Componente | Specifica |
|------------|-----------|
| Architettura | EfficientNet-B5 (30M parametri) |
| Task | Multi-class classification (5 classi DR) |
| Classi | No DR, Mild, Moderate, Severe, Proliferative |
| Loss Function | Focal Loss + Label Smoothing (epsilon=0.1) |

### Fase 2 - Explainability Analysis

- **Saliency Maps:** Grad-CAM++ per la visualizzazione delle regioni rilevanti
- **Validazione Quantitativa:** Confronto tra Grad-CAM e ground-truth lesions (DDR dataset)
- **Clustering:** K-means su activation patterns per identificazione prototypes

### Fase 3 - Rule Extraction

Tre metodi comparati per l'estrazione di regole interpretabili:

| Metodo | Output |
|--------|--------|
| A - Decision Tree Distillation | Interpretable tree con if-then rules |
| B - LIME-based Rule Induction | Unordered rule set (CN2 Algorithm) |
| C - Activation Pattern Rules | Ordered rule list (RIPPER Algorithm) |

### Fase 4 - Sistema Ibrido

| Modalita | Strategia |
|----------|-----------|
| Post-hoc Explanation | CNN prediction con post-hoc rule explanation |
| Rule-guided Prediction | IF rule confidence > 0.8 THEN use rule ELSE use CNN |
| Ensemble Voting | CNN + all 3 rule-based predictors |

## Dataset

| Dataset | Immagini | Utilizzo |
|---------|----------|----------|
| APTOS 2019 | 3,662 | Training CNN |
| Kaggle EyePACS 2015 | 35,126 | Training CNN (merge per ~38,788 totali) |
| Messidor-2 | 1,744 | External validation (Testing) |
| DDR (lesion annotations) | 757 | XAI Validation (pixel-level masks) |

### Struttura della Cartella Data

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

## Metriche di Valutazione

| Categoria | Metrica |
|-----------|---------|
| CNN Teacher | Accuracy |
| CNN Teacher | Per-class F1-score |
| CNN Teacher | Cohen's Kappa |
| Rule Extraction | Rule Fidelity |
| Rule Extraction | Rule Complexity |
| Rule Extraction | Rule Coverage |
| XAI Validation | IoU (Grad-CAM vs lesion masks) |


## Contatti

**Antonio Colamartino**
Email: a.colamartino6@studenti.uniba.it
Universita degli Studi di Bari Aldo Moro
Matricola: 778730
