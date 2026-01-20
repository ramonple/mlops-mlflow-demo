# MLOps Demo Using MLflow

This repository demonstrates how **MLflow fits into an MLOps workflow**, providing experiment tracking, model versioning, and reproducibility to support governance-ready, end-to-end machine learning pipelines.

MLflow is used here as the **MLOps layer** that bridges model development and production-style workflows.

---

## 🚀 What This Project Shows

- Experiment tracking (parameters, metrics, artifacts)  
- Reproducible training runs  
- Model logging and versioning  
- A minimal, production-style ML pipeline  
- Local MLflow server usage (macOS / Windows notes included)

---

## 🧠 Why MLflow in MLOps?

MLflow fits into the MLOps layer of the machine learning lifecycle, providing experiment tracking, model versioning, and reproducibility to support governance-ready, end-to-end ML pipelines.

In practice, this enables:
- Consistent experiment management  
- Model traceability across versions  
- Reproducible training and evaluation  
- Cleaner handoff from development to deployment

---

## 📂 Contents

- `mlflow_all_in_one_Mac.py`  
  End-to-end example script covering:
  - data preparation  
  - model training  
  - metric logging  
  - artifact logging  
  - model registration  

- `MLflow (Windows) — Clean Restart.docx`  
- `MLflow (macOS) — Clean Restart .docx`  
  Notes on restarting a local MLflow tracking server cleanly.

---

## ⚙️ How To Run

1. Install dependencies:
   ```bash
   pip install mlflow scikit-learn pandas numpy
