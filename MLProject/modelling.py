from pathlib import Path
import tempfile
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.utils import estimator_html_repr
from sklearn import set_config

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def load_data(base_dir: Path):
    for p in base_dir.rglob("X_train.csv"):
        d = p.parent
        return (
            pd.read_csv(d / "X_train.csv"),
            pd.read_csv(d / "X_test.csv"),
            pd.read_csv(d / "y_train.csv").squeeze(),
            pd.read_csv(d / "y_test.csv").squeeze(),
        )
    raise FileNotFoundError("Dataset preprocessing tidak ditemukan")

def save_plot(fig, filename, tmp_dir):
    path = tmp_dir / filename
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    mlflow.log_artifact(str(path))

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("rf_baseline_experiment")

    # Load Data
    base_dir = Path(__file__).resolve().parent
    X_train, X_test, y_train, y_test = load_data(base_dir)

    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
    }
    
    model = RandomForestClassifier(**params, n_jobs=-1)

    with mlflow.start_run(run_name="rf_baseline_manual"):
        # 1. Log Params & Train
        mlflow.log_params(params)
        model.fit(X_train, y_train)

        # 2. Predict & Metrics
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        # 3. Artifacts 
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # File Teks & JSON
            (tmp / "classification_report.txt").write_text(classification_report(y_test, y_pred))
            (tmp / "metric_info.json").write_text(json.dumps(metrics, indent=4))
            
            # Estimator HTML
            set_config(display="diagram")
            (tmp / "estimator.html").write_text(estimator_html_repr(model))

            # Confusion Matrix Plot
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set(xlabel="Predicted", ylabel="Actual")
            
            # Logging Artefak ke Root
            mlflow.log_artifact(str(tmp / "classification_report.txt"))
            mlflow.log_artifact(str(tmp / "metric_info.json"))
            mlflow.log_artifact(str(tmp / "estimator.html"))
            save_plot(fig, "training_confusion_matrix.png", tmp)

        # 4. Log Model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3)
        )

    print("Manual MLflow logging selesai")

if __name__ == "__main__":
    main()