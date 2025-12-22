from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	roc_auc_score,
	classification_report,
	confusion_matrix,
)

import mlflow
import mlflow.sklearn
import dagshub
import pickle
import shutil
from mlflow.models.signature import infer_signature
import uuid
from datetime import datetime
import platform
import sklearn as _sklearn


def load_data(base_dir: Path):
	# try several likely locations for the preprocessed files
	candidates = [
		base_dir / "online_shoppers_intention_preprocessing",
		base_dir / "online_shoppers_intention_preprocessing" / "online_shoppers_intention_preprocessing",
	]
	data_dir = None
	for c in candidates:
		if (c / "X_train.csv").exists():
			data_dir = c
			break
	if data_dir is None:
		# fallback: search recursively under base_dir (one level deep)
		for child in base_dir.iterdir():
			p = child / "X_train.csv"
			if p.exists():
				data_dir = child
				break
	if data_dir is None:
		raise FileNotFoundError(f"Could not find X_train.csv under {base_dir}")

	X_train = pd.read_csv(data_dir / "X_train.csv")
	X_test = pd.read_csv(data_dir / "X_test.csv")
	y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
	y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
	return X_train, X_test, y_train, y_test


def save_classification_report(y_true, y_pred, out_path: Path):
	report = classification_report(y_true, y_pred)
	out_path.write_text(report)


def plot_and_save_confusion_matrix(y_true, y_pred, out_path: Path):
	cm = confusion_matrix(y_true, y_pred)
	plt.figure(figsize=(6, 4))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def plot_and_save_roc(y_true, y_proba, out_path: Path):
	from sklearn.metrics import roc_curve, auc

	fpr, tpr, _ = roc_curve(y_true, y_proba)
	roc_auc = auc(fpr, tpr)
	plt.figure(figsize=(6, 4))
	plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
	plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def main():
	mlflow.set_tracking_uri("http://127.0.0.1:5000")
	Path("./mlruns").mkdir(parents=True, exist_ok=True)
	mlflow.autolog()

	base_dir = Path(__file__).resolve().parent
	X_train, X_test, y_train, y_test = load_data(base_dir)

	# define model and parameters to log
	model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
	params = {
		"n_estimators": 200,
		"max_depth": 10,
		"min_samples_split": 5,
		"random_state": 42,
	}

	mlflow.set_experiment("rf_baseline_experiment")

	with mlflow.start_run(run_name="rf_baseline"):
		for k, v in params.items():
			mlflow.log_param(k, v)

		# fit
		model.fit(X_train, y_train)

		y_pred = model.predict(X_test)
		if hasattr(model, "predict_proba"):
			y_proba = model.predict_proba(X_test)[:, 1]
		else:
			y_proba = model.decision_function(X_test)

		# compute metrics
		acc = accuracy_score(y_test, y_pred)
		prec = precision_score(y_test, y_pred, zero_division=0)
		rec = recall_score(y_test, y_pred, zero_division=0)
		f1 = f1_score(y_test, y_pred, zero_division=0)
		roc_auc = roc_auc_score(y_test, y_proba)
			
		# Metrics are logged automatically by mlflow.autolog()

		# save artifacts: classification report, confusion matrix, ROC plot
		artifacts_dir = base_dir / "mlflow_artifacts"
		artifacts_dir.mkdir(exist_ok=True)

		cls_report_path = artifacts_dir / "classification_report.txt"
		save_classification_report(y_test, y_pred, cls_report_path)
		mlflow.log_artifact(str(cls_report_path))

		cm_path = artifacts_dir / "confusion_matrix.png"
		plot_and_save_confusion_matrix(y_test, y_pred, cm_path)
		mlflow.log_artifact(str(cm_path))

		roc_path = artifacts_dir / "roc_curve.png"
		plot_and_save_roc(y_test, y_proba, roc_path)
		mlflow.log_artifact(str(roc_path))

		# save model as pickle (additional artifact)
		model_pkl = artifacts_dir / "model.pkl"
		with open(model_pkl, "wb") as f:
			pickle.dump(model, f)
		mlflow.log_artifact(str(model_pkl))

		# compute signature and input_example; resave model via mlflow with explicit signature
		try:
			input_example = X_train.head(3).copy()
			signature = infer_signature(X_train, model.predict(X_train))
		except Exception:
			input_example = None
			signature = None

		# log model again (autolog still active) with signature and input_example
		try:
			mlflow.sklearn.log_model(
				model,
				artifact_path="model_tuned",
				conda_env=str(conda_path),
				signature=signature,
				input_example=input_example,
			)
		except Exception:
			pass

		# create MLmodel descriptor and log it alongside artifacts
		try:
			active_run = mlflow.active_run()
			run_id = active_run.info.run_id if active_run is not None else None
		except Exception:
			run_id = None

		model_tuned_dir = artifacts_dir / "model_tuned"
		model_tuned_dir.mkdir(exist_ok=True)

		try:
			model_size = int(os.path.getsize(model_pkl)) if model_pkl.exists() else 0
		except Exception:
			model_size = 0

		mlmodel_content = f"""artifact_path: model_tuned
flavors:
  python_function:
    env: conda.yaml
    python_env: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: {platform.python_version()}
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: {_sklearn.__version__}
mlflow_version: {mlflow.__version__}
model_size_bytes: {model_size}
model_uuid: {uuid.uuid4().hex}
run_id: {run_id}
utc_time_created: '{datetime.utcnow().isoformat(sep=" ")}'
"""

		mlmodel_path = model_tuned_dir / "MLmodel"
		mlmodel_path.write_text(mlmodel_content)
		mlflow.log_artifact(str(mlmodel_path), artifact_path="model_tuned")

		repo_req = base_dir / "requirements.txt"
		if repo_req.exists():
			dest_req = artifacts_dir / "requirements.txt"
			shutil.copy(repo_req, dest_req)
			mlflow.log_artifact(str(dest_req))
		
		conda_path = artifacts_dir / "conda.yaml"
		conda_content = (
			"name: mlflow-env\nchannels:\n  - defaults\ndependencies:\n  - python=3.10\n  - pip\n  - pip:\n    - -r requirements.txt\n"
		)
		conda_path.write_text(conda_content)
		mlflow.log_artifact(str(conda_path))

		python_env = artifacts_dir / "python_env.yaml"
		python_env.write_text(conda_content)
		mlflow.log_artifact(str(python_env))

		# Model is logged automatically by mlflow.autolog()

		print("Run completed. Logged params, metrics, model, and artifacts to MLflow.")


if __name__ == "__main__":
	main()

