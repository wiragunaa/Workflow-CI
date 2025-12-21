from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
import pickle
import shutil


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
	mlflow.set_tracking_uri("file:./mlruns")
	Path("./mlruns").mkdir(parents=True, exist_ok=True)
	mlflow.autolog()

	base_dir = Path(__file__).resolve().parent
	X_train, X_test, y_train, y_test = load_data(base_dir)

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
  
		try:
			mlflow.sklearn.log_model(model, "model")
		except Exception:
			pass

		try:
			run_id = mlflow.active_run().info.run_id
			run_id_path = base_dir / "run_id.txt"
			run_id_path.write_text(run_id)
			artifacts_run_id = artifacts_dir / "run_id.txt"
			artifacts_run_id.write_text(run_id)
			mlflow.log_artifact(str(artifacts_run_id))
		except Exception:
			pass

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

		# Also save a final set of artifacts similar to the tuning script's "best model" output
		final_artifacts = base_dir / "tuning_final"
		final_artifacts.mkdir(exist_ok=True)
		conda_final = final_artifacts / "conda.yaml"
		conda_final.write_text(conda_content)
		mlflow.log_artifact(str(conda_final), artifact_path="model_tuned")

		# attempt to log the sklearn model with conda env
		try:
			mlflow.sklearn.log_model(model, artifact_path="model_tuned", conda_env=str(conda_final))
		except Exception:
			try:
				mlflow.sklearn.log_model(model, artifact_path="model_tuned")
			except Exception:
				pass

		# save top feature importances plot
		try:
			fi_path = final_artifacts / "best_feature_importances.png"
			fi_series = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)[:50]
			plt.figure(figsize=(8, 6))
			sns.barplot(x=fi_series.values, y=fi_series.index)
			plt.xlabel("Importance")
			plt.tight_layout()
			plt.savefig(fi_path)
			plt.close()
			mlflow.log_artifact(str(fi_path), artifact_path="model_tuned")
		except Exception:
			fi_series = None

		# predictions, probabilities and metrics
		y_pred_best = y_pred
		try:
			if hasattr(model, "predict_proba"):
				y_proba_best = model.predict_proba(X_test)[:, 1]
			else:
				y_proba_best = model.decision_function(X_test)
		except Exception:
			y_proba_best = np.zeros(len(X_test))

		metrics = {
			"test_accuracy": float(acc),
			"test_precision": float(prec),
			"test_recall": float(rec),
			"test_f1": float(f1),
			"test_roc_auc": float(roc_auc) if len(np.unique(y_test)) > 1 else 0.0,
		}
		metrics_path = final_artifacts / "metrics.json"
		metrics_path.write_text(json.dumps(metrics, indent=2))
		mlflow.log_artifact(str(metrics_path), artifact_path="model_tuned")

		for k, v in metrics.items():
			mlflow.log_metric(f"best_{k}", float(v))

		preds_path = final_artifacts / "predictions.csv"
		preds_df = pd.DataFrame({
			"y_true": np.asarray(y_test).ravel(),
			"y_pred": np.asarray(y_pred_best).ravel(),
			"y_proba": np.asarray(y_proba_best).ravel(),
		})
		preds_df.to_csv(preds_path, index=False)
		mlflow.log_artifact(str(preds_path), artifact_path="model_tuned")

		# save model pickle
		model_pkl = final_artifacts / "model.pkl"
		with open(model_pkl, "wb") as f:
			pickle.dump(model, f)
		mlflow.log_artifact(str(model_pkl), artifact_path="model_tuned")

		# copy requirements if present
		repo_req = base_dir / "requirements.txt"
		if repo_req.exists():
			dest_req = final_artifacts / "requirements.txt"
			shutil.copy(repo_req, dest_req)
			mlflow.log_artifact(str(dest_req), artifact_path="model_tuned")

		# write a simple estimator summary HTML
		try:
			params_html = "\n".join([f"<li>{k}: {v}</li>" for k, v in params.items()])
		except Exception:
			params_html = ""
		try:
			fi_html = "\n".join([f"<li>{feat}: {val:.6f}</li>" for feat, val in (fi_series.items() if fi_series is not None else [])])
		except Exception:
			fi_html = ""
		estimator_content = f"""
<!doctype html>
<html>
  <head><meta charset=\"utf-8\"><title>Estimator Summary</title></head>
  <body>
    <h1>Estimator Summary</h1>
    <h2>Model Params</h2>
    <ul>
    {params_html}
    </ul>
    <h2>Top Feature Importances</h2>
    <ul>
    {fi_html}
    </ul>
  </body>
</html>
"""
		estimator_html = final_artifacts / "estimator.html"
		estimator_html.write_text(estimator_content)
		mlflow.log_artifact(str(estimator_html), artifact_path="model_tuned")

		# training confusion matrix
		try:
			y_train_pred = model.predict(X_train)
			train_cm_path = final_artifacts / "training_confusion_matrix.png"
			plot_and_save_confusion_matrix(y_train, y_train_pred, train_cm_path)
			mlflow.log_artifact(str(train_cm_path), artifact_path="model_tuned")
		except Exception:
			pass

		# additional info
		metric_info = {
			"model_type": type(model).__name__,
			"params": params,
			"metrics": metrics,
		}
		metric_info_path = final_artifacts / "metric_info.json"
		metric_info_path.write_text(json.dumps(metric_info, indent=2))
		mlflow.log_artifact(str(metric_info_path), artifact_path="model_tuned")

		print("Run completed. Logged params, metrics, model, and artifacts to MLflow.")


if __name__ == "__main__":
	main()

