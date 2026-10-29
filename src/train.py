from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

def train_model(output_dir = "artifacts", C = 1.0):
	# 1 Load dataset
	X, y = load_breast_cancer(return_X_y=True)

	# 2 Split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	# 3 Create and train model
	model = make_pipeline(
		StandardScaler(),
		LogisticRegression(C=C, max_iter=1000)
	)

	boosting = GradientBoostingClassifier()

	# 4 Set MLflow experiment
	mlflow.set_experiment("breast-cancer-classification")

	with mlflow.start_run() as run:
		mlflow.log_param("model_type", "LogisticRegression")
		mlflow.log_param("C", C)
		mlflow.log_param("max_iter", 1000)
		mlflow.log_param("test_size", 0.2)
		mlflow.log_param("random_state", 42)

		mlflow.log_param("model_type", "GradientBoosting")

		# 5 Train model
		model.fit(X_train, y_train)
		boosting.fit(X_train, y_train)

		predictions = model.predict(X_test)
		predictions2 = boosting.predict(X_test)

		# 6 Compute metrics
		metrics = {
			"f1_score": f1_score(y_test, predictions),
			"accuracy_score": accuracy_score(y_test, predictions)

			"boosting_f1_score": f1_score(y_test, predictions2),
			"boosting_accuracy_score": accuracy_score(y_test, predictions2)
		}

		# 7 Log metrics
		mlflow.log_metrics(metrics)

		model_filename = (
			f"model_C_{C}_{run.info.run_id[:8]}.joblib"
			f"model_boosting_C_{C}_{run.info.run_id[:8]}.joblib"
		)

		# 8 Save trained model
		output_path = Path(output_dir)
		output_path.mkdir(parents=True, exist_ok=True)

		joblib.dump(model, output_path / model_filename)

		# 9 Log model artifact to MLflow
		mlflow.sklearn.log_model(
			sk_model=model,
			name="model",
		)

		mlflow.set_tag("local_model_file", model_filename)

	return metrics

if __name__ == "__main__":
	for C in [0.1, 1.0, 1.0]:
		metrics = train_model(C=C)

		print(f"C: {C}, {metrics}")

