from pathlib import Path
import joblib
from src.train import train_model
import mlflow
import mlflow.sklearn

def test_training_model(tmp_path):
	metrics = train_model(output_dir=tmp_path)

	assert 0.0 <= metrics["accuracy_score"] <= 1.0
	assert 0.0 <= metrics["f1_score"] <= 1.0

	model_path = Path(tmp_path) / "model.joblib"

	assert model_path.exists()

	# model can be loaded and predict

	model = joblib.load(model_path)
	predictions = model.predict([[10] * 30])

	assert len(predictions) == 1
	assert predictions[0] in (0, 1)

def mlflow_test(tmp_path):
	mlflow.set_experiment("test-experiment")

	train_model(output_dir=tmp_path / "artifacts", C=0.1)

	experiment = mlflow.get_experiment_by_name("test-experiment")
	runs = mlflow.search_runs(
		experiment_ids=[experiment.experiment_id]
	)

	assert len(runs) >= 1
	latest_run = runs.iloc[0]

	assert latest_run["params.C"] == 0.1
	assert latest_run["metrics.accuracy_score"] <= 1.0
	assert latest_run["metrics.f1_score"] <= 1.0

