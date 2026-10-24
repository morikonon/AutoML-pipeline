from pathlib import Path
import joblib
from src.train import train_model

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