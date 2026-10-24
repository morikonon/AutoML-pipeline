from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

def train_model(output_dir = "artifacts"):
	#1 Load dataset
	X, y = load_breast_cancer(return_X_y=True)

	#2 Split data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	#3 Create and train model
	model = make_pipeline(
		StandardScaler(),
		LogisticRegression(max_iter=1000)
	)
	model.fit(X_train, y_train)

	#4 Evaulate predictions
	predictions = model.predict(X_test)

	metrics = {
		"f1_score": f1_score(y_test, predictions),
		"accuracy_score": accuracy_score(y_test, predictions)
	}

	#5 Save trained model
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	joblib.dump(model, output_path / "model.joblib")

	return metrics

if __name__ == "__main__":
	metrics = train_model()

	for name, value in metrics.items():
		print(f"name: {value:.4f}")

