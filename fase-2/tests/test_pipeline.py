import json

import joblib
import pandas as pd
import pytest

from pipeline import train_and_save


def dataset() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "customerID": f"id-{index}",
                "gender": "Female" if index % 2 else "Male",
                "SeniorCitizen": index % 2,
                "Partner": "Yes" if index % 3 else "No",
                "Dependents": "No",
                "tenure": index + 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "Contract": "Month-to-month" if index % 2 else "One year",
                "MonthlyCharges": 30.0 + index,
                "TotalCharges": 30.0 + index,
                "Churn": "Yes" if index % 2 else "No",
            }
        )
    return pd.DataFrame(rows)


def train(tmp_path):
    data = tmp_path / "data.csv"
    dataset().to_csv(data, index=False)
    return train_and_save(data, "Churn", tmp_path / "models", "v1")


def test_training_creates_loadable_artifact_metrics_and_metadata(tmp_path):
    version = train(tmp_path)
    assert (version / "modelo.joblib").is_file()
    assert json.loads((version / "metrics.json").read_text())["model"]["f1"] >= 0
    metadata = json.loads((version / "metadata.json").read_text())
    assert metadata["version"] == "v1"
    assert metadata["parameters"]["model"] == {
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 100,
    }


def test_invalid_target_is_rejected(tmp_path):
    data = tmp_path / "invalid.csv"
    frame = dataset()
    frame.loc[0, "Churn"] = "Unknown"
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError, match="solo puede contener"):
        train_and_save(data, "Churn", tmp_path / "models", "v1")


def test_missing_value_is_imputed_during_training(tmp_path):
    data = tmp_path / "missing.csv"
    frame = dataset()
    frame.loc[0, "tenure"] = None
    frame.to_csv(data, index=False)
    version = train_and_save(data, "Churn", tmp_path / "models", "v1")
    assert (version / "modelo.joblib").is_file()


def test_missing_required_column_is_rejected(tmp_path):
    data = tmp_path / "invalid.csv"
    dataset().drop(columns=["tenure"]).to_csv(data, index=False)
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        train_and_save(data, "Churn", tmp_path / "models", "v1")


def test_saved_artifact_can_be_loaded_and_predict(tmp_path):
    version = train(tmp_path)
    bundle = joblib.load(version / "modelo.joblib")
    frame = dataset()
    features = frame[bundle["feature_columns"]].copy()
    features["SeniorCitizen"] = features["SeniorCitizen"].map({0: "No", 1: "Yes"})
    predictions = bundle["pipeline"].predict(features)
    assert len(predictions) == len(frame)
    assert set(predictions).issubset({0, 1})


def test_versions_are_preserved_and_duplicate_is_rejected(tmp_path):
    data = tmp_path / "data.csv"
    dataset().to_csv(data, index=False)
    models = tmp_path / "models"
    train_and_save(data, "Churn", models, "v1")
    train_and_save(data, "Churn", models, "v2")
    assert (models / "v1" / "modelo.joblib").is_file()
    assert (models / "v2" / "modelo.joblib").is_file()
    with pytest.raises(FileExistsError):
        train_and_save(data, "Churn", models, "v1")
