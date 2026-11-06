import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import predict as predict_cli
from inference import (
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    load_bundle,
    predict_file,
)
from pipeline import build_pipeline, prepare_features

FASE2 = Path(__file__).resolve().parents[1]
LABELS = {"0": "No", "1": "Yes"}
FEATURES = ["gender", "SeniorCitizen", "Partner", "tenure", "Contract", "MonthlyCharges"]


def make_labeled(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    churn = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "customerID": [f"C-{i:03d}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(1, 72, n),
            "Contract": np.where(churn == 1, "Month-to-month", rng.choice(["Month-to-month", "One year"], n)),
            "MonthlyCharges": np.round(rng.uniform(20, 110, n), 2),
            "TotalCharges": np.round(rng.uniform(20, 5000, n), 2),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    

@pytest.fixture()
def saved_model(tmp_path):
    labeled = make_labeled()
    features = prepare_features(labeled.drop(columns=["Churn"]))[FEATURES]
    target = (labeled["Churn"] == "Yes").astype(int)
    model = build_pipeline(FEATURES, seed=42).fit(features, target)
    path = tmp_path / "models" / "v1" / "modelo.joblib"
    path.parent.mkdir(parents=True)
    joblib.dump(
        {"pipeline": model, "feature_columns": FEATURES, "target": "Churn", "labels": LABELS},
        path,
    )
    return {
        "path": path,
        "model": model,
        "features": features,
        "unlabeled": labeled.drop(columns=["Churn"]),
        "tmp": tmp_path,
    }
    

def write_input(saved, frame: pd.DataFrame, name: str = "entrada.csv") -> Path:
    path = saved["tmp"] / name
    frame.to_csv(path, index=False)
    return path


def run_cli(saved, input_path: Path, capsys, model=None):
    output = saved["tmp"] / "salida" / "pred.csv"
    code = predict_cli.main(
        ["--model", str(model or saved["path"]), "--input", str(input_path), "--output", str(output)]
    )
    return code, output, capsys.readouterr()


def test_one_prediction_per_row_in_original_order(saved_model, capsys):
    unlabeled = saved_model["unlabeled"]
    code, output, _ = run_cli(saved_model, write_input(saved_model, unlabeled), capsys)
    result = pd.read_csv(output)
    assert code == 0
    assert len(result) == len(unlabeled)
    assert result["customerID"].tolist() == unlabeled["customerID"].tolist()  # orden conservado
    assert set(result[PREDICTION_COLUMN]) <= {0, 1}
    assert result[LABEL_COLUMN].tolist() == [LABELS[str(p)] for p in result[PREDICTION_COLUMN]]
    assert result[PROBABILITY_COLUMN].between(0, 1).all()


def test_loaded_artifact_gives_same_results_as_model_before_saving(saved_model, tmp_path):
    expected = saved_model["model"].predict_proba(saved_model["features"])[:, 1]
    output = tmp_path / "pred.csv"
    predict_file(saved_model["path"], write_input(saved_model, saved_model["unlabeled"]), output)
    result = pd.read_csv(output)
    np.testing.assert_allclose(result[PROBABILITY_COLUMN], expected, atol=1e-6)
    expected_classes = saved_model["model"].predict(saved_model["features"])
    assert result[PREDICTION_COLUMN].tolist() == expected_classes.tolist()
    
    
def test_empty_cells_are_imputed_with_the_training_median(saved_model, tmp_path):
    frame = saved_model["unlabeled"].copy()
    frame["tenure"] = frame["tenure"].astype(float)
    frame.loc[0, "tenure"] = np.nan
    frame.loc[1, "Contract"] = np.nan
    output = tmp_path / "pred.csv"
    summary = predict_file(saved_model["path"], write_input(saved_model, frame), output)

    result = pd.read_csv(output)
    assert len(result) == len(frame)
    assert result[PROBABILITY_COLUMN].notna().all()
    assert summary["nulos_imputados"] == {"tenure": 1, "Contract": 1}

    median = float(saved_model["features"]["tenure"].median())
    filled = saved_model["unlabeled"].copy()
    filled["tenure"] = filled["tenure"].astype(float)
    filled.loc[0, "tenure"] = median
    filled.loc[1, "Contract"] = saved_model["features"]["Contract"].mode().iloc[0]
    reference = tmp_path / "ref.csv"
    predict_file(saved_model["path"], write_input(saved_model, filled, "relleno.csv"), reference)
    np.testing.assert_allclose(
        result[PROBABILITY_COLUMN].iloc[:2], pd.read_csv(reference)[PROBABILITY_COLUMN].iloc[:2], atol=1e-6
    )


def test_missing_required_column_fails_clearly_and_creates_no_output(saved_model, capsys):
    frame = saved_model["unlabeled"].drop(columns=["tenure"])
    code, output, captured = run_cli(saved_model, write_input(saved_model, frame), capsys)
    assert code == 1
    assert "Faltan columnas requeridas: tenure" in captured.err
    assert not output.exists()


def test_incompatible_type_fails_clearly_and_creates_no_output(saved_model, capsys):
    frame = saved_model["unlabeled"].copy()
    frame["tenure"] = frame["tenure"].astype(object)
    frame.loc[3, "tenure"] = "abc"
    code, output, captured = run_cli(saved_model, write_input(saved_model, frame), capsys)
    assert code == 1
    assert "tenure" in captured.err and "no compatibles" in captured.err
    assert not output.exists()
    
    
def test_missing_model_exits_nonzero_from_the_command_line(saved_model):
    output = saved_model["tmp"] / "pred.csv"
    completed = subprocess.run(
        [
            sys.executable, "predict.py",
            "--model", str(saved_model["tmp"] / "models" / "v9" / "modelo.joblib"),
            "--input", str(write_input(saved_model, saved_model["unlabeled"])),
            "--output", str(output),
        ],
        cwd=FASE2, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "No se encontró el modelo" in completed.stderr
    assert "v1" in completed.stderr
    assert not output.exists()


def test_column_order_and_extra_columns_do_not_change_the_result(saved_model, tmp_path):
    baseline = tmp_path / "base.csv"
    predict_file(saved_model["path"], write_input(saved_model, saved_model["unlabeled"]), baseline)

    altered = saved_model["unlabeled"][list(reversed(saved_model["unlabeled"].columns))].copy()
    altered["ColumnaExtra"] = "x"
    other = tmp_path / "otro.csv"
    predict_file(saved_model["path"], write_input(saved_model, altered, "alterada.csv"), other)
    np.testing.assert_allclose(
        pd.read_csv(baseline)[PROBABILITY_COLUMN], pd.read_csv(other)[PROBABILITY_COLUMN], atol=1e-9
    )


def test_unseen_category_is_encoded_without_failing_and_is_reported(saved_model, tmp_path):
    frame = saved_model["unlabeled"].copy()
    frame.loc[0, "Contract"] = "Two year"
    summary = predict_file(saved_model["path"], write_input(saved_model, frame), tmp_path / "pred.csv")
    assert summary["categorias_no_vistas"] == {"Contract": ["Two year"]}
    assert len(pd.read_csv(tmp_path / "pred.csv")) == len(frame)


def test_version_folder_can_be_used_instead_of_the_file(saved_model, tmp_path):
    by_file, by_folder = tmp_path / "a.csv", tmp_path / "b.csv"
    entrada = write_input(saved_model, saved_model["unlabeled"])
    predict_file(saved_model["path"], entrada, by_file)
    predict_file(saved_model["path"].parent, entrada, by_folder)
    assert by_file.read_text() == by_folder.read_text()


def test_file_with_wrong_format_is_rejected(tmp_path):
    fake = tmp_path / "modelo.joblib"
    joblib.dump({"otra": "cosa"}, fake)
    with pytest.raises(ValueError, match="formato esperado"):
        load_bundle(fake)


def test_output_cannot_overwrite_the_input(saved_model):
    entrada = write_input(saved_model, saved_model["unlabeled"])
    with pytest.raises(ValueError, match="mismo que el de entrada"):
        predict_file(saved_model["path"], entrada, entrada)


def test_help_lists_the_arguments():
    completed = subprocess.run(
        [sys.executable, "predict.py", "--help"], cwd=FASE2, capture_output=True, text=True
    )
    assert completed.returncode == 0
    for argument in ("--model", "--input", "--output"):
        assert argument in completed.stdout
        

def test_semicolon_separated_csv_gets_a_helpful_message(saved_model, tmp_path):
    path = saved_model["tmp"] / "punto_y_coma.csv"
    saved_model["unlabeled"].to_csv(path, index=False, sep=";")
    with pytest.raises(ValueError, match="punto y coma"):
        predict_file(saved_model["path"], path, tmp_path / "pred.csv")
    assert not (tmp_path / "pred.csv").exists()