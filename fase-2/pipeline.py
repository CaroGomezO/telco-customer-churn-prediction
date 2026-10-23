"""Reusable training logic for the churn model."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = ["tenure", "MonthlyCharges"]
REQUIRED_FEATURE_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges"]
EXCLUDED_COLUMNS = ["customerID", "TotalCharges"]
MODEL_FILENAME = "modelo.joblib"
MODEL_PARAMETERS = {
    "learning_rate": 0.1,
    "max_depth": 3,
    "n_estimators": 100,
}


def read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe el archivo CSV: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"No se pudo leer el CSV '{csv_path}': {exc}") from exc
    if frame.empty:
        raise ValueError(f"El CSV '{csv_path}' no contiene filas.")
    return frame


def feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    if target not in frame.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el CSV.")
    missing = [column for column in REQUIRED_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")
    columns = [column for column in frame.columns if column not in [target, *EXCLUDED_COLUMNS]]
    if not columns:
        raise ValueError("No quedaron variables de entrada después de excluir identificador y TotalCharges.")
    return columns


def validate_features(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")
    for column in NUMERIC_COLUMNS:
        if column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & values.isna()
            if invalid.any():
                raise ValueError(f"La columna numérica '{column}' contiene valores no compatibles.")


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "SeniorCitizen" in prepared:
        senior_citizen = prepared["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"})
        invalid = prepared["SeniorCitizen"].notna() & senior_citizen.isna()
        if invalid.any():
            raise ValueError("La columna 'SeniorCitizen' solo puede contener 0/1 o Yes/No.")
        prepared["SeniorCitizen"] = senior_citizen
    return prepared


def build_pipeline(columns: list[str], seed: int) -> Pipeline:
    numeric = [column for column in NUMERIC_COLUMNS if column in columns]
    categorical = [column for column in columns if column not in numeric]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessor),
            (
                "model",
                GradientBoostingClassifier(random_state=seed, **MODEL_PARAMETERS),
            ),
        ]
    )


def _metrics(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    values = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    if y_test.nunique() == 2:
        values["roc_auc"] = roc_auc_score(y_test, probabilities)
    return {key: round(float(value), 6) for key, value in values.items()}


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "no-disponible"


def train_and_save(
    data_path: str | Path,
    target: str,
    output_dir: str | Path,
    version: str,
    seed: int = 42,
    test_size: float = 0.2,
) -> Path:
    if not version or Path(version).name != version:
        raise ValueError("La versión debe ser un nombre simple, por ejemplo 'v1'.")
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")
    version_dir = Path(output_dir) / version
    if version_dir.exists():
        raise FileExistsError(f"La versión '{version}' ya existe; no se sobrescribirá.")
    print(f"[1/5] Leyendo datos de {data_path}...")
    frame = prepare_features(read_csv(data_path))
    print("[2/5] Validando columnas y variable objetivo...")
    columns = feature_columns(frame, target)
    validate_features(frame, columns)
    labels = frame[target].astype(str).str.strip().map({"No": 0, "Yes": 1})
    if labels.isna().any():
        raise ValueError("La variable objetivo solo puede contener las etiquetas 'Yes' y 'No'.")
    x = frame[columns].copy()
    y = labels.astype(int)
    print("[3/5] Separando entrenamiento y prueba...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    print("[4/5] Ajustando preprocesamiento y modelo...")
    model = build_pipeline(columns, seed)
    model.fit(x_train, y_train)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)

    print("[5/5] Guardando artefactos...")
    version_dir.mkdir(parents=True)
    bundle = {
        "pipeline": model,
        "feature_columns": columns,
        "target": target,
        "labels": {"0": "No", "1": "Yes"},
    }
    joblib.dump(bundle, version_dir / MODEL_FILENAME)
    metrics = {"model": _metrics(model, x_test, y_test), "baseline": _metrics(baseline, x_test, y_test)}
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "problem_type": "binary_classification",
        "target": target,
        "algorithm": "GradientBoostingClassifier",
        "input_columns": columns,
        "expected_types": {column: str(frame[column].dtype) for column in columns},
        "seed": seed,
        "parameters": {
            "test_size": test_size,
            "random_state": seed,
            "model": MODEL_PARAMETERS,
        },
        "data": {"source": str(data_path), "observations": len(frame), "train": len(x_train), "test": len(x_test)},
        "python": platform.python_version(),
        "requirements": "requirements.txt",
        "git_revision": _git_revision(),
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return version_dir
