from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline import MODEL_FILENAME, prepare_features, read_csv, validate_features

ID_COLUMN = "customerID"
PREDICTION_COLUMN = "prediccion"
LABEL_COLUMN = "etiqueta"
PROBABILITY_COLUMN = "probabilidad_churn"
REQUIRED_BUNDLE_KEYS = ("pipeline", "feature_columns", "labels")

def _available_versions(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(c.name for c in root.iterdir() if c.is_dir() and (c / MODEL_FILENAME).is_file())


def resolve_model_path(model: str | Path) -> Path:
    given = Path(model)
    path = given / MODEL_FILENAME if given.is_dir() else given
    if not path.is_file():
        versions, root = [], None
        for candidate_root in (given.parent.parent, given.parent):
            versions = _available_versions(candidate_root)
            if versions:
                root = candidate_root
                break
        hint = f" Versiones disponibles en '{root}': {', '.join(versions)}." if versions else ""
        raise FileNotFoundError(f"No se encontró el modelo '{model}'.{hint}")
    return path


def load_bundle(model: str | Path) -> dict:
    path = resolve_model_path(model)
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            f"No se pudo cargar el modelo '{path}'. El archivo puede estar dañado o haberse "
            f"creado con otra versión de scikit-learn. Detalle: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(bundle, dict) or any(key not in bundle for key in REQUIRED_BUNDLE_KEYS):
        raise ValueError(
            f"El archivo '{path}' no tiene el formato esperado: se esperaba un diccionario "
            f"con las claves {', '.join(REQUIRED_BUNDLE_KEYS)} generado por train.py."
        )
    if not hasattr(bundle["pipeline"], "predict"):
        raise ValueError(f"El archivo '{path}' no contiene un modelo con método predict().")
    return bundle


def prepare_input(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    prepared = prepare_features(frame)
    validate_features(prepared, feature_columns)  # columnas obligatorias y tipos numéricos
    return prepared[feature_columns].copy()  # mismo orden que en el entrenamiento


def count_missing(features: pd.DataFrame) -> dict[str, int]:
    return {column: int(n) for column, n in features.isna().sum().items() if n}


def unseen_categories(pipeline, features: pd.DataFrame) -> dict[str, list[str]]:
    try:
        transformer = pipeline.named_steps["preprocessing"]
        columns = next(cols for name, _, cols in transformer.transformers_ if name == "categorical")
        encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]
        known = dict(zip(columns, encoder.categories_))
    except (AttributeError, KeyError, StopIteration):
        return {}
    result = {}
    for column, categories in known.items():
        if column in features.columns:
            new_values = {str(v) for v in features[column].dropna().unique()} - {str(c) for c in categories}
            if new_values:
                result[column] = sorted(new_values)
    return result


def predict_frame(bundle: dict, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = prepare_input(frame, bundle["feature_columns"])
    info = {
        "filas": int(len(features)),
        "nulos_imputados": count_missing(features),
        "categorias_no_vistas": unseen_categories(bundle["pipeline"], features),
    }
    try:
        classes = bundle["pipeline"].predict(features)
        probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    except Exception as exc:
        raise ValueError(f"El modelo no pudo procesar los datos de entrada: {exc}") from exc

    labels = bundle["labels"]
    result = pd.DataFrame(
        {
            PREDICTION_COLUMN: np.asarray(classes).astype(int),
            LABEL_COLUMN: [labels[str(int(c))] for c in classes],
            PROBABILITY_COLUMN: np.round(probabilities, 6),
        }
    )
    if ID_COLUMN in frame.columns:  # permite relacionar cada resultado con su cliente
        result.insert(0, ID_COLUMN, frame[ID_COLUMN].to_numpy())
    return result, info


def write_csv_atomically(table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=".prediccion-", suffix=".tmp", dir=output_path.parent)
    os.close(handle)
    try:
        table.to_csv(tmp_name, index=False)
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, output_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    

def predict_file(model: str | Path, input_path: str | Path, output_path: str | Path) -> dict:
    output_path = Path(output_path)
    if output_path.is_dir():
        raise ValueError(f"La ruta de salida es una carpeta, no un archivo: {output_path}")
    if output_path.exists() and Path(input_path).exists() and output_path.resolve() == Path(input_path).resolve():
        raise ValueError("El archivo de salida no puede ser el mismo que el de entrada.")

    model_path = resolve_model_path(model)
    bundle = load_bundle(model_path)
    frame = read_csv(input_path)
    if frame.shape[1] == 1 and ";" in str(frame.columns[0]):
        raise ValueError("El CSV de entrada parece usar punto y coma (;) como separador. Guárdalo separado por comas (,).")
    table, info = predict_frame(bundle, frame)
    try:
        write_csv_atomically(table, output_path)
    except OSError as exc:
        raise ValueError(f"No se pudo escribir el archivo de salida '{output_path}': {exc}") from exc

    counts = table[PREDICTION_COLUMN].value_counts()
    return {
        **info,
        "modelo": model_path,
        "salida": output_path,
        "etiquetas": bundle["labels"],
        "conteo": {bundle["labels"][str(k)]: int(counts.get(k, 0)) for k in (1, 0)},
    }