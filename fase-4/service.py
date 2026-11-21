from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import Settings
from inference import (
    LABEL_COLUMN,
    PROBABILITY_COLUMN,
    load_bundle,
    predict_frame,
)
from pipeline import MODEL_FILENAME, NUMERIC_COLUMNS, train_and_save

BINARY_COLUMN = "SeniorCitizen"
VERSION_PATTERN = re.compile(r"v(\d+)")
MAX_VERSION_ATTEMPTS = 5


class ModelUnavailableError(Exception):
    """La versión configurada para predecir no existe o no se pudo cargar."""


class TrainingInProgressError(Exception):
    """Ya hay un entrenamiento en ejecución."""


def _natural_key(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name))


def list_versions(models_dir: Path) -> list[str]:
    if not models_dir.is_dir():
        return []
    names = [c.name for c in models_dir.iterdir() if c.is_dir() and (c / MODEL_FILENAME).is_file()]
    return sorted(names, key=_natural_key)


def next_version(models_dir: Path) -> str:
    numbers = [0]
    if models_dir.is_dir():
        for child in models_dir.iterdir():
            match = VERSION_PATTERN.fullmatch(child.name)
            if child.is_dir() and match:
                numbers.append(int(match.group(1)))
    return f"v{max(numbers) + 1}"


class ModelStore:

    def __init__(self, models_dir: Path, version: str) -> None:
        self.models_dir = Path(models_dir)
        self.version = version
        self._lock = threading.Lock()
        self._cache: tuple[tuple[Path, int], dict] | None = None

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.version / MODEL_FILENAME

    def status(self) -> dict:
        return {
            "version_activa": self.version,
            "modelo_disponible": self.model_path.is_file(),
            "versiones_disponibles": list_versions(self.models_dir),
        }

    def get(self) -> dict:
        path = self.model_path
        if not path.is_file():
            available = list_versions(self.models_dir)
            hint = (
                f" Versiones disponibles: {', '.join(available)}."
                if available
                else " No hay ninguna versión guardada; entrene una con POST /train."
            )
            raise ModelUnavailableError(
                f"El modelo de la versión '{self.version}' no está disponible.{hint} "
                "Configure MODEL_VERSION con una versión existente."
            )
        key = (path, path.stat().st_mtime_ns)
        with self._lock:
            if self._cache is None or self._cache[0] != key:
                try:
                    self._cache = (key, load_bundle(path))
                except (FileNotFoundError, ValueError) as exc:
                    self._cache = None
                    raise ModelUnavailableError(
                        f"No se pudo cargar el modelo de la versión '{self.version}': {exc}"
                    ) from exc
            return self._cache[1]


def record_to_frame(record: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = {column: np.nan if record[column] is None else record[column] for column in feature_columns}
    frame = pd.DataFrame([row], columns=list(feature_columns))
    categorical = [c for c in feature_columns if c not in NUMERIC_COLUMNS and c != BINARY_COLUMN]
    if categorical:
        frame[categorical] = frame[categorical].astype(object)
    return frame


def predict_record(bundle: dict, record: dict) -> dict:
    table, info = predict_frame(bundle, record_to_frame(record, bundle["feature_columns"]))
    row = table.iloc[0]
    return {
        "prediccion": str(row[LABEL_COLUMN]),
        "probabilidad_churn": float(row[PROBABILITY_COLUMN]),
        "nulos_imputados": sorted(info["nulos_imputados"]),
        "categorias_no_vistas": info["categorias_no_vistas"],
    }


def _stamp_git_commit(version_dir: Path, git_commit: str) -> None:
    path = version_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("git_revision") in (None, "no-disponible"):
        metadata["git_revision"] = git_commit
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_new_version(settings: Settings, lock: threading.Lock) -> dict:
    if not lock.acquire(blocking=False):
        raise TrainingInProgressError("Ya hay un entrenamiento en curso; espere a que termine.")
    try:
        last_error: FileExistsError | None = None
        for _ in range(MAX_VERSION_ATTEMPTS):
            version = next_version(settings.models_dir)
            try:
                version_dir = train_and_save(
                    settings.train_data_path,
                    settings.target,
                    settings.models_dir,
                    version,
                    settings.seed,
                    settings.test_size,
                )
            except FileExistsError as exc:  # otro proceso tomó el identificador: probar el siguiente
                last_error = exc
                continue
            if settings.git_commit:
                _stamp_git_commit(version_dir, settings.git_commit)
            metrics = json.loads((version_dir / "metrics.json").read_text(encoding="utf-8"))
            return {"version": version, "ruta": version_dir.as_posix(), "metricas": metrics}
        raise RuntimeError(f"No se pudo reservar un identificador de versión libre: {last_error}")
    finally:
        lock.release()