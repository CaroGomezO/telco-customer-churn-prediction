from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_MODELS_DIR = "models"
DEFAULT_MODEL_VERSION = "v3"
DEFAULT_TRAIN_DATA = "../fase-1/DatosCrudos.csv"
DEFAULT_LOG_PATH = "logs/predicciones.csv"
DEFAULT_TARGET = "Churn"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


def validate_version_name(version: str) -> str:
    if not version or version in {".", ".."} or Path(version).name != version or "\\" in version:
        raise ValueError(
            f"Versión inválida '{version}': debe ser un nombre simple de carpeta, por ejemplo 'v1'."
        )
    return version


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    model_version: str
    train_data_path: Path
    target: str = DEFAULT_TARGET
    seed: int = DEFAULT_SEED
    test_size: float = DEFAULT_TEST_SIZE
    git_commit: str | None = None
    # Registro de /predict (Fase 4). from_env() siempre lo define (LOG_PATH);
    # si se construye Settings a mano sin log_path, la API no registra solicitudes.
    log_path: Path | None = None

    def __post_init__(self) -> None:
        validate_version_name(self.model_version)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env

        def read(name: str, default: str) -> str:
            value = env.get(name)
            return default if value is None or not value.strip() else value.strip()

        try:
            seed = int(read("SEED", str(DEFAULT_SEED)))
        except ValueError as exc:
            raise ValueError(f"SEED debe ser un número entero (valor recibido: {env.get('SEED')!r}).") from exc
        try:
            test_size = float(read("TEST_SIZE", str(DEFAULT_TEST_SIZE)))
        except ValueError as exc:
            raise ValueError(
                f"TEST_SIZE debe ser un número entre 0 y 1 (valor recibido: {env.get('TEST_SIZE')!r})."
            ) from exc
        return cls(
            models_dir=Path(read("MODELS_DIR", DEFAULT_MODELS_DIR)),
            model_version=read("MODEL_VERSION", DEFAULT_MODEL_VERSION),
            train_data_path=Path(read("TRAIN_DATA_PATH", DEFAULT_TRAIN_DATA)),
            target=read("TARGET", DEFAULT_TARGET),
            seed=seed,
            test_size=test_size,
            git_commit=(env.get("GIT_COMMIT") or "").strip() or None,
            log_path=Path(read("LOG_PATH", DEFAULT_LOG_PATH)),
        )