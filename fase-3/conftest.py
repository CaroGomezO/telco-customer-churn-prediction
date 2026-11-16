from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FASE3 = Path(__file__).resolve().parent
if str(FASE3) not in sys.path:
    sys.path.insert(0, str(FASE3))

from api import create_app
from config import Settings
from pipeline import train_and_save

FEATURES = ["gender", "SeniorCitizen", "Partner", "tenure", "Contract", "MonthlyCharges"]


def make_labeled(n: int = 80) -> pd.DataFrame:
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


@pytest.fixture(scope="session")
def datos_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("datos") / "datos.csv"
    make_labeled().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def modelos_base(tmp_path_factory, datos_csv):
    """Directorio con dos versiones entrenadas (v1 con semilla 42 y v2 con semilla 7)."""
    models = tmp_path_factory.mktemp("modelos") / "models"
    train_and_save(datos_csv, "Churn", models, "v1", seed=42)
    train_and_save(datos_csv, "Churn", models, "v2", seed=7)
    return models


@pytest.fixture()
def settings(tmp_path, modelos_base, datos_csv):
    """Copia aislada de los modelos para que cada prueba pueda modificarla."""
    models = tmp_path / "models"
    shutil.copytree(modelos_base, models)
    return Settings(models_dir=models, model_version="v1", train_data_path=datos_csv)


@pytest.fixture()
def client(settings):
    return create_app(settings).test_client()


@pytest.fixture()
def registro_valido():
    row = make_labeled().iloc[0]
    return {
        "customerID": row["customerID"],
        "gender": row["gender"],
        "SeniorCitizen": int(row["SeniorCitizen"]),
        "Partner": row["Partner"],
        "tenure": int(row["tenure"]),
        "Contract": row["Contract"],
        "MonthlyCharges": float(row["MonthlyCharges"]),
    }