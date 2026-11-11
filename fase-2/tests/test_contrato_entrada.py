import re
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from inference import (
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    load_bundle,
    predict_file,
    predict_frame,
)

FASE2 = Path(__file__).resolve().parents[1]
MODELOS = FASE2 / "models"
EJEMPLO = FASE2 / "data" / "ejemplo_prediccion.csv"
README = FASE2 / "README.md"

# Versiones entregadas: todas las carpetas de models/ que contienen un modelo.
VERSIONES = sorted(p.parent.name for p in MODELOS.glob("*/modelo.joblib"))

# Contrato de entrada: las 18 variables que el modelo necesita (el orden no importa).
COLUMNAS_REQUERIDAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
]

# Valores categóricos que el modelo conoce (SeniorCitizen llega como 0/1 y se convierte a No/Yes antes del modelo. Se prueba aparte).
VALORES_PERMITIDOS = {
    "gender": {"Female", "Male"},
    "Partner": {"No", "Yes"},
    "Dependents": {"No", "Yes"},
    "PhoneService": {"No", "Yes"},
    "MultipleLines": {"No", "No phone service", "Yes"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"No", "No internet service", "Yes"},
    "OnlineBackup": {"No", "No internet service", "Yes"},
    "DeviceProtection": {"No", "No internet service", "Yes"},
    "TechSupport": {"No", "No internet service", "Yes"},
    "StreamingTV": {"No", "No internet service", "Yes"},
    "StreamingMovies": {"No", "No internet service", "Yes"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"No", "Yes"},
    "PaymentMethod": {"Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"},
}

COLUMNAS_SALIDA = ["customerID", PREDICTION_COLUMN, LABEL_COLUMN, PROBABILITY_COLUMN]


def ejemplo() -> pd.DataFrame:
    return pd.read_csv(EJEMPLO)


def escribir(frame: pd.DataFrame, carpeta: Path, nombre: str = "entrada.csv") -> Path:
    ruta = carpeta / nombre
    frame.to_csv(ruta, index=False)
    return ruta


def clientes_nuevos() -> pd.DataFrame:
    base = ejemplo().iloc[[0, 2, 4]].copy().reset_index(drop=True)
    base["customerID"] = ["NUEVO-0001", "NUEVO-0002", "NUEVO-0003"]
    base.loc[2, "PaymentMethod"] = "Criptomoneda" # nunca vista en el entrenamiento
    base.loc[2, "InternetService"] = "5G" # nunca vista en el entrenamiento
    return base


def test_las_versiones_entregadas_existen():
    assert {"v1", "v2"} <= set(VERSIONES), f"Se esperaban al menos v1 y v2 en models/, hay: {VERSIONES}"


def test_el_ejemplo_cumple_el_contrato_y_no_trae_la_variable_objetivo():
    columnas = set(ejemplo().columns)
    assert set(COLUMNAS_REQUERIDAS) <= columnas
    assert "Churn" not in columnas


def test_el_readme_documenta_cada_columna_requerida():
    texto = README.read_text(encoding="utf-8")
    faltan = [c for c in COLUMNAS_REQUERIDAS if f"`{c}`" not in texto]
    assert not faltan, f"El README no documenta estas columnas del contrato: {faltan}"


@pytest.mark.parametrize("version", VERSIONES)
def test_cada_version_exige_exactamente_las_columnas_del_contrato(version):
    bundle = load_bundle(MODELOS / version)
    assert sorted(bundle["feature_columns"]) == sorted(COLUMNAS_REQUERIDAS)
    assert bundle["target"] == "Churn"
    assert bundle["labels"] == {"0": "No", "1": "Yes"}


@pytest.mark.parametrize("version", VERSIONES)
def test_los_valores_documentados_coinciden_con_los_que_aprendio_el_modelo(version):
    bundle = joblib.load(MODELOS / version / "modelo.joblib")
    transformador = bundle["pipeline"].named_steps["preprocessing"]
    columnas = next(cols for nombre, _, cols in transformador.transformers_ if nombre == "categorical")
    codificador = transformador.named_transformers_["categorical"].named_steps["encoder"]
    aprendido = {col: {str(c) for c in cats} for col, cats in zip(columnas, codificador.categories_)}
    assert aprendido.pop("SeniorCitizen") == {"No", "Yes"}  # 0/1 -> No/Yes antes del modelo
    assert aprendido == VALORES_PERMITIDOS


@pytest.mark.parametrize("columna", COLUMNAS_REQUERIDAS)
def test_si_falta_una_columna_requerida_el_error_la_nombra_y_no_se_crea_salida(columna, tmp_path):
    entrada = escribir(ejemplo().drop(columns=[columna]), tmp_path)
    salida = tmp_path / "pred.csv"
    with pytest.raises(ValueError, match=f"Faltan columnas requeridas: {re.escape(columna)}"):
        predict_file(MODELOS / "v1", entrada, salida)
    assert not salida.exists()


def test_senior_citizen_acepta_0_y_1_como_numero_o_texto_y_rechaza_otros_valores():
    bundle = load_bundle(MODELOS / "v1")
    base = ejemplo()
    referencia, _ = predict_frame(bundle, base)

    como_texto = base.copy()
    como_texto["SeniorCitizen"] = como_texto["SeniorCitizen"].astype(str)
    resultado, _ = predict_frame(bundle, como_texto)
    pd.testing.assert_series_equal(resultado[PROBABILITY_COLUMN], referencia[PROBABILITY_COLUMN])

    for invalido in (2, "Yes"):
        malo = base.copy()
        malo["SeniorCitizen"] = malo["SeniorCitizen"].astype(object)
        malo.loc[0, "SeniorCitizen"] = invalido
        with pytest.raises(ValueError, match="0 o 1"):
            predict_frame(bundle, malo)


def test_la_salida_cumple_el_contrato_y_conserva_el_orden_de_las_filas(tmp_path):
    entrada = ejemplo()
    salida = tmp_path / "pred.csv"
    predict_file(MODELOS / "v1", escribir(entrada, tmp_path), salida)
    resultado = pd.read_csv(salida)
    assert list(resultado.columns) == COLUMNAS_SALIDA
    assert resultado["customerID"].tolist() == entrada["customerID"].tolist()
    assert set(resultado[PREDICTION_COLUMN]) <= {0, 1}
    assert resultado[LABEL_COLUMN].tolist() == ["Yes" if p == 1 else "No" for p in resultado[PREDICTION_COLUMN]]
    assert resultado[PROBABILITY_COLUMN].between(0, 1).all()


def test_clientes_nuevos_con_categorias_desconocidas_en_un_proceso_independiente(tmp_path):
    nuevos = clientes_nuevos()
    salida = tmp_path / "pred.csv"
    completado = subprocess.run(
        [sys.executable, "predict.py", "--model", str(MODELOS / "v1"),
        "--input", str(escribir(nuevos, tmp_path)), "--output", str(salida)],
        cwd=FASE2, capture_output=True, text=True,
    )
    assert completado.returncode == 0, completado.stderr
    resultado = pd.read_csv(salida)
    assert resultado["customerID"].tolist() == ["NUEVO-0001", "NUEVO-0002", "NUEVO-0003"]
    assert resultado[PROBABILITY_COLUMN].between(0, 1).all()
    # Las categorías desconocidas no rompen la ejecución, pero se avisan por stderr.
    assert "PaymentMethod" in completado.stderr and "InternetService" in completado.stderr


_CODIGO_SIN_AVISOS_DE_VERSION = (
    "import runpy, sys, warnings\n"
    "from sklearn.exceptions import InconsistentVersionWarning\n"
    "warnings.simplefilter('error', InconsistentVersionWarning)\n"
    "sys.argv = ['predict.py'] + sys.argv[1:]\n"
    "runpy.run_path('predict.py', run_name='__main__')\n"
)


@pytest.mark.parametrize("version", VERSIONES)
def test_cada_version_carga_y_predice_en_un_proceso_independiente_sin_avisos_de_version(version, tmp_path):
    salida = tmp_path / "pred.csv"
    completado = subprocess.run(
        [sys.executable, "-c", _CODIGO_SIN_AVISOS_DE_VERSION,
        "--model", str(MODELOS / version), "--input", str(EJEMPLO), "--output", str(salida)],
        cwd=FASE2, capture_output=True, text=True,
    )
    assert completado.returncode == 0, (
        f"La versión '{version}' no cargó sin avisos de versión (¿se guardó con otra versión de "
        f"scikit-learn que la fijada en requirements.txt?):\n{completado.stderr}"
    )
    assert len(pd.read_csv(salida)) == len(ejemplo())