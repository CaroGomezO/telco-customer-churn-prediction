from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import pytest

import cliente_prueba
from api import create_app
from config import Settings
from conftest import requiere_entorno_fijado

FASE4 = Path(__file__).resolve().parents[1]
MODELOS = FASE4 / "models"
EJEMPLO = FASE4 / "data" / "ejemplo_prediccion.csv"
README = FASE4 / "README.md"
REFERENCIA = FASE4 / "tests" / "referencia"

ENTREGADAS = ["v1", "v2", "v3"]

# Contrato de entrada: las 18 variables que el modelo necesita (el orden no importa).
COLUMNAS_REQUERIDAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
]

# Valores categóricos que el modelo conoce (SeniorCitizen llega como 0/1 y se convierte a No/Yes).
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


def leer_ejemplo() -> list[dict]:
    with open(EJEMPLO, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def leer_referencia(version: str) -> dict[str, dict]:
    with open(REFERENCIA / f"predicciones_{version}.csv", newline="", encoding="utf-8") as handle:
        return {fila["customerID"]: fila for fila in csv.DictReader(handle)}


@pytest.mark.parametrize("version", ENTREGADAS)
def test_las_versiones_entregadas_conservan_modelo_metricas_y_metadatos(version):
    carpeta = MODELOS / version
    assert {p.name for p in carpeta.iterdir()} == {"modelo.joblib", "metrics.json", "metadata.json"}
    metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    metricas = json.loads((carpeta / "metrics.json").read_text(encoding="utf-8"))
    assert metadata["version"] == version
    assert {"seed", "python", "git_revision", "created_at", "data"} <= set(metadata)
    assert set(metricas) == {"model", "baseline"}


def test_el_readme_documenta_cada_columna_requerida_y_sus_valores():
    texto = README.read_text(encoding="utf-8")
    sin_columna = [c for c in COLUMNAS_REQUERIDAS if f"`{c}`" not in texto]
    assert not sin_columna, f"El README no documenta estas columnas del contrato: {sin_columna}"
    sin_valor = sorted(
        {valor for valores in VALORES_PERMITIDOS.values() for valor in valores if f"`{valor}`" not in texto}
    )
    assert not sin_valor, f"El README no documenta estos valores admitidos: {sin_valor}"


@pytest.mark.parametrize("version", ENTREGADAS)
def test_cada_version_exige_exactamente_las_columnas_del_contrato(version):
    metadata = json.loads((MODELOS / version / "metadata.json").read_text(encoding="utf-8"))
    assert sorted(metadata["input_columns"]) == sorted(COLUMNAS_REQUERIDAS)
    assert metadata["target"] == "Churn"


@requiere_entorno_fijado
@pytest.mark.parametrize("version", ENTREGADAS)
def test_los_valores_documentados_coinciden_con_los_que_aprendio_el_modelo(version):
    bundle = joblib.load(MODELOS / version / "modelo.joblib")
    transformador = bundle["pipeline"].named_steps["preprocessing"]
    columnas = next(cols for nombre, _, cols in transformador.transformers_ if nombre == "categorical")
    codificador = transformador.named_transformers_["categorical"].named_steps["encoder"]
    aprendido = {col: {str(c) for c in cats} for col, cats in zip(columnas, codificador.categories_)}
    assert aprendido.pop("SeniorCitizen") == {"No", "Yes"}  # 0/1 -> No/Yes antes del modelo
    assert aprendido == VALORES_PERMITIDOS


_CODIGO_SIN_AVISOS_DE_VERSION = """
import json, sys, warnings
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("error", InconsistentVersionWarning)
import cliente_prueba
from api import create_app
from config import Settings

version, models, ejemplo = sys.argv[1:4]
cliente = create_app(Settings(Path(models), version, Path("sin-datos.csv"))).test_client()
fila = cliente_prueba.leer_fila(Path(ejemplo), 0)
respuesta = cliente.post("/predict", json=cliente_prueba.fila_a_registro(fila))
print(json.dumps({"estado": respuesta.status_code, **respuesta.get_json()}))
"""


@requiere_entorno_fijado
@pytest.mark.parametrize("version", ENTREGADAS)
def test_cada_version_carga_y_predice_en_un_proceso_independiente_sin_avisos_de_version(version):
    completado = subprocess.run(
        [sys.executable, "-c", _CODIGO_SIN_AVISOS_DE_VERSION, version, str(MODELOS), str(EJEMPLO)],
        cwd=FASE4, capture_output=True, text=True,
    )
    assert completado.returncode == 0, (
        f"La versión '{version}' no cargó sin avisos de versión (¿se guardó con otra versión de "
        f"scikit-learn que la fijada en requirements.txt?):\n{completado.stderr}"
    )
    cuerpo = json.loads(completado.stdout.strip().splitlines()[-1])
    assert cuerpo["estado"] == 200
    assert cuerpo["version"] == version
    assert cuerpo["prediccion"] in {"Yes", "No"}


def _puerto_libre() -> int:
    with socket.socket() as servidor:
        servidor.bind(("127.0.0.1", 0))
        return servidor.getsockname()[1]


def _consultar(url: str, cuerpo: dict | None = None) -> tuple[int, dict]:
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    peticion = urllib.request.Request(url, data=datos, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(peticion, timeout=10) as respuesta:
            return respuesta.status, json.loads(respuesta.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_la_api_arranca_como_proceso_independiente_y_responde_por_http(settings, registro_valido, tmp_path):
    puerto = _puerto_libre()
    entorno = {
        **os.environ,
        "HOST": "127.0.0.1",
        "PORT": str(puerto),
        "MODELS_DIR": str(settings.models_dir),
        "MODEL_VERSION": "v2",
        "LOG_PATH": str(tmp_path / "predicciones.csv"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    registro = tmp_path / "api.log"
    with open(registro, "w", encoding="utf-8") as salida:
        proceso = subprocess.Popen(
            [sys.executable, "api.py"], cwd=FASE4, env=entorno, stdout=salida, stderr=subprocess.STDOUT
        )
    base = f"http://127.0.0.1:{puerto}"
    try:
        limite = time.time() + 40
        salud = None
        while time.time() < limite and salud is None:
            if proceso.poll() is not None:
                break
            try:
                salud = _consultar(f"{base}/health")
            except (urllib.error.URLError, OSError):
                time.sleep(0.3)
        assert salud is not None, f"La API no respondió. Registro:\n{registro.read_text(encoding='utf-8')}"
        assert salud[0] == 200 and salud[1]["version_activa"] == "v2"

        estado, cuerpo = _consultar(f"{base}/predict", registro_valido)
        assert estado == 200 and cuerpo["version"] == "v2" and cuerpo["prediccion"] in {"Yes", "No"}

        estado, cuerpo = _consultar(f"{base}/predict", {k: v for k, v in registro_valido.items() if k != "tenure"})
        assert estado == 422 and "tenure" in cuerpo["mensaje"]
    finally:
        proceso.terminate()
        try:
            proceso.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proceso.kill()


@requiere_entorno_fijado
@pytest.mark.parametrize("version", ENTREGADAS)
def test_las_predicciones_de_la_entrada_conocida_no_cambian_respecto_a_la_fase_2(version):
    referencia = leer_referencia(version)
    cliente = create_app(Settings(MODELOS, version, FASE4.parent / "fase-1" / "DatosCrudos.csv")).test_client()
    filas = leer_ejemplo()
    assert set(referencia) == {f["customerID"] for f in filas}
    for fila in filas:
        cuerpo = cliente.post("/predict", json=cliente_prueba.fila_a_registro(fila)).get_json()
        esperado = referencia[fila["customerID"]]
        assert cuerpo["version"] == version
        assert cuerpo["prediccion"] == esperado["etiqueta"], fila["customerID"]
        assert cuerpo["probabilidad_churn"] == pytest.approx(float(esperado["probabilidad_churn"]), abs=2e-6), fila["customerID"]
