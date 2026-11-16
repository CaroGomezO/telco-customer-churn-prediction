from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import sklearn

import cliente_prueba
from api import create_app
from config import Settings
from inference import load_bundle
from pipeline import NUMERIC_COLUMNS
from validation import validate_record

FASE3 = Path(__file__).resolve().parents[1]
FASE2 = FASE3.parent / "fase-2"
MODULOS_FASE2 = ["pipeline.py", "inference.py", "train.py", "predict.py"]
CAMPOS_OBLIGATORIOS = ["gender", "SeniorCitizen", "Partner", "tenure", "Contract", "MonthlyCharges"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_health_responde_200_y_estado(client):
    response = client.get("/health")
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["version_activa"] == "v1"
    assert body["modelo_disponible"] is True
    assert body["versiones_disponibles"] == ["v1", "v2"]


def test_health_sigue_en_200_si_el_modelo_no_esta_disponible(settings):
    settings = Settings(settings.models_dir, "v99", settings.train_data_path)
    response = create_app(settings).test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
    assert response.get_json()["modelo_disponible"] is False


def test_predict_entrada_valida_devuelve_prediccion_y_version(client, registro_valido):
    response = client.post("/predict", json=registro_valido)
    body = response.get_json()
    assert response.status_code == 200
    assert set(body) >= {"prediccion", "version"}
    assert body["prediccion"] in {"Yes", "No"}
    assert body["version"] == "v1"
    assert 0.0 <= body["probabilidad_churn"] <= 1.0
    assert body["nulos_imputados"] == []
    assert body["categorias_no_vistas"] == {}
    assert body["customerID"] == registro_valido["customerID"]


def test_predict_coincide_con_el_modelo_guardado(client, settings, registro_valido):
    import pandas as pd
    from inference import predict_frame

    bundle = load_bundle(settings.models_dir / "v1")
    esperado, _ = predict_frame(bundle, pd.DataFrame([registro_valido]))
    body = client.post("/predict", json=registro_valido).get_json()
    assert body["prediccion"] == esperado["etiqueta"].iloc[0]
    assert body["probabilidad_churn"] == pytest.approx(float(esperado["probabilidad_churn"].iloc[0]), abs=1e-9)


def test_predict_acepta_json_sin_cabecera_content_type(client, registro_valido):
    response = client.post("/predict", data=json.dumps(registro_valido))
    assert response.status_code == 200


def test_predict_ignora_campos_adicionales_y_no_exige_customerid(client, registro_valido):
    registro = {k: v for k, v in registro_valido.items() if k != "customerID"}
    registro["TotalCharges"] = 1234.5
    registro["campo_extra"] = "x"
    response = client.post("/predict", json=registro)
    assert response.status_code == 200
    assert "customerID" not in response.get_json()


def test_predict_no_depende_del_orden_de_las_claves(client, registro_valido):
    invertido = dict(reversed(list(registro_valido.items())))
    a = client.post("/predict", json=registro_valido).get_json()
    b = client.post("/predict", json=invertido).get_json()
    assert a == b


def test_predict_campo_obligatorio_ausente_se_rechaza(client, registro_valido):
    registro = {k: v for k, v in registro_valido.items() if k != "tenure"}
    response = client.post("/predict", json=registro)
    body = response.get_json()
    assert response.status_code in (400, 422)
    assert "tenure" in body["mensaje"]
    assert body["detalles"] == [{"campo": "tenure", "problema": "campo obligatorio ausente"}]
    assert "prediccion" not in body


@pytest.mark.parametrize("campo", CAMPOS_OBLIGATORIOS)
def test_predict_rechaza_cada_campo_obligatorio_ausente(client, registro_valido, campo):
    registro = {k: v for k, v in registro_valido.items() if k != campo}
    response = client.post("/predict", json=registro)
    assert response.status_code == 422
    assert campo in response.get_json()["mensaje"]


def test_predict_informa_todos_los_campos_ausentes_a_la_vez(client, registro_valido):
    registro = {k: v for k, v in registro_valido.items() if k not in {"tenure", "Contract"}}
    response = client.post("/predict", json=registro)
    campos = {d["campo"] for d in response.get_json()["detalles"]}
    assert response.status_code == 422
    assert campos == {"tenure", "Contract"}


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("tenure", "abc"),
        ("tenure", "12"),
        ("tenure", 3.5),
        ("tenure", -1),
        ("tenure", True),
        ("MonthlyCharges", "alto"),
        ("MonthlyCharges", -5.0),
        ("MonthlyCharges", [20.5]),
        ("MonthlyCharges", 10**400),
        ("SeniorCitizen", 2),
        ("SeniorCitizen", "Yes"),
        ("SeniorCitizen", True),
        ("gender", 5),
        ("gender", ["Male"]),
        ("gender", ""),
        ("Contract", {"tipo": "mensual"}),
    ],
)
def test_predict_rechaza_tipos_o_valores_incompatibles(client, registro_valido, campo, valor):
    response = client.post("/predict", json={**registro_valido, campo: valor})
    body = response.get_json()
    assert response.status_code == 422
    assert body["error"] == "entrada_invalida"
    assert [d["campo"] for d in body["detalles"]] == [campo]
    assert campo in body["mensaje"]


def test_predict_rechaza_nan_e_infinito(client, registro_valido):
    cuerpo = json.dumps({**registro_valido, "MonthlyCharges": float("nan")})
    response = client.post("/predict", data=cuerpo)
    assert response.status_code == 422
    assert "MonthlyCharges" in response.get_json()["mensaje"]


@pytest.mark.parametrize("cuerpo", ["", "{malo", "[1, 2, 3]", "null", '"texto"', "42"])
def test_predict_rechaza_cuerpos_que_no_son_un_objeto_json(client, cuerpo):
    response = client.post("/predict", data=cuerpo, content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error"] == "json_invalido"


def test_predict_nulo_numerico_se_imputa_con_la_mediana_del_entrenamiento(client, settings, registro_valido):
    bundle = load_bundle(settings.models_dir / "v1")
    columnas = list(bundle["pipeline"].named_steps["preprocessing"].transformers_[0][2])
    mediana = bundle["pipeline"].named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"].statistics_
    mediana_cargos = float(mediana[columnas.index("MonthlyCharges")])

    con_nulo = client.post("/predict", json={**registro_valido, "MonthlyCharges": None})
    con_mediana = client.post("/predict", json={**registro_valido, "MonthlyCharges": mediana_cargos})
    assert con_nulo.status_code == 200
    assert con_nulo.get_json()["nulos_imputados"] == ["MonthlyCharges"]
    assert con_nulo.get_json()["probabilidad_churn"] == pytest.approx(
        con_mediana.get_json()["probabilidad_churn"], abs=1e-6
    )


def test_predict_nulo_categorico_se_imputa_con_la_moda_del_entrenamiento(client, settings, registro_valido):
    bundle = load_bundle(settings.models_dir / "v1")
    transformador = bundle["pipeline"].named_steps["preprocessing"]
    columnas = list(transformador.transformers_[1][2])
    moda = transformador.named_transformers_["categorical"].named_steps["imputer"].statistics_
    moda_contrato = str(moda[columnas.index("Contract")])

    con_nulo = client.post("/predict", json={**registro_valido, "Contract": None})
    con_moda = client.post("/predict", json={**registro_valido, "Contract": moda_contrato})
    assert con_nulo.status_code == 200
    assert con_nulo.get_json()["nulos_imputados"] == ["Contract"]
    assert con_nulo.get_json()["probabilidad_churn"] == pytest.approx(
        con_moda.get_json()["probabilidad_churn"], abs=1e-6
    )


def test_predict_con_todas_las_variables_nulas_sigue_funcionando(client):
    registro = {campo: None for campo in CAMPOS_OBLIGATORIOS}
    response = client.post("/predict", json=registro)
    assert response.status_code == 200
    assert sorted(response.get_json()["nulos_imputados"]) == sorted(CAMPOS_OBLIGATORIOS)


def test_predict_categoria_no_vista_se_procesa_y_se_advierte(client, registro_valido):
    response = client.post("/predict", json={**registro_valido, "Contract": "Weekly"})
    assert response.status_code == 200
    assert response.get_json()["categorias_no_vistas"] == {"Contract": ["Weekly"]}
    

def test_predict_usa_la_version_configurada(settings, registro_valido):
    settings_v2 = Settings(settings.models_dir, "v2", settings.train_data_path)
    body = create_app(settings_v2).test_client().post("/predict", json=registro_valido).get_json()
    assert body["version"] == "v2"


def test_predict_devuelve_503_si_la_version_configurada_no_existe(settings, registro_valido):
    settings = Settings(settings.models_dir, "v99", settings.train_data_path)
    response = create_app(settings).test_client().post("/predict", json=registro_valido)
    body = response.get_json()
    assert response.status_code == 503
    assert body["error"] == "modelo_no_disponible"
    assert "v99" in body["mensaje"]
    assert body["versiones_disponibles"] == ["v1", "v2"]
    assert "MODEL_VERSION" in body["mensaje"]


def test_predict_devuelve_503_si_el_modelo_esta_danado(settings, registro_valido):
    carpeta = settings.models_dir / "v-roto"
    carpeta.mkdir()
    (carpeta / "modelo.joblib").write_bytes(b"esto no es un modelo")
    settings = Settings(settings.models_dir, "v-roto", settings.train_data_path)
    response = create_app(settings).test_client().post("/predict", json=registro_valido)
    assert response.status_code == 503
    assert response.get_json()["error"] == "modelo_no_disponible"


def test_la_version_no_puede_ser_una_ruta():
    with pytest.raises(ValueError):
        Settings.from_env({"MODEL_VERSION": "../fase-2/models/v1"})


def test_configuracion_desde_variables_de_entorno():
    settings = Settings.from_env(
        {"MODELS_DIR": "otros", "MODEL_VERSION": "v7", "SEED": "7", "TARGET": "Salida", "GIT_COMMIT": "abc1234"}
    )
    assert (settings.models_dir, settings.model_version, settings.seed) == (Path("otros"), "v7", 7)
    assert (settings.target, settings.git_commit) == ("Salida", "abc1234")
    por_defecto = Settings.from_env({})
    assert (por_defecto.model_version, por_defecto.seed, por_defecto.git_commit) == ("v1", 42, None)
    with pytest.raises(ValueError):
        Settings.from_env({"SEED": "no-es-un-numero"})


def test_train_guarda_una_version_nueva_y_conserva_las_anteriores(client, settings, registro_valido):
    antes = {f"{v}/{n}": sha256(settings.models_dir / v / n) for v in ("v1", "v2") for n in ("modelo.joblib", "metrics.json", "metadata.json")}

    response = client.post("/train")
    body = response.get_json()
    assert response.status_code == 201
    assert body["version"] == "v3"
    assert body["archivos"] == ["modelo.joblib", "metrics.json", "metadata.json"]
    assert set(body["metricas"]) == {"model", "baseline"}
    nueva = settings.models_dir / "v3"
    assert {p.name for p in nueva.iterdir()} == {"modelo.joblib", "metrics.json", "metadata.json"}
    assert json.loads((nueva / "metadata.json").read_text(encoding="utf-8"))["version"] == "v3"

    despues = {k: sha256(settings.models_dir / k) for k in antes}
    assert despues == antes  # v1 y v2 intactas
    assert load_bundle(nueva)["target"] == "Churn"

    # La predicción sigue usando la versión configurada hasta que se cambie.
    assert client.post("/predict", json=registro_valido).get_json()["version"] == "v1"
    assert client.get("/health").get_json()["versiones_disponibles"] == ["v1", "v2", "v3"]


def test_train_dos_veces_no_sobrescribe_versiones(client, settings):
    primera = client.post("/train").get_json()["version"]
    segunda = client.post("/train").get_json()["version"]
    assert (primera, segunda) == ("v3", "v4")
    assert (settings.models_dir / "v3" / "modelo.joblib").is_file()
    assert (settings.models_dir / "v4" / "modelo.joblib").is_file()


def test_train_sobre_una_carpeta_vacia_crea_v1(tmp_path, datos_csv):
    settings = Settings(tmp_path / "nuevo" / "models", "v1", datos_csv)
    response = create_app(settings).test_client().post("/train")
    assert response.status_code == 201
    assert response.get_json()["version"] == "v1"


def test_train_registra_el_commit_cuando_git_no_esta_disponible(settings, monkeypatch):
    import pipeline

    monkeypatch.setattr(pipeline, "_git_revision", lambda: "no-disponible")
    con_commit = Settings(settings.models_dir, "v1", settings.train_data_path, git_commit="abc1234")
    body = create_app(con_commit).test_client().post("/train").get_json()
    metadata = json.loads((Path(body["ruta"]) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["git_revision"] == "abc1234"


def test_train_sin_datos_devuelve_503_y_no_crea_versiones(settings):
    sin_datos = Settings(settings.models_dir, "v1", settings.models_dir / "no_existe.csv")
    response = create_app(sin_datos).test_client().post("/train")
    assert response.status_code == 503
    assert response.get_json()["error"] == "datos_no_disponibles"
    assert not (settings.models_dir / "v3").exists()


def test_train_con_datos_invalidos_devuelve_500_con_mensaje(settings, tmp_path):
    csv_malo = tmp_path / "malo.csv"
    csv_malo.write_text("a,b\n1,2\n", encoding="utf-8")
    malos = Settings(settings.models_dir, "v1", csv_malo)
    response = create_app(malos).test_client().post("/train")
    assert response.status_code == 500
    assert response.get_json()["error"] == "entrenamiento_fallido"
    assert "Churn" in response.get_json()["mensaje"]


def test_train_rechaza_un_segundo_entrenamiento_simultaneo(settings):
    app = create_app(settings)
    lock = app.extensions["churn"]["train_lock"]
    assert lock.acquire(blocking=False)
    try:
        response = app.test_client().post("/train")
    finally:
        lock.release()
    assert response.status_code == 409
    assert response.get_json()["error"] == "entrenamiento_en_curso"
    assert not (settings.models_dir / "v3").exists()


def test_rutas_y_metodos_inexistentes_devuelven_json(client):
    no_existe = client.get("/no-existe")
    assert no_existe.status_code == 404
    assert no_existe.get_json()["error"] == "ruta_no_encontrada"
    metodo = client.get("/predict")
    assert metodo.status_code == 405
    assert "POST" in metodo.headers["Allow"]
    assert metodo.get_json()["error"] == "metodo_no_permitido"


def test_importar_la_api_no_lee_modelos_ni_datos(tmp_path):
    app = create_app(Settings(tmp_path / "x", "v1", tmp_path / "y.csv"))
    assert app.test_client().get("/health").status_code == 200


def test_los_modulos_reutilizados_son_identicos_a_los_de_la_fase_2():
    if not FASE2.is_dir():
        pytest.skip("No existe ../fase-2 (el repositorio se copió sin ella).")
    for nombre in MODULOS_FASE2:
        copia = (FASE3 / nombre).read_bytes().replace(b"\r\n", b"\n")
        original = (FASE2 / nombre).read_bytes().replace(b"\r\n", b"\n")
        assert copia == original, f"{nombre} difiere de fase-2/{nombre}"


def test_el_csv_de_ejemplo_cumple_el_esquema_de_las_versiones_entregadas():
    with open(FASE3 / "data" / "ejemplo_prediccion.csv", newline="", encoding="utf-8") as handle:
        filas = list(csv.DictReader(handle))
    assert len(filas) == 8
    for version in ("v1", "v2"):
        metadata = json.loads((FASE3 / "models" / version / "metadata.json").read_text(encoding="utf-8"))
        for fila in filas:
            registro = cliente_prueba.fila_a_registro(fila)
            _, problemas = validate_record(registro, metadata["input_columns"])
            assert problemas == [], f"{version}, {fila['customerID']}: {problemas}"


def test_los_json_de_ejemplo_de_la_documentacion_son_coherentes():
    valido = json.loads((FASE3 / "data" / "ejemplo_solicitud.json").read_text(encoding="utf-8"))
    invalido = json.loads((FASE3 / "data" / "ejemplo_solicitud_invalida.json").read_text(encoding="utf-8"))
    columnas = json.loads((FASE3 / "models" / "v1" / "metadata.json").read_text(encoding="utf-8"))["input_columns"]
    assert validate_record(valido, columnas)[1] == []
    assert [p["campo"] for p in validate_record(invalido, columnas)[1]] == ["tenure"]


def _scikit_learn_fijado() -> str | None:
    for linea in (FASE3 / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if linea.lower().startswith("scikit-learn=="):
            return linea.split("==")[1].strip()
    return None


@pytest.mark.skipif(
    sklearn.__version__ != _scikit_learn_fijado(),
    reason="Los modelos entregados se guardaron con la versión de scikit-learn fijada en requirements.txt.",
)
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_la_api_predice_con_los_modelos_entregados(version):
    entregados = Settings(FASE3 / "models", version, FASE3.parent / "fase-1" / "DatosCrudos.csv")
    cliente = create_app(entregados).test_client()
    with open(FASE3 / "data" / "ejemplo_prediccion.csv", newline="", encoding="utf-8") as handle:
        filas = list(csv.DictReader(handle))
    for fila in filas:
        response = cliente.post("/predict", json=cliente_prueba.fila_a_registro(fila))
        body = response.get_json()
        assert response.status_code == 200, body
        assert body["version"] == version
        assert body["prediccion"] in {"Yes", "No"}
        assert body["customerID"] == fila["customerID"]


@pytest.mark.skipif(
    sklearn.__version__ != _scikit_learn_fijado(),
    reason="Los modelos entregados se guardaron con la versión de scikit-learn fijada en requirements.txt.",
)
def test_la_api_reproduce_la_prediccion_registrada_en_la_fase_2():
    entregados = Settings(FASE3 / "models", "v1", FASE3.parent / "fase-1" / "DatosCrudos.csv")
    with open(FASE3 / "data" / "ejemplo_prediccion.csv", newline="", encoding="utf-8") as handle:
        fila = next(csv.DictReader(handle))
    body = create_app(entregados).test_client().post("/predict", json=cliente_prueba.fila_a_registro(fila)).get_json()
    assert body["prediccion"] == "No"
    assert body["probabilidad_churn"] == pytest.approx(0.006983, abs=1e-6)