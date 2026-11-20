"""Pruebas de la Fase 4: registro de /predict y métricas de operación."""
from __future__ import annotations

import csv
import dataclasses
import threading
from datetime import datetime

import pytest

import metricas
from api import create_app
from registro import COLUMNAS


@pytest.fixture()
def ruta_log(tmp_path):
    return tmp_path / "logs" / "predicciones.csv"


@pytest.fixture()
def settings_log(settings, ruta_log):
    return dataclasses.replace(settings, log_path=ruta_log)


@pytest.fixture()
def client_log(settings_log):
    return create_app(settings_log).test_client()


def leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as handle:
        lineas = handle.read().splitlines()
    with open(ruta, newline="", encoding="utf-8") as handle:
        return lineas[0], list(csv.DictReader(handle))


def test_ensayo_del_enunciado_dos_validas_y_una_sin_variable(client_log, ruta_log, registro_valido):
    assert client_log.post("/predict", json=registro_valido).status_code == 200
    assert client_log.post("/predict", json=registro_valido).status_code == 200
    incompleto = {k: v for k, v in registro_valido.items() if k != "tenure"}
    assert client_log.post("/predict", json=incompleto).status_code == 422

    encabezado, filas = leer(ruta_log)
    assert encabezado == ",".join(COLUMNAS)
    assert len(filas) == 3, "cada intento debe generar exactamente una fila"
    assert [f["estado_http"] for f in filas] == ["200", "200", "422"]
    assert [f["version"] for f in filas] == ["v1", "v1", ""]
    assert filas[0]["prediccion"] in {"Yes", "No"}
    assert filas[2]["prediccion"] == ""
    for fila in filas:
        datetime.fromisoformat(fila["fecha_hora"])

    resultado = metricas.calcular(ruta_log)
    assert (resultado["total"], resultado["fallidas"]) == (3, 1)
    assert metricas.formatear_porcentaje(resultado["porcentaje_fallos"]) == "33,3 %"


def test_la_fila_guarda_la_prediccion_y_la_version_de_la_respuesta(settings_log, ruta_log, registro_valido):
    v2 = dataclasses.replace(settings_log, model_version="v2")
    body = create_app(v2).test_client().post("/predict", json=registro_valido).get_json()
    _, filas = leer(ruta_log)
    assert filas == [{**filas[0], "estado_http": "200", "version": "v2", "prediccion": body["prediccion"]}]


@pytest.mark.parametrize(
    "kwargs, estado",
    [
        ({"data": "esto no es json", "content_type": "application/json"}, 400),
        ({"json": [1, 2, 3]}, 400),
        ({"json": {"gender": "Male"}}, 422),
    ],
)
def test_las_solicitudes_rechazadas_quedan_registradas_sin_version(client_log, ruta_log, kwargs, estado):
    assert client_log.post("/predict", **kwargs).status_code == estado
    _, filas = leer(ruta_log)
    assert len(filas) == 1
    assert (filas[0]["estado_http"], filas[0]["version"], filas[0]["prediccion"]) == (str(estado), "", "")


def test_modelo_no_disponible_se_registra_como_503_sin_version(settings_log, ruta_log, registro_valido):
    sin_modelo = dataclasses.replace(settings_log, model_version="v99")
    assert create_app(sin_modelo).test_client().post("/predict", json=registro_valido).status_code == 503
    _, filas = leer(ruta_log)
    assert [(f["estado_http"], f["version"], f["prediccion"]) for f in filas] == [("503", "", "")]


def test_metodo_no_permitido_en_predict_tambien_se_registra(client_log, ruta_log):
    assert client_log.get("/predict").status_code == 405
    _, filas = leer(ruta_log)
    assert [f["estado_http"] for f in filas] == ["405"]


def test_las_solicitudes_options_a_predict_no_se_registran(client_log, ruta_log):
    assert client_log.options("/predict").status_code == 200
    assert not ruta_log.exists()


def test_error_interno_en_predict_se_registra_como_500(settings_log, ruta_log, registro_valido, monkeypatch):
    import api

    def falla(*_):
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(api, "predict_record", falla)
    assert create_app(settings_log).test_client().post("/predict", json=registro_valido).status_code == 500
    _, filas = leer(ruta_log)
    assert [(f["estado_http"], f["version"], f["prediccion"]) for f in filas] == [("500", "", "")]


def test_solo_se_registran_las_llamadas_a_predict(client_log, ruta_log):
    client_log.get("/health")
    client_log.get("/no-existe")
    client_log.get("/train")  # 405 en otra ruta
    assert not ruta_log.exists()


def test_el_registro_se_conserva_entre_ejecuciones_y_el_encabezado_no_se_repite(
    settings_log, ruta_log, registro_valido
):
    create_app(settings_log).test_client().post("/predict", json=registro_valido)
    # Segunda "ejecución": una aplicación nueva sobre el mismo archivo.
    create_app(settings_log).test_client().post("/predict", json={})
    with open(ruta_log, encoding="utf-8") as handle:
        lineas = handle.read().splitlines()
    assert lineas.count(",".join(COLUMNAS)) == 1
    assert len(lineas) == 3


def test_solicitudes_simultaneas_generan_una_fila_cada_una(settings_log, ruta_log, registro_valido):
    app = create_app(settings_log)

    def enviar():
        app.test_client().post("/predict", json=registro_valido)

    hilos = [threading.Thread(target=enviar) for _ in range(20)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()
    _, filas = leer(ruta_log)
    assert len(filas) == 20
    assert all(f["estado_http"] == "200" for f in filas)


def test_sin_log_path_la_api_no_crea_registro(settings):
    assert settings.log_path is None
    assert create_app(settings).extensions["churn"]["registro"] is None


# ---------- metricas.py ----------


@pytest.mark.parametrize("contenido", [None, "", ",".join(COLUMNAS) + "\n"])
def test_metricas_registro_vacio_muestra_cero_y_sin_solicitudes(tmp_path, capsys, contenido):
    ruta = tmp_path / "registro.csv"
    if contenido is not None:
        ruta.write_text(contenido, encoding="utf-8")
    assert metricas.main(["--log", str(ruta)]) == 0
    salida = capsys.readouterr().out
    assert "Total de solicitudes: 0" in salida
    assert "Porcentaje de fallos: sin solicitudes" in salida


def test_metricas_cuenta_4xx_y_5xx_como_fallidas(tmp_path, capsys):
    ruta = tmp_path / "registro.csv"
    filas = [
        ("2026-11-18T10:00:00-05:00", 200, "v3", "No"),
        ("2026-11-18T10:01:00-05:00", 201, "", ""),
        ("2026-11-18T10:02:00-05:00", 399, "", ""),
        ("2026-11-18T10:03:00-05:00", 400, "", ""),
        ("2026-11-18T10:04:00-05:00", 422, "", ""),
        ("2026-11-18T10:05:00-05:00", 503, "", ""),
        ("2026-11-18T10:06:00-05:00", 599, "", ""),
        ("2026-11-18T10:07:00-05:00", 600, "", ""),
    ]
    with open(ruta, "w", newline="", encoding="utf-8") as handle:
        escritor = csv.writer(handle)
        escritor.writerow(COLUMNAS)
        escritor.writerows(filas)
    assert metricas.calcular(ruta) == {"total": 8, "fallidas": 4, "porcentaje_fallos": 50.0}
    metricas.main(["--log", str(ruta)])
    salida = capsys.readouterr().out
    assert "Total de solicitudes: 8" in salida
    assert "Porcentaje de fallos: 50,0 %" in salida


def test_metricas_rechaza_un_registro_con_formato_incorrecto(tmp_path, capsys):
    ruta = tmp_path / "registro.csv"
    ruta.write_text("a,b\n1,2\n", encoding="utf-8")
    assert metricas.main(["--log", str(ruta)]) == 1
    assert "estado_http" in capsys.readouterr().err
    ruta.write_text(",".join(COLUMNAS) + "\n2026-11-18T10:00:00,abc,,\n", encoding="utf-8")
    assert metricas.main(["--log", str(ruta)]) == 1


@pytest.mark.parametrize(
    "nombre, total, fallidas, porcentaje",
    [
        ("registro_prueba.csv", 3, 1, "33,3 %"),  # ensayo local (Reporte, ensayo 1)
        ("registro_docker.csv", 8, 2, "25,0 %"),  # ensayo en Docker (Reporte, ensayo 2)
    ],
)
def test_las_muestras_entregadas_coinciden_con_el_reporte(nombre, total, fallidas, porcentaje):
    from pathlib import Path

    muestra = Path(__file__).resolve().parents[1] / "muestras" / nombre
    encabezado, filas = leer(muestra)
    assert encabezado == ",".join(COLUMNAS)
    for fila in filas:
        datetime.fromisoformat(fila["fecha_hora"])
        exitosa = fila["estado_http"] == "200"
        assert bool(fila["version"]) == exitosa and bool(fila["prediccion"]) == exitosa
    resultado = metricas.calcular(muestra)
    assert (resultado["total"], resultado["fallidas"]) == (total, fallidas)
    assert metricas.formatear_porcentaje(resultado["porcentaje_fallos"]) == porcentaje
    