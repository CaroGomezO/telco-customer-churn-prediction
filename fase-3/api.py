from __future__ import annotations

import os
import threading

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from config import Settings
from service import (
    ModelStore,
    ModelUnavailableError,
    TrainingInProgressError,
    predict_record,
    train_new_version,
)
from validation import extract_identifier, problems_message, validate_record

MAX_BODY_BYTES = 1_000_000

HTTP_MESSAGES = {
    404: "La ruta solicitada no existe. Rutas disponibles: GET /health, POST /predict, POST /train.",
    405: "Método no permitido para esta ruta. Use GET /health, POST /predict o POST /train.",
    413: "El cuerpo de la solicitud es demasiado grande.",
}


HTTP_ERROR_CODES = {404: "ruta_no_encontrada", 405: "metodo_no_permitido", 413: "cuerpo_demasiado_grande"}


def error_response(status: int, error: str, message: str, **extra):
    response = jsonify({"error": error, "mensaje": message, **extra})
    response.status_code = status
    return response


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.from_env()
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.json.ensure_ascii = False
    app.json.sort_keys = False

    store = ModelStore(settings.models_dir, settings.model_version)
    train_lock = threading.Lock()
    app.extensions["churn"] = {"settings": settings, "store": store, "train_lock": train_lock}

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "servicio": "churn-api", **store.status()})

    @app.post("/predict")
    def predict():
        payload = request.get_json(force=True, silent=True)
        if not isinstance(payload, dict):
            return error_response(
                400,
                "json_invalido",
                "El cuerpo debe ser un JSON válido con un objeto que contenga las variables predictoras.",
            )
        try:
            bundle = store.get()
        except ModelUnavailableError as exc:
            return error_response(503, "modelo_no_disponible", str(exc), **store.status())

        record, problems = validate_record(payload, bundle["feature_columns"])
        if problems:
            return error_response(
                422, "entrada_invalida", problems_message(problems), detalles=problems
            )
        result = predict_record(bundle, record)
        body = {
            "prediccion": result["prediccion"],
            "probabilidad_churn": result["probabilidad_churn"],
            "version": store.version,
            "nulos_imputados": result["nulos_imputados"],
            "categorias_no_vistas": result["categorias_no_vistas"],
        }
        identifier = extract_identifier(payload)
        if identifier is not None:
            body["customerID"] = identifier
        return jsonify(body), 200

    @app.post("/train")
    def train():
        try:
            outcome = train_new_version(settings, train_lock)
        except TrainingInProgressError as exc:
            return error_response(409, "entrenamiento_en_curso", str(exc))
        except FileNotFoundError as exc:
            return error_response(503, "datos_no_disponibles", str(exc))
        except (ValueError, RuntimeError) as exc:
            app.logger.exception("Falló el entrenamiento")
            return error_response(500, "entrenamiento_fallido", str(exc))
        outcome["archivos"] = ["modelo.joblib", "metrics.json", "metadata.json"]
        outcome["version_activa"] = store.version
        outcome["mensaje"] = (
            f"Versión {outcome['version']} guardada. La API sigue prediciendo con {store.version}; "
            f"para usar la nueva, configure MODEL_VERSION={outcome['version']} y reinicie la API."
        )
        return jsonify(outcome), 201

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        message = HTTP_MESSAGES.get(exc.code, exc.description)
        response = error_response(exc.code, HTTP_ERROR_CODES.get(exc.code, f"http_{exc.code}"), message)
        if exc.code == 405 and getattr(exc, "valid_methods", None):
            response.headers["Allow"] = ", ".join(exc.valid_methods)
        return response

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        app.logger.exception("Error inesperado")
        return error_response(500, "error_interno", "Ocurrió un error interno al procesar la solicitud.")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=False,
    )