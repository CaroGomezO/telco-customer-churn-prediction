from __future__ import annotations

import argparse
import sys

from inference import LABEL_COLUMN, PREDICTION_COLUMN, PROBABILITY_COLUMN, predict_file

EJEMPLO = (
    "ejemplo:\n"
    "  python predict.py --model models/v1/modelo.joblib "
    "--input data/ejemplo_prediccion.csv --output outputs/predicciones.csv\n\n"
    "El CSV de salida trae las columnas: customerID (si venía en la entrada), "
    f"{PREDICTION_COLUMN} (1 = Yes, el cliente se va; 0 = No, se queda), "
    f"{LABEL_COLUMN} (Yes/No) y {PROBABILITY_COLUMN} (probabilidad entre 0 y 1 de que se vaya).\n"
    "Códigos de salida: 0 = éxito, 1 = error previsible (mensaje claro y sin archivo de salida), "
    "2 = argumentos inválidos."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Genera predicciones de churn con una versión guardada del modelo.",
        epilog=EJEMPLO,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Ruta del modelo (models/v1/modelo.joblib) o de la carpeta de la versión (models/v1).")
    parser.add_argument("--input", required=True, help="CSV con las variables de entrada (la variable objetivo no es necesaria).")
    parser.add_argument("--output", required=True, help="CSV de salida con las predicciones (se crea, junto con su carpeta, si no existe).")
    return parser


def print_summary(result: dict) -> None:
    print(f"Predicción finalizada: {result['filas']} filas procesadas.")
    print(f"Modelo: {result['modelo']}")
    resumen = " | ".join(f"{etiqueta}: {n}" for etiqueta, n in result["conteo"].items())
    print(f"Resultado -> {resumen}")
    if result["nulos_imputados"]:
        detalle = ", ".join(f"{col}={n}" for col, n in result["nulos_imputados"].items())
        print(f"Celdas vacías imputadas con lo aprendido en el entrenamiento: {detalle}")
    for column, values in result["categorias_no_vistas"].items():
        print(f"Aviso: '{column}' trae categorías que el modelo no vio al entrenar ({', '.join(values)}); se codifican con ceros.", file=sys.stderr)
    print(f"Archivo de salida: {result['salida']}")
    print(f"Columnas: {PREDICTION_COLUMN} (1 = {result['etiquetas']['1']}, 0 = {result['etiquetas']['0']}), {LABEL_COLUMN}, {PROBABILITY_COLUMN}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = predict_file(args.model, args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())