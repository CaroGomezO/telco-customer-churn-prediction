"""Métricas de operación de la API a partir del registro de /predict (Fase 4).

Muestra el total de solicitudes y el porcentaje de solicitudes fallidas
(respuestas HTTP 4xx o 5xx). Si el registro está vacío o no existe, muestra
total 0 y «sin solicitudes» para el porcentaje.

Uso:
    python metricas.py                         # usa LOG_PATH o logs/predicciones.csv
    python metricas.py --log muestras/registro_prueba.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from registro import COLUMNAS

DEFAULT_LOG = os.environ.get("LOG_PATH", "logs/predicciones.csv")


def es_fallida(estado: int) -> bool:
    return 400 <= estado <= 599


def calcular(ruta: Path) -> dict:
    """Cuenta las filas de datos del registro y las que tienen un código entre 400 y 599."""
    total = fallidas = 0
    if ruta.is_file():
        with open(ruta, newline="", encoding="utf-8") as handle:
            lector = csv.DictReader(handle)
            if lector.fieldnames is not None and "estado_http" not in lector.fieldnames:
                raise ValueError(
                    f"{ruta} no tiene la columna 'estado_http' (columnas esperadas: {', '.join(COLUMNAS)})."
                )
            for numero, fila in enumerate(lector, start=2):
                try:
                    estado = int(fila["estado_http"])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{ruta}, línea {numero}: estado_http no es un código HTTP ({fila['estado_http']!r})."
                    ) from None
                total += 1
                fallidas += es_fallida(estado)
    porcentaje = None if total == 0 else 100 * fallidas / total
    return {"total": total, "fallidas": fallidas, "porcentaje_fallos": porcentaje}


def formatear_porcentaje(porcentaje: float | None) -> str:
    if porcentaje is None:
        return "sin solicitudes"
    return f"{porcentaje:.1f}".replace(".", ",") + " %"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Total de solicitudes y porcentaje de fallos de /predict.")
    parser.add_argument("--log", type=Path, default=Path(DEFAULT_LOG), help="CSV del registro de solicitudes.")
    args = parser.parse_args(argv)

    try:
        resultado = calcular(args.log)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Registro: {args.log}" + ("" if args.log.is_file() else " (no existe todavía)"))
    print(f"Total de solicitudes: {resultado['total']}")
    if resultado["total"]:
        print(f"Solicitudes fallidas (HTTP 4xx o 5xx): {resultado['fallidas']}")
    print(f"Porcentaje de fallos: {formatear_porcentaje(resultado['porcentaje_fallos'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
