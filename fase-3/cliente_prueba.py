from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "ejemplo_prediccion.csv"
ENTEROS = {"SeniorCitizen", "tenure"}
DECIMALES = {"MonthlyCharges"}
EXCLUIDAS = {"TotalCharges"}  # el modelo no la usa. La API la ignoraría


def fila_a_registro(fila: dict) -> dict:
    registro = {}
    for columna, valor in fila.items():
        if columna in EXCLUIDAS:
            continue
        valor = (valor or "").strip()
        if valor == "":
            registro[columna] = None  # dato faltante
        elif columna in ENTEROS:
            registro[columna] = int(float(valor))
        elif columna in DECIMALES:
            registro[columna] = float(valor)
        else:
            registro[columna] = valor
    return registro


def leer_fila(ruta: Path, indice: int) -> dict:
    with open(ruta, newline="", encoding="utf-8") as handle:
        filas = list(csv.DictReader(handle))
    if not 0 <= indice < len(filas):
        raise SystemExit(f"--fila debe estar entre 0 y {len(filas) - 1} (el CSV tiene {len(filas)} filas).")
    return filas[indice]


def solicitar(metodo: str, url: str, cuerpo: dict | None = None) -> tuple[int, dict]:
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    peticion = urllib.request.Request(
        url, data=datos, method=metodo, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(peticion, timeout=120) as respuesta:
            return respuesta.status, json.loads(respuesta.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def mostrar(titulo: str, estado: int, cuerpo: dict) -> None:
    print(f"\n== {titulo}\nHTTP {estado}\n{json.dumps(cuerpo, indent=2, ensure_ascii=False)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cliente de prueba de la API de churn.")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base de la API.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV, help="CSV con clientes de ejemplo.")
    parser.add_argument("--fila", type=int, default=0, help="Fila del CSV a enviar (desde 0).")
    parser.add_argument("--train", action="store_true", help="Además, llama a POST /train.")
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")
    registro = fila_a_registro(leer_fila(args.input, args.fila))

    try:
        estado, cuerpo = solicitar("GET", f"{base}/health")
        mostrar("GET /health", estado, cuerpo)

        estado_ok, cuerpo_ok = solicitar("POST", f"{base}/predict", registro)
        mostrar("POST /predict (registro válido)", estado_ok, cuerpo_ok)

        incompleto = {k: v for k, v in registro.items() if k != "tenure"}
        estado_mal, cuerpo_mal = solicitar("POST", f"{base}/predict", incompleto)
        mostrar("POST /predict (sin 'tenure', debe rechazarse)", estado_mal, cuerpo_mal)

        if args.train:
            estado_tr, cuerpo_tr = solicitar("POST", f"{base}/train")
            mostrar("POST /train", estado_tr, cuerpo_tr)
            if estado_tr not in (200, 201):
                return 1
    except (urllib.error.URLError, OSError) as exc:
        print(f"Error: no se pudo conectar con la API en {base}: {exc}", file=sys.stderr)
        return 1

    valida = estado_ok == 200 and "prediccion" in cuerpo_ok and "version" in cuerpo_ok
    rechazada = estado_mal in (400, 422)
    print(f"\nResultado: predicción válida={'OK' if valida else 'FALLÓ'} | entrada inválida rechazada={'OK' if rechazada else 'FALLÓ'}")
    return 0 if valida and rechazada else 1


if __name__ == "__main__":
    raise SystemExit(main())