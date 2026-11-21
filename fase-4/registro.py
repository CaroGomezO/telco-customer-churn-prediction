"""Registro de las solicitudes a POST /predict (Fase 4).

Cada llamada a /predict agrega una fila a un CSV con cuatro campos:

    fecha_hora,estado_http,version,prediccion

El archivo se abre en modo agregar para conservar el historial entre ejecuciones
y el encabezado solo se escribe cuando el archivo se crea (o está vacío).
"""
from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

COLUMNAS = ["fecha_hora", "estado_http", "version", "prediccion"]


class RegistroSolicitudes:
    """Agrega una fila por solicitud a un CSV. Es seguro entre los hilos de un mismo proceso."""

    def __init__(self, ruta: Path) -> None:
        self.ruta = Path(ruta)
        self._lock = threading.Lock()

    def registrar(self, estado_http: int, version: str | None = None, prediccion: str | None = None) -> dict:
        fila = {
            "fecha_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
            "estado_http": int(estado_http),
            "version": version or "",
            "prediccion": prediccion or "",
        }
        with self._lock:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ruta, "a", newline="", encoding="utf-8") as handle:
                escritor = csv.DictWriter(handle, fieldnames=COLUMNAS)
                if handle.tell() == 0:
                    escritor.writeheader()
                escritor.writerow(fila)
        return fila
