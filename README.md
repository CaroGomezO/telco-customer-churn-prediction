# 📡 Telco Customer Churn Prediction

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Pytest](https://img.shields.io/badge/Tests-Pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)

> Proyecto del curso **Modelos y Simulación de Sistemas I** — Universidad de Antioquia (UdeA).

Modelo de Machine Learning supervisado para identificar, de forma temprana, a los clientes de una telco con alta probabilidad de cancelar su servicio (*churn*), para que el área de fidelización actúe antes de que se vayan.

## Integrantes del Equipo

* **Carolina Gómez Osorno** — [@CaroGomezO](https://github.com/CaroGomezO)
* **Julian Esteban Hurtado Serna** — [@Julian-debug-s](https://github.com/Julian-debug-s)
* **Emanuel López Franco** — [@ema28pro](https://github.com/ema28pro)

## Descripción del Problema

La deserción de clientes representa uno de los mayores impactos sobre los ingresos de una empresa de telecomunicaciones, y retener a un cliente cuesta bastante menos que captar uno nuevo. El objetivo es clasificar binariamente si un cliente cancelará (`Churn = Yes`) o no (`Churn = No`), priorizando el **Recall** sin degradar excesivamente la **Precisión**, y evaluando con **F1-score** y **ROC-AUC**.

**Datos:** dataset público *Telco Customer Churn* (IBM / [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)), con 7,043 clientes de una empresa ficticia. Los resultados son metodológicos: no son evidencia sobre clientes reales de ningún operador.

## Estructura del Repositorio

```
telco-customer-churn-prediction/
├── fase-1/   # Análisis exploratorio, modelado y selección del modelo (notebook)
├── fase-2/   # Scripts de entrenamiento y predicción, pruebas y Docker
├── fase-3/   # API REST (Flask) para predecir y entrenar mediante solicitudes HTTP
└── README.md
```

| Fase | Contenido | Documentación |
|---|---|---|
| **Fase 1** | EDA, limpieza, comparación de algoritmos, análisis de umbral y modelo final en notebook. | [`fase-1/README.md`](./fase-1/README.md) |
| **Fase 2** | Aplicación reproducible: `train.py`, `predict.py`, versionamiento de modelos, pruebas con pytest y Dockerfile. | [`fase-2/README.md`](./fase-2/README.md) |
| **Fase 3** | API REST: `GET /health`, `POST /predict` y `POST /train`, con validación de entradas, pruebas, cliente de prueba y Docker. | [`fase-3/README.md`](./fase-3/README.md) |

Cada fase tiene su propio `requirements.txt` y sus instrucciones de ejecución. Los comandos deben ejecutarse desde la carpeta de la fase correspondiente.

## Requisitos Generales

* **Python 3.11**
* Git
* Docker Desktop (opcional, solo para la ejecución en contenedor de las Fases 2 y 3)

## Inicio Rápido

Clona el repositorio e instala las dependencias de la aplicación (Fase 2) para realizar predicciones de inmediato:

```bash
# 1. Clonar e ingresar a la carpeta de la aplicación
git clone https://github.com/CaroGomezO/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction/fase-2

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate       # En Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Ejecutar predicción de prueba con un modelo ya versionado
python predict.py --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

Para levantar la API REST (Fase 3) con la misma versión del modelo:

```bash
cd ../fase-3
python -m pip install -r requirements.txt
python api.py                                   # API en http://localhost:8000
python cliente_prueba.py                        # en otra terminal: /health, /predict válido y /predict inválido
```

Para consultar la guía detallada de cada etapa:

* **API REST, validación de entradas y Docker (Fase 3):** consulta [`fase-3/README.md`](./fase-3/README.md).
* **Entrenar nuevos modelos, pruebas y Docker (Fase 2):** consulta [`fase-2/README.md`](./fase-2/README.md).
* **Revisar el análisis exploratorio y selección de modelos en notebook (Fase 1):** consulta [`fase-1/README.md`](./fase-1/README.md).

## Resumen de Resultados

* **Modelo final:** `GradientBoostingClassifier` dentro de un `Pipeline` de scikit-learn que incluye el preprocesamiento (imputación, escalamiento y One-Hot Encoding).
* **Modelo base (`DummyClassifier`):** F1 = 0.00 y Recall = 0.00.
* **Modelo final en prueba:** F1 ≈ 0.58 y ROC-AUC ≈ 0.84.
* **Variables más determinantes:** contrato mes a mes, baja antigüedad (`tenure`) y cargos mensuales elevados.

El umbral de 0.5 no es necesariamente el óptimo para el negocio: según el criterio (F1 o costo de falsos negativos), el análisis de la Fase 1 sugiere umbrales distintos (0.30 o 0.15).

## Limitaciones

* Los datos describen una empresa ficticia en un corte transversal; no hay historial por cliente.
* El modelo clasifica perfiles a partir de un corte transversal; no se puede verificar que prediga el futuro.
* Los patrones aprendidos podrían no cumplirse en otro mercado. Para un uso real, habría que reentrenar y validar con datos propios del operador.

## Licencia y Fuentes

Proyecto académico. Dataset: IBM Cognos Analytics (Telco Customer Churn), disponible públicamente en Kaggle.