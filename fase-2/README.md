# Fase 2: Entrenamiento, Predicción y Despliegue del Modelo de Churn

> Aplicación reproducible para entrenar, versionar y desplegar el modelo de predicción de churn desarrollado en la Fase 1.

## Integrantes

* **Carolina Gómez Osorno** — [@CaroGomezO](https://github.com/CaroGomezO)
* **Julian Esteban Hurtado Serna** — [@Julian-debug-s](https://github.com/Julian-debug-s)
* **Emanuel López Franco** — [@ema28pro](https://github.com/ema28pro)

Proyecto Integrador — Modelos y Simulación de Sistemas I (2026 II).

## Descripción

**Problema:** predecir si un cliente de una empresa de telecomunicaciones abandonará el servicio (clasificación binaria; variable objetivo `Churn`).

**Relación con la Fase 1:** se conservan el problema, los datos (`../fase-1/DatosCrudos.csv`) y el enfoque de modelado (Gradient Boosting) de la carpeta `fase-1/`, y se convierten en una aplicación con scripts, versionamiento de modelos, pruebas y Docker.

Esta fase implementa el flujo de producción del modelo de churn:

1. **Entrenamiento** (`train.py`): entrena un `Pipeline` de scikit-learn y guarda una versión del modelo.
2. **Predicción** (`predict.py`): carga una versión guardada y predice sobre datos nuevos (datos crudos, sin preprocesar).
3. **Pruebas** (`tests/`): valida el pipeline, la inferencia y el contrato de entrada.
4. **Docker**: empaqueta todo en una imagen reproducible.

El `Pipeline` de scikit-learn conserva el preprocesamiento junto con el modelo. Se mantiene el tratamiento de Fase 1:
- `SeniorCitizen` se convierte de `0/1` a `No/Yes` y pasa por one-hot.
- `TotalCharges` se excluye (multicolinealidad).
- `customerID` se trata como identificador.

Como corrección de robustez de Fase 2, el pipeline incorpora imputación aprendida **solo con el conjunto de entrenamiento**: mediana para variables numéricas y moda para categóricas. Esta estrategia permite procesar valores faltantes sin fuga de información.

El modelo usa explícitamente `GradientBoostingClassifier` con `learning_rate=0.1`, `max_depth=3` y `n_estimators=100`, equivalentes a la configuración seleccionada en Fase 1. Estos valores también se guardan en `metadata.json`.

---

## Alcance de los datos

Los datos son el conjunto de ejemplo *Telco Customer Churn* de IBM, que describe una empresa de telecomunicaciones **ficticia** (ver «Alcance y Limitaciones de los Resultados» en `fase-1/README.md`). Las métricas de este modelo describen su desempeño sobre ese conjunto de ejemplo y **no** son evidencia sobre clientes reales de operadores colombianos ni de ninguna empresa real.

La etiqueta `Churn` es una *foto* (corte transversal) cuyo horizonte es «el último mes» según la documentación de IBM, con una ambigüedad documentada en la Fase 1. La aplicación *clasifica* clientes a partir de su perfil; con estos datos no se puede verificar que prediga el futuro.

---

## Preparar el entorno

Desde `fase-2`:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Linux / macOS (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Python y versiones.** Se usa **Python 3.11** (el mismo del `Dockerfile`; probado con 3.11.18). Las versiones exactas están en `requirements.txt`: `numpy 1.26.4`, `pandas 2.2.2`, `scipy 1.13.1`, `scikit-learn 1.5.1`, `joblib 1.4.2` y `pytest 8.3.2`. Los modelos de `models/` se guardaron con estas versiones; cargarlos con otra versión de scikit-learn o de numpy puede dar avisos o fallar.

El CSV etiquetado usado en el ejemplo está en `../fase-1/DatosCrudos.csv` (Linux/macOS) o `..\fase-1\DatosCrudos.csv` (Windows).

---

## Entrenar dos versiones

**Windows (PowerShell):**
```powershell
python train.py --data ..\fase-1\DatosCrudos.csv --target Churn --output-dir models --version v1 --seed 42
python train.py --data ..\fase-1\DatosCrudos.csv --target Churn --output-dir models --version v2 --seed 7
```

**Linux / macOS (Bash):**
```bash
python train.py --data ../fase-1/DatosCrudos.csv --target Churn --output-dir models --version v1 --seed 42
python train.py --data ../fase-1/DatosCrudos.csv --target Churn --output-dir models --version v2 --seed 7
```

Cada carpeta contiene `modelo.joblib`, `metrics.json` y `metadata.json`. Una versión existente **nunca se sobrescribe**.

---

## Pruebas

**Windows (PowerShell):**
```powershell
python -m pytest
```

**Linux / macOS (Bash):**
```bash
python -m pytest
```

---

## Predicción (`predict.py`)

`predict.py` carga una versión guardada y genera una predicción por cada fila de un CSV nuevo. **No entrena ni reajusta nada**: la imputación, el escalamiento y el One-Hot vienen guardados dentro de `modelo.joblib`, junto con el orden de las columnas, el nombre de la variable objetivo y el significado de las etiquetas. La lógica está en `inference.py` y reutiliza la lectura y limpieza de `pipeline.py`, de modo que los datos nuevos se procesan igual que en el entrenamiento.

### Ejecutar la predicción

Desde `fase-2`, se puede indicar el archivo `.joblib` o la carpeta de la versión (`models/v1` o `models\v1` según el sistema operativo).

**Windows (PowerShell):**
```powershell
python predict.py --model models\v1\modelo.joblib --input data\ejemplo_prediccion.csv --output outputs\predicciones.csv
python predict.py --model models\v2 --input data\ejemplo_prediccion.csv --output outputs\predicciones_v2.csv
python predict.py --help
```

**Linux / macOS (Bash):**
```bash
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
python predict.py --model models/v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv
python predict.py --help
```

La carpeta `outputs/` se crea sola si no existe. El archivo de ejemplo (`data/ejemplo_prediccion.csv`, 8 clientes reales sin la columna `Churn`) tiene el esquema completo del conjunto de datos.

### Esquema del CSV de entrada

| Columna | Tipo | Valores admitidos |
|---|---|---|
| `gender` | texto | `Female`, `Male` |
| `SeniorCitizen` | número | `0` o `1` |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | texto | `Yes`, `No` |
| `tenure` | número entero | meses como cliente (0 o más) |
| `MultipleLines` | texto | `Yes`, `No`, `No phone service` |
| `InternetService` | texto | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | texto | `Yes`, `No`, `No internet service` |
| `Contract` | texto | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | texto | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `MonthlyCharges` | número | cargo mensual |

Opcionales: `customerID` (se copia a la salida para identificar cada fila) y `TotalCharges` (se ignora: se excluyó del modelo por multicolinealidad). La variable objetivo `Churn` no es necesaria para predecir.

### CSV de salida

| Columna | Significado |
|---|---|
| `customerID` | Solo si venía en la entrada. |
| `prediccion` | `1` = el cliente se va (churn), `0` = se queda. |
| `etiqueta` | `Yes` (se va) o `No` (se queda). |
| `probabilidad_churn` | Probabilidad entre 0 y 1 de que el cliente se vaya. |

Hay una fila de salida por cada fila de entrada y en el mismo orden; nunca se descartan filas en silencio.

### Qué pasa con los casos especiales

- **Orden de columnas:** no importa, se ordenan según el esquema del entrenamiento.
- **Columnas adicionales:** se ignoran.
- **Celdas vacías:** se imputan con la mediana (numéricas) o la moda (categóricas) aprendidas en el entrenamiento, y el resumen indica cuántas fueron.
- **Categorías no vistas en el entrenamiento:** no producen error, se codifican con ceros y el script lo avisa por `stderr`.
- **Columna obligatoria ausente o texto en una columna numérica:** error claro.
- **Errores previsibles** (modelo o CSV inexistente, archivo dañado, columna ausente, tipo incompatible, CSV separado por punto y coma): mensaje `Error: ...` en `stderr`, código de salida `1` y **no se crea ni se modifica el archivo de salida**. Argumentos inválidos terminan con código `2`.

### Evidencia de ejecución

Salida real del comando ejecutado en Linux con la versión `v3`:

```text
$ python predict.py --model models/v3/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_v3.csv
Predicción finalizada: 8 filas procesadas.
Modelo: models/v3/modelo.joblib
Resultado -> Yes: 2 | No: 6
Archivo de salida: outputs/predicciones_v3.csv
Columnas: prediccion (1 = Yes, 0 = No), etiqueta, probabilidad_churn
```

| customerID | prediccion | etiqueta | probabilidad_churn |
|---|---|---|---|
| 0464-WJTKO | 0 | No | 0.006983 |
| 4282-MSACW | 0 | No | 0.143424 |
| 4094-NSEDU | 1 | Yes | 0.736106 |
| 4472-LVYGI | 0 | No | 0.052893 |
| 8559-WNQZS | 0 | No | 0.064388 |
| 2955-BJZHG | 0 | No | 0.39951 |
| 4385-GZQXV | 1 | Yes | 0.534434 |
| 9402-ROUMJ | 0 | No | 0.111843 |

Las pruebas de predicción están en `tests/test_predict.py` (se ejecutan con `python -m pytest`).

---

## Contenerización con Docker

### 1. Construcción de la imagen

La imagen se construye a partir de `python:3.11-slim-bookworm`, utilizando Python 3.11 e instalando las dependencias especificadas en `requirements.txt`.

Desde el directorio `fase-2`, ejecutar:

```bash
docker build -t modelos-fase2:1.0 .
```

Para comprobar la versión de Python de la imagen:

```bash
docker run --rm modelos-fase2:1.0 python --version
```

**Resultado obtenido:** `Python 3.11.18`.

### 2. Entrenamiento del modelo

El conjunto de datos de entrenamiento se proporciona mediante un montaje de solo lectura desde la carpeta `fase-1` del proyecto. Los artefactos generados se guardan en `fase-2/models`, de manera que persisten después de eliminar el contenedor.

Antes de ejecutar los comandos, crear las carpetas de resultados si todavía no existen:

```bash
mkdir -p models outputs
```

Ejecutar el entrenamiento:

```bash
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/../fase-1:/data:ro" \
  modelos-fase2:1.0 python train.py \
  --data /data/DatosCrudos.csv \
  --target Churn \
  --output-dir models \
  --version V-docker \
  --seed 42
```

El comando recibe como parámetros la ruta del conjunto de datos, la variable objetivo, el directorio de salida, el identificador de versión y la semilla aleatoria.

**Resultado obtenido:** el entrenamiento finalizó correctamente y generó los siguientes archivos en `models/V-docker/`:

- `modelo.joblib`: modelo entrenado.
- `metrics.json`: métricas de evaluación.
- `metadata.json`: metadatos de la versión.

### 3. Ejecución de predicciones

Para realizar predicciones se utiliza el modelo generado durante el entrenamiento. El directorio de modelos se monta en modo de solo lectura, mientras que el directorio de salida permite escribir los resultados.

```bash
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  modelos-fase2:1.0 python predict.py \
  --model models/V-docker/modelo.joblib \
  --input data/ejemplo_prediccion.csv \
  --output outputs/predicciones_docker.csv
```

**Resultado obtenido:** el proceso finalizó correctamente y procesó 8 filas. El archivo `outputs/predicciones_docker.csv` contiene las predicciones generadas, junto con las etiquetas y probabilidades de churn.

### 4. Persistencia y reproducibilidad

Los datos de entrada y el modelo se proporcionan al contenedor mediante montajes de directorios del equipo anfitrión. Los modelos, las métricas, los metadatos y las predicciones se almacenan fuera del contenedor, por lo que permanecen disponibles después de su eliminación.

La imagen permite ejecutar los scripts de entrenamiento y predicción sin entrenar durante la construcción. Los parámetros de entrada permiten seleccionar los datos, la variable objetivo, la versión del modelo y otros valores de ejecución.

**Entorno de verificación:** Linux, terminal Bash y Docker. La construcción de la imagen, la comprobación de Python y las ejecuciones de entrenamiento y predicción se verificaron manualmente.

### 5. Ejecución en Windows

Los usuarios de Windows pueden ejecutar los comandos anteriores utilizando Docker Desktop con el motor de contenedores iniciado. También pueden utilizar PowerShell, siempre que Docker esté instalado y disponible en la terminal.

Los comandos siguientes son equivalentes a los anteriores y deben ejecutarse desde el directorio `fase-2` en PowerShell.

**Construcción de la imagen y comprobación de Python:**

```powershell
docker build -t modelos-fase2:1.0 .

docker run --rm modelos-fase2:1.0 python --version
```

**Creación de las carpetas necesarias:**

```powershell
New-Item -ItemType Directory -Force -Path models, outputs
```

**Entrenamiento del modelo:**

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/../fase-1:/data:ro" `
  modelos-fase2:1.0 python train.py `
  --data /data/DatosCrudos.csv `
  --target Churn `
  --output-dir models `
  --version V-docker `
  --seed 42
```

**Ejecución de predicciones:**

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/models:/app/models:ro" `
  -v "${PWD}/outputs:/app/outputs" `
  modelos-fase2:1.0 python predict.py `
  --model models/V-docker/modelo.joblib `
  --input data/ejemplo_prediccion.csv `
  --output outputs/predicciones_docker.csv
```

En PowerShell, el acento grave (`` ` ``) permite continuar un comando en la siguiente línea. Debe colocarse al final de cada línea que continúa, sin espacios después de él.

Las rutas de origen de los montajes corresponden a las carpetas del equipo anfitrión, mientras que las rutas de destino corresponden al contenedor. Por tanto, los archivos generados permanecen en el equipo anfitrión después de que el contenedor se elimina.

---

## Versiones del modelo

El directorio `models/` contiene las versiones entrenadas. Cada una tiene `modelo.joblib`, `metrics.json` y `metadata.json`:

| Versión | Descripción | Semilla | F1 | ROC-AUC |
|---|---|---|---|---|
| `v1` | Semilla 42, partición 80/20 estratificada | 42 | 0.579 | 0.843 |
| `v2` | Semilla 7 (otra partición) | 7 | 0.622 | 0.865 |
| `v3` | Semilla 42, mismos parámetros que `v1` | 42 | 0.579 | 0.843 |
| `V-docker` | Semilla 42, entrenada dentro del contenedor (ver «Contenerización con Docker») | 42 | 0.579 | 0.843 |

**Cómo leer esta tabla:**

- `v1`, `v3` y `V-docker` son prácticamente idénticas: mismos datos, parámetros y semilla (el modelo es determinista). Cambia el entorno o el código con el que se generó cada una.
- `v2` usa otra semilla, y eso **cambia la partición** entrenamiento/prueba. Por eso su métrica **no es comparable** con la de las demás: un F1 más alto no significa un modelo mejor.
- Todas se guardaron con Python 3.11 y las versiones de `requirements.txt` (scikit-learn 1.5.1). El ROC-AUC puede cambiar en el 5.º decimal al cambiar de versión de scikit-learn (se midieron diferencias de 0.00001 a 0.00004 entre las versiones 1.5.1 y 1.9.0); el resto de las métricas no cambió.
- Las pruebas de `tests/test_contrato_entrada.py` comprueban que cada versión de `models/` carga en un proceso independiente sin avisos de versión.

---

## Resultados y comparación con el modelo base

Métrica principal: **F1 de la clase `Churn = 1`**. El conjunto está desbalanceado (aproximadamente 73.5 % "No" y 26.5 % "Yes") y un modelo que nunca detecta abandonos igual tendría 73.5 % de accuracy, por eso el accuracy no sirve como criterio. Evaluación: partición 80/20 estratificada (5634 clientes de entrenamiento y 1409 de prueba) con semilla 42. El modelo base predice siempre la clase más frecuente («No»).

| Métrica (prueba) | Modelo base | Modelo (`v1`) |
|---|---|---|
| accuracy | 0.735 | 0.803 |
| precision | 0.000 | 0.668 |
| recall | 0.000 | 0.511 |
| F1 | 0.000 | 0.579 |
| ROC-AUC | 0.500 | 0.843 |

El modelo detecta cerca de la mitad de los clientes que se van (recall 0.511) con una precisión de 0.668; el modelo base no detecta ninguno. Los valores salen de `models/v1/metrics.json` (que guarda también las métricas del modelo base).

---

## Limitaciones conocidas

- **Datos ficticios:** los resultados no son evidencia sobre clientes reales (ver «Alcance de los datos»).
- **Etiqueta y horizonte:** el conjunto es una foto y el horizonte de `Churn` es ambiguo en la documentación de IBM.
- **Una sola partición:** las métricas salen de una partición 80/20 con semilla fija. La comparación entre familias de modelos con validación cruzada está en el notebook de la Fase 1, y allí Gradient Boosting y la regresión logística no se distinguen de forma concluyente.
- **Umbral fijo:** `predict.py` etiqueta con el umbral por defecto de 0.5. El análisis de umbral con costos (ilustrativos) está en el notebook de la Fase 1 y no se incorporó a la aplicación.
- **Contrato de entrada:** `SeniorCitizen` solo admite `0` o `1`; las categorías no vistas en el entrenamiento se codifican con ceros (con aviso) y las columnas adicionales se ignoran.
- **Reproducibilidad:** depende de las versiones fijadas en `requirements.txt`; cargar los modelos con otras versiones puede dar avisos o fallar, y el ROC-AUC puede variar en el 5.º decimal.
- **Seguridad:** `joblib` ejecuta código al cargar un archivo; solo deben cargarse modelos generados por este proyecto.

---

## Correcciones respecto a la entrega anterior

Cambios hechos después de la retroalimentación de la Fase 1:

- **Flujo completo guardado:** la entrega de la Fase 1 guardaba solo el estimador, que recibía 45 columnas ya transformadas. Ahora `modelo.joblib` contiene el `Pipeline` (preprocesamiento + modelo) con el orden de columnas, y `predict.py` recibe las variables originales.
- **Imputación sin fuga:** mediana (numéricas) y moda (categóricas) aprendidas solo con el entrenamiento; antes no había tratamiento de nulos en el flujo guardado.
- **Separación antes de decidir variables (notebook de la Fase 1):** el `train_test_split` pasó al inicio del EDA y la decisión de eliminar `TotalCharges` se tomó solo con entrenamiento (la conclusión no cambió).
- **Preprocesamiento dentro de la validación cruzada (notebook de la Fase 1):** el escalador y el codificador forman parte de un `Pipeline`, de modo que se reajustan en cada pliegue.
- **Umbral con validación y costos ilustrativos** (notebook de la Fase 1).
- **Alcance de los datos y horizonte de la etiqueta:** se documenta que los datos son de una empresa ficticia y que el horizonte de `Churn` es ambiguo.
- **Versiones fijadas:** `requirements.txt` fija todas las librerías (incluidas `numpy` y `scipy`) y `v1` y `v2` se regeneraron con esas versiones, con Python 3.11.
- **Pruebas de carga independiente y del contrato de entrada:** nuevas pruebas, y una celda en el notebook de la Fase 1 que carga el modelo guardado en un proceso de Python nuevo con clientes nuevos y categorías desconocidas.

---

## Fuentes consultadas

- Conjunto de datos *Telco Customer Churn* (muestra de IBM publicada en Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Documentación de IBM Cognos Analytics sobre esta muestra: https://www.ibm.com/docs/SSEP7J_11.1.0/com.ibm.swg.ba.cognos.ig_smples.doc/c_telco_dm_sam.html
- Documentación de scikit-learn sobre `Pipeline` y `ColumnTransformer`: https://scikit-learn.org/stable/modules/compose.html
- Documentación de scikit-learn sobre persistencia de modelos y sus limitaciones: https://scikit-learn.org/stable/model_persistence.html

---

## Estructura del proyecto (Fase 2)

```
fase-2/
├── data/
│   └── ejemplo_prediccion.csv
├── models/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   └── V-docker/
├── outputs/
├── tests/
│   ├── test_pipeline.py
│   ├── test_predict.py
│   └── test_contrato_entrada.py
├── pipeline.py
├── train.py
├── inference.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## Pruebas

El proyecto incluye pruebas automatizadas con `pytest`:

| Archivo | Qué prueba |
|---|---|
| `tests/test_pipeline.py` | Pipeline de entrenamiento: carga de datos, validación de columnas, preprocesamiento, entrenamiento, métricas, guardado y carga. |
| `tests/test_predict.py` | Inferencia con modelos sintéticos: columnas faltantes, tipos incorrectos, categorías desconocidas, celdas vacías, esquema de salida, casos especiales. |
| `tests/test_contrato_entrada.py` | Contrato de entrada sobre los **modelos entregados** en `models/`: las 18 columnas obligatorias (una prueba por columna), valores categóricos que conoce cada versión, `SeniorCitizen`, esquema de salida, clientes nuevos con categorías desconocidas en un proceso independiente, y que cada versión cargue sin avisos de versión de scikit-learn. |

Ejecutar todas las pruebas:

```bash
python -m pytest
```

**Resultado obtenido:** 57 pruebas pasan correctamente (`57 passed`).

---

## Requisitos

- Python 3.11
- Dependencias en `requirements.txt`
- Docker (opcional, para contenerización)