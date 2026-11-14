# Fase 3: API REST del Modelo de Churn

> API que expone el modelo de la Fase 2 para consultarlo (`POST /predict`), entrenar una versión nueva (`POST /train`) y comprobar que el servicio está activo (`GET /health`).

## Integrantes

* **Carolina Gómez Osorno** — [@CaroGomezO](https://github.com/CaroGomezO)
* **Julian Esteban Hurtado Serna** — [@Julian-debug-s](https://github.com/Julian-debug-s)

Proyecto Integrador — Modelos y Simulación de Sistemas I (2026 II).

## Descripción

**Problema:** predecir si un cliente de una empresa de telecomunicaciones abandonará el servicio (clasificación binaria; variable objetivo `Churn`).

**Relación con las fases anteriores:** se conservan `fase-1/` y `fase-2/`. Esta fase no cambia el modelo ni el preprocesamiento: la API **reutiliza sin modificar** las funciones de la Fase 2.

| Operación de la API | Función de la Fase 2 que reutiliza |
|---|---|
| `POST /predict` | `inference.load_bundle` (carga la versión guardada) e `inference.predict_frame` (mismas transformaciones que `predict.py`) |
| `POST /train` | `pipeline.train_and_save` (la misma función que usa `train.py`) |

El registro JSON recibido se convierte en una tabla de una fila con las columnas en el orden del esquema guardado en la versión del modelo, y pasa por el `Pipeline` guardado (imputación, escalamiento, One-Hot y modelo). **No se reajusta nada al predecir.**

---

## Decisiones

| Decisión | Motivo |
|---|---|
| **Flask** | La validación que se pide (campos obligatorios, tipos, mensajes que nombran el campo) se escribió explícitamente en `validation.py`, por lo que no depende del framework. |
| **`prediccion` devuelve la etiqueta** `"Yes"` / `"No"`, mientras que el CSV de `predict.py` (Fase 2) usa `1` / `0` en `prediccion` y `Yes` / `No` en `etiqueta`. | El enunciado pide devolver la etiqueta para clasificación. El significado es el mismo: `"Yes"` = `1` (el cliente se va). |
| La respuesta incluye además `probabilidad_churn`, `nulos_imputados`, `categorias_no_vistas` y, si se envía, `customerID`. | Son los mismos datos auxiliares que ya entregaba `predict.py` en la Fase 2; permiten saber si la predicción se apoyó en imputación o en una categoría desconocida. Las claves `prediccion` y `version` del enunciado se mantienen. |
| Un campo con valor **`null`** se acepta y se imputa (mediana o moda del entrenamiento). Una **clave ausente** se rechaza. | Es la regla de la Fase 2 (una celda nula no equivale a una columna obligatoria ausente) |
| La validación de valores es **un poco más estricta que la del CSV**: `tenure` y `MonthlyCharges` no pueden ser negativos, `tenure` debe ser entero, no se aceptan números escritos como texto (`"12"`), booleanos ni textos vacíos. | En JSON los tipos son explícitos: el enunciado pide que un número se envíe como número y una categoría como texto. Los límites ya figuran en el esquema documentado de la Fase 2 (`tenure`: entero de 0 o más). |
| Una **categoría no vista** en el entrenamiento **no se rechaza**: se codifica con ceros y se informa en `categorias_no_vistas`. | Es el contrato documentado de la Fase 2 (`OneHotEncoder(handle_unknown="ignore")` con aviso). La API se comporta igual que `predict.py`. |
| Los módulos de la Fase 2 (`pipeline.py`, `inference.py`, `train.py`, `predict.py`) **se copian** a `fase-3/`. | Cada fase debe poder construirse con `docker build` desde su propia carpeta. Para que la copia no se desincronice, la prueba `test_los_modulos_reutilizados_son_identicos_a_los_de_la_fase_2` verifica que sean idénticos a los de `fase-2/`. |
| `fase-3/models/` contiene **copias de `v1` y `v2`** de la Fase 2 (mismos archivos). | Así `POST /train` crea versiones nuevas sin modificar la carpeta entregada en la Fase 2, y la Fase 4 puede usar `v1`, `v2` y la versión creada con `/train`. |
| El contenedor usa **gunicorn** (no el servidor de desarrollo de Flask), con un solo proceso y 4 hilos. | El servidor de desarrollo no es apto para un contenedor. Un solo proceso mantiene en memoria el candado de `/train` y la caché del modelo. |
| **`POST /train` responde `201`**, devuelve `409` si ya hay un entrenamiento en curso y `503` si no encuentra los datos. | Evitar entrenamientos simultáneos impide que dos solicitudes compitan por el mismo identificador. |
| La **versión activa no cambia sola** después de `/train`; se cambia con `MODEL_VERSION` y reiniciando. | Evita activar un modelo sin revisarlo. |
| **Más de tres pruebas** (incluida una automática de `/train`). | El enunciado exige tres; las demás cubren los casos de error y de coherencia con la Fase 2. |
| `HEALTHCHECK` en el `Dockerfile` y `--build-arg GIT_COMMIT` (opcional). | `docker ps` muestra el estado `healthy`. Dentro de la imagen no hay `.git`, y sin el argumento el metadato `git_revision` de las versiones entrenadas desde Docker queda en `no-disponible`. |

---

## Estructura de la carpeta

```
fase-3/
├── api.py                  # Aplicación Flask: rutas GET /health, POST /predict, POST /train
├── validation.py           # Validación del registro JSON (campos obligatorios y tipos)
├── service.py              # Carga de versiones, predicción de un registro, entrenamiento
├── config.py               # Configuración mediante variables de entorno
├── pipeline.py             # (Fase 2, sin cambios) entrenamiento y preprocesamiento
├── inference.py            # (Fase 2, sin cambios) carga del modelo y predicción
├── train.py                # (Fase 2, sin cambios) entrenamiento por línea de comandos
├── predict.py              # (Fase 2, sin cambios) predicción por línea de comandos
├── cliente_prueba.py       # Cliente de prueba (script Python)
├── data/
│   ├── ejemplo_prediccion.csv        # 8 clientes del conjunto, sin la columna Churn
│   ├── ejemplo_solicitud.json        # Primer cliente del CSV, listo para POST /predict
│   └── ejemplo_solicitud_invalida.json  # El mismo cliente sin `tenure` (se rechaza)
├── models/
│   ├── v1/                 # modelo.joblib, metrics.json, metadata.json (copia de la Fase 2)
│   └── v2/                 # ídem
├── tests/
│   └── test_api.py
├── conftest.py             # Fixtures de las pruebas
├── pytest.ini
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## Guía paso a paso: ejecutar y comprobar todo

Esta guía recorre la ejecución completa del proyecto: primero **sin Docker** (pasos 1 a 3) y luego **con Docker** (pasos 4 a 8). Todos los comandos se ejecutan desde la carpeta `fase-3`. Cuando un comando cambia según el sistema, se muestra para **Windows (PowerShell)** y para **Linux / macOS (Bash)**.

| Paso | Qué se hace | Terminales | Requiere Docker |
|---|---|---|---|
| 0 | Requisitos y ubicación en `fase-3` | 1 | No |
| 1 | Entorno virtual e instalación de dependencias | 1 | No |
| 2 | Pruebas automáticas | 1 | No |
| 3 | API sin Docker | 2 | No |
| 4 | Construcción de la imagen y arranque de la API en Docker | 1 | Sí |
| 5 | Prueba de la API que corre en Docker | 1 | Sí |
| 6 | Entrenamiento de una versión nueva (`POST /train`) y persistencia | 1 | Sí |
| 7 | Predicción con la versión nueva | 1 | Sí |
| 8 | Comprobación de los errores (422, 400 y 503) | 1 | Sí |

### Paso 0. Requisitos y ubicación

* **Python 3.11** (versión de referencia, la misma del `Dockerfile`).
* **Docker** con el motor en ejecución (Docker Desktop en Windows y macOS), necesario para los pasos 4 a 8.
* **Git**, opcional: solo se usa para anotar el commit del código en la imagen.
* La carpeta `fase-1/` junto a `fase-3/`: contiene `DatosCrudos.csv`, que usa `POST /train`.

Los comandos se ejecutan desde la carpeta `fase-3` del repositorio:

**Windows (PowerShell):**
```powershell
cd telco-customer-churn-prediction\fase-3
```

**Linux / macOS (Bash):**
```bash
cd telco-customer-churn-prediction/fase-3
```

### Paso 1. Entorno virtual e instalación

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

**Resultado esperado:** el prompt empieza con `(.venv)` y la instalación termina con `Successfully installed ...`.

### Paso 2. Pruebas automáticas

```bash
python -m pytest
```

**Resultado esperado:** `62 passed`, sin pruebas fallidas ni omitidas. No se requiere que la API esté en marcha. Tres de las pruebas cargan los modelos entregados y solo se ejecutan con la versión de scikit-learn fijada en `requirements.txt` (1.5.1); con otra versión se muestran como omitidas (`skipped`).

### Paso 3. API sin Docker (dos terminales)

`python api.py` permanece en ejecución y ocupa su terminal, por eso se usan dos.

**Terminal 1** (con el entorno activado): arranque de la API.
```bash
python api.py
```
**Resultado esperado:** aparece `Running on http://127.0.0.1:8000`.

**Terminal 2** (en `fase-3`): ejecución del cliente de prueba.
```bash
python cliente_prueba.py
```
**Resultado esperado:** al final aparece `Resultado: predicción válida=OK | entrada inválida rechazada=OK`. Antes se muestran:

* `GET /health` con HTTP 200 y `"status": "ok"`.
* `POST /predict` con HTTP 200: cliente `0464-WJTKO`, `"prediccion": "No"`, `"probabilidad_churn": 0.006983` y `"version": "v1"`.
* `POST /predict` sin `tenure` con HTTP 422 y el mensaje `tenure: campo obligatorio ausente`.

Las mismas solicitudes con curl (en PowerShell se escribe `curl.exe`; en Bash, `curl`):
```bash
curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@data/ejemplo_solicitud.json"
```

La API se detiene con `Ctrl+C` en la terminal 1. En este paso no se entrena: `POST /train` se comprueba en el paso 6.

### Paso 4. Construcción de la imagen y arranque en Docker

El motor de Docker debe estar en ejecución y el puerto 8000 libre (la API del paso 3, detenida).

**Windows (PowerShell):**
```powershell
docker build -t modelos-fase3:1.0 --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .

docker run --rm -d --name churn-api -p 8000:8000 `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/../fase-1:/data:ro" `
  modelos-fase3:1.0
```

**Linux / macOS (Bash):**
```bash
docker build -t modelos-fase3:1.0 --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .

docker run --rm -d --name churn-api -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/../fase-1:/data:ro" \
  modelos-fase3:1.0
```

| Parte | Significado |
|---|---|
| `docker build -t modelos-fase3:1.0 .` | Construye la imagen `modelos-fase3:1.0` a partir del `Dockerfile` de esta carpeta (la primera vez tarda unos minutos). |
| `--build-arg GIT_COMMIT=...` | Opcional: anota el commit del código en los `metadata.json` de las versiones entrenadas desde Docker. |
| `-d` | Deja el contenedor en segundo plano. |
| `--rm` | Elimina el contenedor al detenerlo (los archivos de `models/` permanecen). |
| `--name churn-api` | Nombre del contenedor. |
| `-p 8000:8000` | Publica el puerto 8000 del contenedor en el 8000 del equipo. |
| `-v ...models:/app/models` | Monta `fase-3/models` (lectura y escritura): de ahí se leen las versiones y ahí guarda `/train` las nuevas. |
| `-v ...fase-1:/data:ro` | Monta `fase-1` (solo lectura) para que `/train` encuentre `DatosCrudos.csv`. |

Estado del contenedor:
```bash
docker ps
```
**Resultado esperado:** una fila `churn-api` con `(healthy)` en el estado (al inicio puede mostrar `(health: starting)`). Con `(unhealthy)`, `docker logs churn-api` muestra la causa.

### Paso 5. Prueba de la API que corre en Docker

```bash
python cliente_prueba.py
```
**Resultado esperado:** el mismo del paso 3 (`predicción válida=OK | entrada inválida rechazada=OK`, versión `v1`); ahora responde el contenedor. Los registros de la API se ven con `docker logs churn-api`. Para enviar clientes propios en lugar del ejemplo, ver Enviar datos propios a la API.

### Paso 6. Entrenamiento de una versión nueva y persistencia

```bash
python cliente_prueba.py --train
```
**Resultado esperado:** además de lo anterior, aparece `POST /train` con HTTP 201, `"version": "v3"` y `"version_activa": "v1"` (la API sigue prediciendo con `v1` hasta que se cambie `MODEL_VERSION`). Cada ejecución con `--train` crea una versión nueva (`v4`, `v5`, …).

La carpeta nueva se comprueba en el equipo anfitrión:

**Windows (PowerShell):**
```powershell
Get-ChildItem models
Get-Content models\v3\metadata.json
```

**Linux / macOS (Bash):**
```bash
ls models
cat models/v3/metadata.json
```
**Resultado esperado:** `models` contiene `v1`, `v2` y `v3`; el `metadata.json` de `v3` muestra `"version": "v3"`, `"python": "3.11..."` y, si se usó `--build-arg GIT_COMMIT`, el hash del commit en `"git_revision"`.

La persistencia se comprueba deteniendo el contenedor:
```bash
docker stop churn-api
```
**Resultado esperado:** al listar `models` de nuevo, `v3` sigue presente aunque el contenedor ya no existe.

### Paso 7. Predicción con la versión nueva

La versión que usa la API se elige con `MODEL_VERSION` al arrancar el contenedor (`-e MODEL_VERSION=v3`):

**Windows (PowerShell):**
```powershell
docker run --rm -d --name churn-api -p 8000:8000 -e MODEL_VERSION=v3 `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/../fase-1:/data:ro" `
  modelos-fase3:1.0

python cliente_prueba.py
docker stop churn-api
```

**Linux / macOS (Bash):**
```bash
docker run --rm -d --name churn-api -p 8000:8000 -e MODEL_VERSION=v3 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/../fase-1:/data:ro" \
  modelos-fase3:1.0

python cliente_prueba.py
docker stop churn-api
```
**Resultado esperado:** la predicción válida muestra `"version": "v3"`. Con `-e MODEL_VERSION=v2` se usa `v2`.

### Paso 8. Comprobación de los errores

Con el contenedor arrancado como en el paso 4 (versión `v1`):

```bash
# 422: falta "tenure" (el mensaje nombra el campo)
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@data/ejemplo_solicitud_invalida.json"

# 400: el cuerpo no es un JSON válido (se envía un archivo que no es JSON)
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@pytest.ini"
```
(En Bash se escribe `curl` en lugar de `curl.exe`.)

**Resultado esperado:** el primero responde `422` con `entrada_invalida` y el segundo `400` con `json_invalido`.

Para el **503**, el contenedor se arranca con una versión inexistente (`-e MODEL_VERSION=v99`, mismo comando del paso 7) y se envía una solicitud válida:
```bash
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@data/ejemplo_solicitud.json"
```
**Resultado esperado:** `503` con `modelo_no_disponible` y la lista de versiones disponibles.

Al terminar, el contenedor se detiene con `docker stop churn-api`; al haberse iniciado con `--rm`, se elimina y los archivos de `models/` permanecen.

---

## Enviar datos propios a la API

`POST /predict` recibe **un cliente por solicitud**, como un objeto JSON con las 18 variables predictoras, y responde si se va (`Yes`) o se queda (`No`). Funciona igual con la API local (paso 3) y con la de Docker (paso 4): ambas atienden en `http://localhost:8000`. Los valores admitidos de cada variable están en la tabla de `POST /predict` (sección Operaciones de la API).

### Opción 1. Datos escritos directamente

**Windows (PowerShell):** los números quedan como números y los textos como texto.
```powershell
$cliente = @{
  gender="Male"; SeniorCitizen=0; Partner="No"; Dependents="No"; tenure=2
  PhoneService="Yes"; MultipleLines="No"; InternetService="Fiber optic"
  OnlineSecurity="No"; OnlineBackup="No"; DeviceProtection="No"; TechSupport="No"
  StreamingTV="No"; StreamingMovies="No"; Contract="Month-to-month"
  PaperlessBilling="Yes"; PaymentMethod="Electronic check"; MonthlyCharges=85.5
}
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body ($cliente | ConvertTo-Json)
```
Ante un dato inválido, `Invoke-RestMethod` muestra un error 422; el mensaje completo, con el campo que falló, se obtiene con la opción 2.

**Linux / macOS (Bash):**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
  "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5
}'
```

### Opción 2. Archivo JSON editado y enviado con curl

Se parte del ejemplo, se modifican los valores y se envía el archivo (en PowerShell se escribe `curl.exe`; en Bash, `curl`):

**Windows (PowerShell):**
```powershell
Copy-Item data\ejemplo_solicitud.json mi_cliente.json
notepad mi_cliente.json
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@mi_cliente.json"
```

**Linux / macOS (Bash):**
```bash
cp data/ejemplo_solicitud.json mi_cliente.json
nano mi_cliente.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @mi_cliente.json
```

### Opción 3. Clientes propios en un CSV

Un CSV con las mismas columnas que `data/ejemplo_prediccion.csv` permite enviar una de sus filas con el cliente de prueba; `--fila` es el número de la fila, empezando en 0:

```bash
python cliente_prueba.py --input mis_clientes.csv --fila 0
```

Además de la predicción, el cliente repite la solicitud sin `tenure` para comprobar el rechazo; es su comportamiento normal.

Para predecir **muchos clientes a la vez** se usa el script de la Fase 2, que genera un CSV con una predicción por fila:

**Windows (PowerShell):**
```powershell
python predict.py --model models\v1 --input mis_clientes.csv --output outputs\resultado.csv
```

**Linux / macOS (Bash):**
```bash
python predict.py --model models/v1 --input mis_clientes.csv --output outputs/resultado.csv
```

### Reglas de los datos

| Regla | Detalle |
|---|---|
| Deben estar **las 18 claves**. | Si falta una, la API responde 422 y nombra el campo ausente. |
| `tenure` y `SeniorCitizen` son **números sin comillas**. | `SeniorCitizen` solo puede ser `0` o `1`; `tenure` debe ser un entero de 0 o más. |
| `MonthlyCharges` es un **número sin comillas**, con punto decimal. | Por ejemplo `85.5`; no puede ser negativo. |
| Las demás variables son **texto entre comillas**. | Se escriben exactamente como en la tabla de valores admitidos (distinguen mayúsculas, por ejemplo `Month-to-month`). |
| Un dato desconocido se envía como **`null`** (sin comillas). | El modelo lo completa con la mediana (numéricas) o la moda (categóricas) del entrenamiento; la respuesta lo informa en `nulos_imputados`. |
| Un texto que el modelo no vio al entrenar **no se rechaza**. | Se procesa y la respuesta lo informa en `categorias_no_vistas`. |
| `customerID` es **opcional**. | Si se envía, se devuelve en la respuesta; las claves adicionales se ignoran. |

**Resultado esperado:** HTTP 200 con `prediccion` (`Yes` si el cliente se va, `No` si se queda), `probabilidad_churn` (entre 0 y 1) y la `version` del modelo usada. El resultado depende del modelo: por ejemplo, un cliente con contrato mes a mes, pocos meses de antigüedad y cargos altos tiende a tener una probabilidad alta de irse.

---

## Preparar el entorno

Desde `fase-3`. Se usa **Python 3.11** (el mismo del `Dockerfile`). Versiones exactas en `requirements.txt`: `numpy 1.26.4`, `pandas 2.2.2`, `scipy 1.13.1`, `scikit-learn 1.5.1`, `joblib 1.4.2`, `Flask 3.1.3`, `gunicorn 23.0.0` y `pytest 8.3.2`. Los modelos de `models/` se guardaron con estas versiones; cargarlos con otra versión de scikit-learn o numpy puede dar avisos o fallar.

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

---

## Configuración (variables de entorno)

| Variable | Valor por defecto | Para qué sirve |
|---|---|---|
| `MODEL_VERSION` | `v1` | **Versión que usa `POST /predict`.** Debe existir en `MODELS_DIR`. |
| `MODELS_DIR` | `models` | Carpeta con las versiones (`models/v1/modelo.joblib`, …). Aquí también guarda `POST /train` las versiones nuevas. |
| `TRAIN_DATA_PATH` | `../fase-1/DatosCrudos.csv` (en Docker: `/data/DatosCrudos.csv`) | CSV etiquetado que usa `POST /train`. |
| `TARGET` | `Churn` | Variable objetivo para `POST /train`. |
| `SEED` | `42` | Semilla de `POST /train`. |
| `TEST_SIZE` | `0.2` | Proporción de prueba de `POST /train` (partición estratificada, como en la Fase 2). |
| `GIT_COMMIT` | vacío | Commit del código; se anota en `metadata.json` si no hay `.git` (Docker). |
| `HOST`, `PORT` | `127.0.0.1`, `8000` | Solo para `python api.py` (en Docker se usa gunicorn en `0.0.0.0:8000`). |

### Cómo seleccionar una versión disponible

Las versiones disponibles se ven en `GET /health` (campo `versiones_disponibles`) o listando las carpetas de `models/`. Para usar otra, se define `MODEL_VERSION` **antes de arrancar** la API:

**Windows (PowerShell):**
```powershell
$env:MODEL_VERSION = "v2"
python api.py
# Para volver a la versión por defecto en esta terminal:
Remove-Item Env:MODEL_VERSION
```

**Linux / macOS (Bash):**
```bash
MODEL_VERSION=v2 python api.py
```

En Docker se usa `-e MODEL_VERSION=v2` (ver Docker). Si la versión configurada no existe o no puede cargarse, `POST /predict` responde `503` y el mensaje indica las versiones disponibles.

---

## Ejecutar la API localmente

> Secuencia completa con los resultados esperados: ver Guía paso a paso (pasos 3 a 5).

Desde `fase-3`, en cualquier terminal (la API queda en `http://localhost:8000`):

```bash
python api.py
```

> `gunicorn` es solo para el contenedor (Linux). En Windows se usa `python api.py`.

Para entrenar con `POST /train` fuera de Docker, los datos deben estar en `../fase-1/DatosCrudos.csv` o en la ruta indicada en `TRAIN_DATA_PATH`.

---

## Operaciones de la API

### `GET /health`

Devuelve siempre HTTP 200 mientras la API esté en ejecución, aunque el modelo configurado no esté disponible (en ese caso `modelo_disponible` es `false`).

```json
{
  "status": "ok",
  "servicio": "churn-api",
  "version_activa": "v1",
  "modelo_disponible": true,
  "versiones_disponibles": ["v1", "v2"]
}
```

### `POST /predict`

Recibe un objeto JSON con las **18 variables predictoras**; `customerID` es opcional (solo se devuelve en la respuesta) y las claves adicionales, como `TotalCharges`, se ignoran. La variable objetivo `Churn` no es necesaria.

| Campo | Tipo JSON | Valores admitidos |
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
| `MonthlyCharges` | número | cargo mensual (0 o más) |

Para representar un dato faltante se envía `null` en esa clave; el modelo lo imputa con la mediana (numéricas) o la moda (categóricas) aprendidas en el entrenamiento. **La clave debe estar presente.**

**Solicitud** (`data/ejemplo_solicitud.json`, primera fila de `data/ejemplo_prediccion.csv`):

```json
{
  "customerID": "0464-WJTKO",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 72,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "No",
  "OnlineSecurity": "No internet service",
  "OnlineBackup": "No internet service",
  "DeviceProtection": "No internet service",
  "TechSupport": "No internet service",
  "StreamingTV": "No internet service",
  "StreamingMovies": "No internet service",
  "Contract": "Two year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 20.1
}
```

**Respuesta** (HTTP 200, versión `v1`):

```json
{
  "prediccion": "No",
  "probabilidad_churn": 0.006983,
  "version": "v1",
  "nulos_imputados": [],
  "categorias_no_vistas": {},
  "customerID": "0464-WJTKO"
}
```

`prediccion` es `"Yes"` si el cliente se va (churn) y `"No"` si se queda; `probabilidad_churn` es la probabilidad (0 a 1) de que se vaya. Este resultado coincide con el de `fase-2/outputs/predicciones.csv` para el mismo cliente y versión.

**Errores:**

| HTTP | `error` | Cuándo ocurre |
|---|---|---|
| `400` | `json_invalido` | El cuerpo está vacío, no es JSON o no es un objeto. |
| `422` | `entrada_invalida` | Falta una variable o llega un tipo/valor incompatible. El campo `mensaje` nombra cada campo con problemas y `detalles` los lista. |
| `503` | `modelo_no_disponible` | La versión configurada no existe o no se pudo cargar. Incluye `versiones_disponibles`. |
| `404` / `405` / `413` | `ruta_no_encontrada` / `metodo_no_permitido` / `cuerpo_demasiado_grande` | Ruta inexistente, método incorrecto o cuerpo mayor de 1 MB. |
| `500` | `error_interno` | Error inesperado (queda registrado en los logs). |

Ejemplo de rechazo (`data/ejemplo_solicitud_invalida.json`, sin `tenure`) — HTTP 422:

```json
{
  "error": "entrada_invalida",
  "mensaje": "Entrada inválida. tenure: campo obligatorio ausente.",
  "detalles": [{"campo": "tenure", "problema": "campo obligatorio ausente"}]
}
```

Con varios problemas, se informan todos a la vez (por ejemplo `SeniorCitizen = 2` y `tenure = "abc"`):

```json
{
  "error": "entrada_invalida",
  "mensaje": "Entrada inválida. SeniorCitizen: debe ser el número 0 o 1 (se recibió el número 2); tenure: debe ser un número (se recibió el texto \"abc\").",
  "detalles": [
    {"campo": "SeniorCitizen", "problema": "debe ser el número 0 o 1 (se recibió el número 2)"},
    {"campo": "tenure", "problema": "debe ser un número (se recibió el texto \"abc\")"}
  ]
}
```

Ejemplo de modelo no disponible (`MODEL_VERSION=v99`) — HTTP 503:

```json
{
  "error": "modelo_no_disponible",
  "mensaje": "El modelo de la versión 'v99' no está disponible. Versiones disponibles: v1, v2. Configure MODEL_VERSION con una versión existente.",
  "version_activa": "v99",
  "modelo_disponible": false,
  "versiones_disponibles": ["v1", "v2"]
}
```

### `POST /train`

Entrena con los datos y la configuración de la Fase 2 (`TRAIN_DATA_PATH`, `TARGET`, `SEED`, `TEST_SIZE`; ver Configuración), **acepta un cuerpo vacío** y guarda una versión nueva en `MODELS_DIR` con `modelo.joblib`, `metrics.json` y `metadata.json`. El identificador es el siguiente número libre con el formato `v<N>`: con `v1` y `v2` guardadas, el primer entrenamiento crea `v3`, el siguiente `v4`. **Las versiones anteriores no se tocan** y la API sigue prediciendo con `MODEL_VERSION` hasta que se cambie. El entrenamiento tarda unos segundos y la respuesta llega cuando los archivos ya están guardados.

Respuesta (HTTP 201). Con `SEED=42` y los mismos datos, las métricas son las de `v1` (igual que `v3` en la Fase 2):

```json
{
  "version": "v3",
  "ruta": "models/v3",
  "metricas": {
    "model": {"accuracy": 0.802697, "precision": 0.667832, "recall": 0.510695, "f1": 0.578788, "roc_auc": 0.843316},
    "baseline": {"accuracy": 0.734564, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5}
  },
  "archivos": ["modelo.joblib", "metrics.json", "metadata.json"],
  "version_activa": "v1",
  "mensaje": "Versión v3 guardada. La API sigue prediciendo con v1; para usar la nueva, configure MODEL_VERSION=v3 y reinicie la API."
}
```

Para comprobarlo manualmente basta listar `models/` (aparece `v3/`) o consultar `GET /health` (`versiones_disponibles` incluye `v3`). Para predecir con ella: reiniciar la API con `MODEL_VERSION=v3`.

| HTTP | `error` | Cuándo ocurre |
|---|---|---|
| `409` | `entrenamiento_en_curso` | Ya hay otro entrenamiento ejecutándose. |
| `503` | `datos_no_disponibles` | No existe el CSV de `TRAIN_DATA_PATH`. No se crea ninguna versión. |
| `500` | `entrenamiento_fallido` | Los datos no cumplen el esquema (por ejemplo, falta la columna objetivo). |

> `POST /train` no tiene autenticación: cualquiera que alcance la API puede entrenar versiones nuevas (ver Limitaciones).

---

## Cliente de prueba y ejemplos con curl

Con la API en marcha, el cliente envía `GET /health`, una predicción válida con un cliente real de `data/ejemplo_prediccion.csv` y la misma solicitud sin `tenure` (que debe ser rechazada). Termina con código `0` si todo salió como se espera.

```bash
python cliente_prueba.py                           # fila 0, API en http://localhost:8000
python cliente_prueba.py --fila 2 --train          # otra fila y, además, POST /train
python cliente_prueba.py --help
```

Equivalente con **curl** (los cuerpos están en archivos para evitar problemas de comillas):

**Linux / macOS (Bash):**
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud_invalida.json
curl -X POST http://localhost:8000/train
```

**Windows (PowerShell)** — se usa `curl.exe` porque `curl` es un alias de otro comando:
```powershell
curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@data/ejemplo_solicitud.json"
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@data/ejemplo_solicitud_invalida.json"
curl.exe -X POST http://localhost:8000/train
```

---

## Docker

> Secuencia completa con los resultados esperados: ver Guía paso a paso (pasos 4 a 8). Esta sección es la referencia de los montajes y de los comandos auxiliares.

La imagen (`python:3.11-slim-bookworm`) contiene el código de la API y de la Fase 2, pero **no contiene modelos ni datos**: se montan desde el equipo anfitrión y no se entrena durante `docker build`.

| Montaje | Destino en el contenedor | Para qué |
|---|---|---|
| `fase-3/models` (lectura y escritura) | `/app/models` | Versiones que usa `/predict` y donde `/train` guarda las nuevas. Persisten fuera del contenedor. |
| `fase-1` (solo lectura) | `/data` | Contiene `DatosCrudos.csv`, que usa `/train` (`TRAIN_DATA_PATH=/data/DatosCrudos.csv`). |

Puerto: `8000` (se publica con `-p 8000:8000`). Todos los comandos se ejecutan desde `fase-3`.

**Linux / macOS (Bash):**
```bash
docker build -t modelos-fase3:1.0 --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .

docker run --rm -d --name churn-api -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/../fase-1:/data:ro" \
  modelos-fase3:1.0
```

**Windows (PowerShell)** — requiere Docker Desktop iniciado:
```powershell
docker build -t modelos-fase3:1.0 --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .

docker run --rm -d --name churn-api -p 8000:8000 `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/../fase-1:/data:ro" `
  modelos-fase3:1.0
```

(`--build-arg GIT_COMMIT=...` es opcional; sin él, las versiones entrenadas desde Docker guardan `git_revision: "no-disponible"`.)

Luego, con la API en marcha, se usa el cliente o los comandos curl de la sección anterior. Otros comandos útiles (iguales en ambas terminales):

```bash
docker ps                      # el estado pasa a "healthy" tras unos segundos (HEALTHCHECK sobre /health)
docker logs churn-api          # registros de la API y progreso de /train
docker stop churn-api          # detiene y elimina el contenedor (--rm)
```

**Seleccionar otra versión:** se agrega `-e MODEL_VERSION=v2` al `docker run` (antes del nombre de la imagen).

**Comprobar `/train` y la persistencia:** después de `POST /train` aparece la carpeta `models/v3/` en el equipo anfitrión y sigue allí tras `docker stop`.

**Scripts de la Fase 2 en la misma imagen:** la imagen no define `ENTRYPOINT`, así que también pueden ejecutarse, por ejemplo:

```bash
mkdir -p outputs
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  modelos-fase3:1.0 python predict.py \
  --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```

(En PowerShell se adapta la sintaxis como en la Fase 2: `${PWD}`, acento grave para continuar la línea y `New-Item -ItemType Directory -Force -Path outputs`.)

---

## Pruebas

> Resultado esperado: ver Guía paso a paso (paso 2).

Desde `fase-3`, en cualquier terminal:

```bash
python -m pytest
```

Las pruebas usan el cliente de pruebas de Flask (no abren puertos), entrenan con `pipeline.train_and_save` un modelo pequeño sobre datos sintéticos en directorios temporales (sin descargas ni cambios en `models/`) y no requieren que la API esté en marcha.

| Grupo | Qué comprueba |
|---|---|
| `/health` | **(mínima 1)** responde 200 con `status: ok`; sigue en 200 si el modelo no está disponible. |
| `/predict` válido | **(mínima 2)** una entrada válida devuelve `prediccion` (`Yes`/`No`), `version` y probabilidad; coincide con `inference.predict_frame` sobre el mismo modelo guardado; acepta JSON sin cabecera `Content-Type`; ignora claves adicionales; no depende del orden de las claves. |
| `/predict` inválido | **(mínima 3)** un campo obligatorio ausente se rechaza (422) y el mensaje nombra el campo; cada uno de los campos obligatorios por separado; varios problemas a la vez; 16 casos de tipo o valor incompatible; `NaN`; cuerpos que no son un objeto JSON (400). |
| Nulos y categorías | `null` numérico equivale a enviar la mediana del entrenamiento y `null` categórico a enviar la moda; todas las variables nulas; categoría no vista se procesa y se advierte. |
| Versiones y `503` | la versión se toma de `MODEL_VERSION`; versión inexistente o archivo dañado → 503 con las versiones disponibles; una versión no puede ser una ruta. |
| `/train` | crea `v3` (y luego `v4`) con sus tres archivos, deja intactas `v1` y `v2` (comparación por hash), no cambia la versión activa; sin datos → 503 y sin carpeta nueva; datos inválidos → 500; entrenamiento simultáneo → 409; registro del commit cuando no hay `.git`. |
| Errores generales | rutas y métodos inexistentes responden JSON; crear la aplicación no lee modelos ni datos. |
| Coherencia con la entrega | los módulos copiados son idénticos a los de `fase-2/`; el CSV y los JSON de ejemplo cumplen el esquema de `v1` y `v2`; con los **modelos entregados** (`v1` y `v2`) la API predice las 8 filas del CSV y reproduce la probabilidad de `fase-2/outputs/predicciones.csv`. |

El conjunto tiene 62 pruebas. Las 3 que cargan los modelos entregados solo se ejecutan si la versión de scikit-learn instalada coincide con la fijada en `requirements.txt` (con otra versión, `pytest` las marca como omitidas e indica el motivo), porque un `.joblib` solo es cargable de forma fiable con la versión con la que se guardó.

---

## Limitaciones conocidas

- **Datos ficticios:** los resultados no son evidencia sobre clientes reales (ver `fase-1/README.md` y `fase-2/README.md`).
- **Sin autenticación:** `POST /train` puede ser invocado por cualquiera que alcance la API. Es aceptable en este entorno académico local; en un uso real habría que protegerlo.
- **Un solo proceso:** la API atiende varias solicitudes con hilos, pero no escala a varios procesos (el candado de `/train` es de memoria). `/train` bloquea su solicitud unos segundos.
- **Activación de versiones:** cambiar la versión activa requiere reiniciar la API con otro `MODEL_VERSION`.
- **Umbral fijo:** la etiqueta se obtiene con el umbral 0.5 del modelo, igual que en la Fase 2.
- **Categorías no vistas:** no se rechazan; se codifican con ceros (la respuesta lo informa en `categorias_no_vistas`).
- **Sin registro de solicitudes ni monitoreo:** corresponden a la Fase 4.
- **Archivos creados desde Docker en Linux:** pertenecen al usuario `root` del contenedor.
- **Reproducibilidad:** depende de las versiones de `requirements.txt` (ver Preparar el entorno). `joblib` ejecuta código al cargar un archivo: solo deben cargarse modelos generados por este proyecto.

---

## Fuentes consultadas

- Documentación de Flask (aplicaciones, manejo de errores y cliente de pruebas): https://flask.palletsprojects.com/
- Documentación de gunicorn (ejecución con hilos y tiempo de espera): https://docs.gunicorn.org/
- Documentación de Docker sobre `HEALTHCHECK` y montajes de carpetas (*bind mounts*): https://docs.docker.com/
- Documentación de scikit-learn sobre persistencia de modelos y sus limitaciones: https://scikit-learn.org/stable/model_persistence.html
- Conjunto de datos *Telco Customer Churn* (muestra de IBM publicada en Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---