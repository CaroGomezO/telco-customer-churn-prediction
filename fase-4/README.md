# Fase 4: Monitoreo de la API de Churn

> Registro de cada solicitud a `POST /predict`, script con dos métricas de operación (total de solicitudes y porcentaje de fallos), versiones del modelo identificadas y política de reentrenamiento.

## Integrantes

* **Carolina Gómez Osorno** — [@CaroGomezO](https://github.com/CaroGomezO)
* **Julian Esteban Hurtado Serna** — [@Julian-debug-s](https://github.com/Julian-debug-s)

Proyecto Integrador — Modelos y Simulación de Sistemas I (2026 II).

## Descripción

**Objetivo de la fase:** comprobar cómo se usa la API e identificar qué versión del modelo responde cada predicción.

**Relación con la Fase 3:** `fase-3/` se conserva sin cambios. `fase-4/` es una copia de `fase-3/` con lo siguiente agregado o modificado:

| Archivo | Cambio |
|---|---|
| `registro.py` | **Nuevo.** Escribe una fila por solicitud en un CSV, abierto en modo agregar. |
| `metricas.py` | **Nuevo.** Lee el registro y muestra el total de solicitudes y el porcentaje de fallos. |
| `api.py` | Registra cada llamada a `/predict` (incluidas las rechazadas) con un `after_request`. |
| `config.py` | Nueva variable `LOG_PATH`. `MODEL_VERSION` por defecto pasa a `v3`. |
| `Dockerfile` | Copia `registro.py` y `metricas.py`. Define `LOG_PATH` y `MODEL_VERSION=v3`. |
| `models/v3/` | Versión creada al probar `POST /train` en la Fase 3 (Docker). |
| `data/ejemplo_solicitud_2.json` | Segundo cliente válido (fila 2 del CSV, `4094-NSEDU`). |
| `muestras/registro_prueba.csv`, `muestras/registro_docker.csv` | Muestras del registro producidas en los dos ensayos del Reporte (API local y Docker). |
| `tests/test_registro.py` | **Nuevo.** Pruebas del registro y de las métricas. |

El modelo, el preprocesamiento, la validación y las respuestas de la API no cambian. Las funciones de la Fase 2 se siguen reutilizando sin modificar.

---

## Estructura de la carpeta

```
fase-4/
├── api.py                  # API Flask (Fase 3) + registro de /predict
├── registro.py             # (Nuevo) escritura del registro CSV
├── metricas.py             # (Nuevo) total de solicitudes y % de fallos
├── config.py               # Variables de entorno (+ LOG_PATH)
├── validation.py           # (Fase 3, sin cambios)
├── service.py              # (Fase 3, sin cambios)
├── pipeline.py, inference.py, train.py, predict.py   # (Fase 2, sin cambios)
├── cliente_prueba.py       # (Fase 3, sin cambios)
├── data/
│   ├── ejemplo_prediccion.csv
│   ├── ejemplo_solicitud.json          # Cliente 0464-WJTKO (válido)
│   ├── ejemplo_solicitud_2.json        # Cliente 4094-NSEDU (válido)
│   └── ejemplo_solicitud_invalida.json # 0464-WJTKO sin `tenure` (se rechaza)
├── models/
│   ├── v1/                 # Fase 2 (versión anterior)
│   ├── v2/                 # Fase 2 (otra partición)
│   └── v3/                 # Creada con POST /train (versión activa)
├── logs/                   # Registro vivo (no se versiona en git; se crea al primer /predict)
│   └── predicciones.csv
├── muestras/
│   ├── registro_prueba.csv # Ensayo 1 del Reporte (API local): 3 solicitudes, 33,3 %
│   └── registro_docker.csv # Ensayo 2 del Reporte (Docker, v3 → v1): 8 solicitudes, 25,0 %
├── tests/
│   ├── referencia/         # Predicciones registradas de v1, v2 y v3
│   ├── test_api.py         # Pruebas de la Fase 3
│   ├── test_versiones.py   # Pruebas de la Fase 3, extendidas a v3
│   └── test_registro.py    # Pruebas de la Fase 4
├── conftest.py, pytest.ini, requirements.txt, Dockerfile, .dockerignore, .gitignore
└── README.md
```

---

## 1. Registro de las solicitudes

Cada llamada a `/predict` agrega **una fila** a `logs/predicciones.csv` (configurable con `LOG_PATH`):

```
fecha_hora,estado_http,version,prediccion
2026-11-18T15:50:12-05:00,200,v3,No
2026-11-18T15:50:14-05:00,422,,
```

| Campo | Contenido |
|---|---|
| `fecha_hora` | Fecha y hora local en ISO 8601, con la zona horaria. En Docker la zona es UTC (`+00:00`). |
| `estado_http` | Código HTTP devuelto. |
| `version` | Versión del modelo que produjo la predicción. Vacía si no se predijo. |
| `prediccion` | `Yes` (el cliente se va) o `No`. Vacía en un error. |

**Cómo se garantiza una fila por intento.** El registro se hace en un `@app.after_request` de Flask. Esa función se ejecuta **una vez por solicitud**, después de la vista o del manejador de errores, cuando el código de respuesta ya es definitivo. Por eso quedan registradas también las solicitudes que no llegan a predecir:

| Caso | `estado_http` | `version` | `prediccion` |
|---|---|---|---|
| Predicción válida | `200` | versión activa | `Yes` / `No` |
| Cuerpo que no es JSON u objeto | `400` | vacía | vacía |
| Falta una variable o tiene un tipo inválido | `422` | vacía | vacía |
| La versión configurada no existe | `503` | vacía | vacía |
| Método distinto de POST en `/predict` / cuerpo > 1 MB | `405` / `413` | vacía | vacía |
| Error inesperado | `500` | vacía | vacía |

La versión y la predicción solo se escriben **después** de que el modelo devuelve el resultado: en un 422 el modelo no se usó, así que la versión queda vacía. `/health` y `/train` no se registran.

**Persistencia.** El archivo se abre en modo agregar (`"a"`), y el encabezado se escribe solo cuando el archivo es nuevo o está vacío, así que el historial se conserva entre ejecuciones. Un candado (`threading.Lock`) evita que dos hilos de gunicorn mezclen sus filas. En Docker, `logs/` se monta desde el equipo anfitrión, igual que `models/`. Si el archivo no se puede escribir, la respuesta al cliente no cambia y el error queda en los logs de la API.

## 2. Métricas de operación

```bash
python metricas.py                                    # usa LOG_PATH o logs/predicciones.csv
python metricas.py --log muestras/registro_prueba.csv # otro archivo
```

El script cuenta las filas de datos (total) y las que tienen un código entre 400 y 599 (fallidas), y calcula `fallidas / total × 100`:

```
Registro: muestras/registro_prueba.csv
Total de solicitudes: 3
Solicitudes fallidas (HTTP 4xx o 5xx): 1
Porcentaje de fallos: 33,3 %
```

Si el registro no existe o no tiene filas de datos:

```
Total de solicitudes: 0
Porcentaje de fallos: sin solicitudes
```

Este porcentaje mide **el funcionamiento del servicio** (cuántas solicitudes no obtuvieron predicción), no la calidad del modelo. Para medir la calidad haría falta saber si cada cliente realmente abandonó el servicio.

## 3. Versiones del modelo

| Identificador | Rol en la Fase 4 | Origen | Semilla | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **`v3`** | **Activa** (`MODEL_VERSION` por defecto) | `POST /train` en Docker (Fase 3), commit `1f3a166` | 42 | 0.579 | 0.843 |
| **`v1`** | **Anterior** (respaldo) | `train.py` (Fase 2), commit `6837a3e` | 42 | 0.579 | 0.843 |
| `v2` | Se conserva; no se usa | `train.py` (Fase 2) | 7 | 0.622 | 0.865 |

**Cuál usa la API.** La versión activa aparece en `GET /health` (`version_activa`), en cada respuesta de `/predict` (`version`) y en la columna `version` del registro.

**Por qué `v3`.** `v1` y `v3` se entrenaron con los mismos datos, parámetros y semilla, así que tienen las mismas métricas sobre la misma partición de prueba y dan las mismas predicciones (en el ensayo, `0.006983` y `0.736106`, iguales a `fase-2/outputs/predicciones.csv`). Se activa `v3` para usar la versión producida por la propia API y demostrar el ciclo «entrenar con `/train` → activar → poder volver». Según la política de abajo, un empate no justificaría cambiar en producción; aquí el cambio no altera ninguna predicción. `v2` no se usa como referencia porque su F1 mayor sale de **otra partición** de prueba (semilla 7), así que no es comparable con el de `v1` o `v3` (ver `fase-2/README.md`).

**Cómo volver a la versión anterior (`v1`).** Con la configuración existente, sin tocar código: se define `MODEL_VERSION=v1` y se reinicia la API.

**Linux / macOS (Bash):**
```bash
MODEL_VERSION=v1 python api.py
```

**Windows (PowerShell):**
```powershell
$env:MODEL_VERSION = "v1"
python api.py
```

**Docker:** se agrega `-e MODEL_VERSION=v1` al `docker run`, antes del nombre de la imagen (ver paso 5 de la guía). Las filas siguientes del registro muestran `v1` en la columna `version`.

---

## Reporte

Se hicieron dos ensayos, cada uno sobre un registro vacío: el **ensayo 1** con la API local (el caso del enunciado) y el **ensayo 2** con la API en Docker, con más solicitudes, otro tipo de error y un cambio de versión.

### Ensayo 1. API local: el caso del enunciado

Con `python api.py` (versión activa `v3`) se enviaron tres solicitudes a `POST /predict`:

| # | Archivo enviado | Cliente | Resultado |
|---|---|---|---|
| 1 | `data/ejemplo_solicitud.json` | `0464-WJTKO` (válido) | HTTP 200, `prediccion: "No"`, `probabilidad_churn: 0.006983`, `version: "v3"` |
| 2 | `data/ejemplo_solicitud_2.json` | `4094-NSEDU` (válido) | HTTP 200, `prediccion: "Yes"`, `probabilidad_churn: 0.736106`, `version: "v3"` |
| 3 | `data/ejemplo_solicitud_invalida.json` | `0464-WJTKO` **sin `tenure`** | HTTP 422, `entrada_invalida`: `tenure: campo obligatorio ausente` |

Registro resultante (`muestras/registro_prueba.csv`):

```
fecha_hora,estado_http,version,prediccion
2026-11-18T15:50:12-05:00,200,v3,No
2026-11-18T15:50:13-05:00,200,v3,Yes
2026-11-18T15:50:14-05:00,422,,
```

| Indicador | Valor |
|---|---|
| Total de solicitudes | **3** |
| Porcentaje de solicitudes fallidas (HTTP 4xx o 5xx) | **33,3 %** (1 de 3) |

**Interpretación.** La API procesó dos clientes con `v3` (uno con predicción `No` y otro con `Yes`) y rechazó el tercero con HTTP 422 porque faltaba la variable `tenure`. La tasa de fallos fue 1/3 (33,3 %). Ese rechazo era el resultado esperado de la prueba de validación, no un fallo del servicio ni del modelo.

### Ensayo 2. API en Docker, con reinicio y vuelta a `v1`

Con la imagen `modelos-fase4:1.0` y `logs/` montado desde el anfitrión (se puede ver mas adelante los comandos en el paso 5 de la guía), el contenedor se ejecutó **dos veces sobre el mismo registro**:

* **Ejecución 1**, con la versión por defecto `v3`: 5 solicitudes.
* Se detuvo el contenedor con `docker stop`.
* **Ejecución 2**, arrancada con `-e MODEL_VERSION=v1` (vuelta a la versión anterior): 3 solicitudes más.

| # | Ejecución | Archivo enviado | Resultado |
|---|---|---|---|
| 1 | 1 (`v3`) | `data/ejemplo_solicitud.json` | HTTP 200, `No` |
| 2 | 1 (`v3`) | `data/ejemplo_solicitud_2.json` | HTTP 200, `Yes` |
| 3 | 1 (`v3`) | `data/ejemplo_solicitud_invalida.json` (sin `tenure`) | HTTP 422, `entrada_invalida` |
| 4 | 1 (`v3`) | `data/ejemplo_solicitud.json` | HTTP 200, `No` |
| 5 | 1 (`v3`) | `pytest.ini` (no es JSON) | HTTP 400, `json_invalido` |
| 6 | 2 (`v1`) | `data/ejemplo_solicitud_2.json` | HTTP 200, `Yes` |
| 7 | 2 (`v1`) | `data/ejemplo_solicitud.json` | HTTP 200, `No` |
| 8 | 2 (`v1`) | `data/ejemplo_solicitud_2.json` | HTTP 200, `Yes` |

Registro resultante (`muestras/registro_docker.csv`). Dentro del contenedor la hora está en UTC; el salto de 02:20:00 a 02:20:04 es el reinicio:

```
fecha_hora,estado_http,version,prediccion
2026-11-19T02:19:56+00:00,200,v3,No
2026-11-19T02:19:57+00:00,200,v3,Yes
2026-11-19T02:19:58+00:00,422,,
2026-11-19T02:19:59+00:00,200,v3,No
2026-11-19T02:20:00+00:00,400,,
2026-11-19T02:20:04+00:00,200,v1,Yes
2026-11-19T02:20:05+00:00,200,v1,No
2026-11-19T02:20:06+00:00,200,v1,Yes
```

Métricas calculadas con `docker exec churn-api python metricas.py`:

| Momento | Total de solicitudes | Fallidas (4xx/5xx) | Porcentaje de fallos |
|---|---|---|---|
| Al final de la ejecución 1 (`v3`) | 5 | 2 | **40,0 %** |
| Al final de la ejecución 2 (`v1`), registro acumulado | **8** | **2** | **25,0 %** |

`python metricas.py` en el anfitrión, sobre el mismo `logs/predicciones.csv`, da el mismo resultado (8 solicitudes, 25,0 %).

**Interpretación.** La API en Docker respondió seis predicciones, tres con `v3` y tres con `v1` después de volver a la versión anterior con `MODEL_VERSION=v1`. Rechazó dos solicitudes: una sin `tenure` (422) y otra cuyo cuerpo no era JSON (400). La tasa de fallos fue 2/8 (25,0 %). Ambos rechazos eran errores del cliente, provocados a propósito, y no fallos del servicio. Además, el ensayo confirma tres cosas:

* **Persistencia:** el registro siguió acumulando filas después de `docker stop` (5 → 8 filas, con un solo encabezado).
* **Trazabilidad:** la columna `version` muestra qué versión respondió cada predicción.
* **`HEALTHCHECK` fuera de las métricas:** los chequeos de salud de Docker a `/health` no se registran.

### Política de reentrenamiento

Al reunir **500 clientes nuevos con `Churn` observado** (el resultado real, conocido al terminar el periodo de observación), un integrante del equipo los agrega al CSV de entrenamiento, reserva aparte una porción de esos casos y ejecuta manualmente `POST /train` para crear una versión candidata. La candidata y la versión activa se evalúan con la **métrica principal de la Fase 2, el F1 de la clase `Churn = Yes`**, sobre **los mismos casos reservados** (no usados en el entrenamiento de ninguna de las dos). La candidata se activa con `MODEL_VERSION` solo si su F1 supera al de la versión activa. Si no lo supera, se mantiene la activa y la candidata queda guardada en `models/`.

Un 4xx en el registro no dispara esta revisión: un campo ausente se corrige en quien envía los datos. La revisión depende de los casos con resultado real. Se eligieron 500 clientes, alrededor del 7 % de los 7 043 actuales y unos 130 abandonos con la tasa observada del 26.5 %, para que la comparación de F1 tenga suficientes casos positivos.

### Identificadores de las versiones

* **Activa:** `v3` (`models/v3/`, creada con `POST /train`).
* **Anterior:** `v1` (`models/v1/`, Fase 2). Se vuelve a ella con `MODEL_VERSION=v1`.

---

## Guía paso a paso: reproducir el reporte

Todos los comandos se ejecutan desde la carpeta `fase-4`. En PowerShell se escribe `curl.exe`; en Bash, `curl`.

### Paso 1. Entorno e instalación

**Windows (PowerShell):**
```powershell
cd telco-customer-churn-prediction\fase-4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Linux / macOS (Bash):**
```bash
cd telco-customer-churn-prediction/fase-4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Paso 2. Pruebas automáticas

```bash
python -m pytest
```

**Resultado esperado:** `98 passed`. Las pruebas escriben el registro en directorios temporales y no tocan `logs/`.

### Paso 3. Partir de un registro vacío

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force logs -ErrorAction SilentlyContinue
python metricas.py
```

**Linux / macOS (Bash):**
```bash
rm -rf logs
python metricas.py
```

**Resultado esperado:** `Total de solicitudes: 0` y `Porcentaje de fallos: sin solicitudes`.

### Paso 4. Ensayo con la API local (dos terminales)

**Terminal 1:**
```bash
python api.py
```

**Terminal 2:**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud_2.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud_invalida.json
python metricas.py
```
(En PowerShell, `curl.exe` y `-d "@data/ejemplo_solicitud.json"`.)

**Resultado esperado:** dos respuestas 200 con `"version": "v3"`, una 422 y luego `Total de solicitudes: 3` y `Porcentaje de fallos: 33,3 %`. El archivo `logs/predicciones.csv` tiene tres filas como las del Reporte. Si aparecen solo dos, el rechazo no se registró.

Si se detiene la API (`Ctrl+C`), se vuelve a arrancar y se envía otra solicitud, el registro pasa a 4 filas con un solo encabezado: el historial se conserva.

### Paso 5. Ensayo en Docker (ensayo 2 del Reporte)

Con la API local detenida, el puerto 8000 libre y el motor de Docker en ejecución. Todos los comandos se ejecutan desde `fase-4` en una sola terminal.

**5.1. Construir la imagen y vaciar el registro**

**Linux / macOS (Bash):**
```bash
docker build -t modelos-fase4:1.0 --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
rm -rf logs && mkdir logs
```

**Windows (PowerShell):**
```powershell
docker build -t modelos-fase4:1.0 --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
Remove-Item -Recurse -Force logs -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force -Path logs
```

**5.2. Ejecución 1: versión por defecto (`v3`)**

**Linux / macOS (Bash):**
```bash
docker run --rm -d --name churn-api -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/logs:/app/logs" \
  -v "$PWD/../fase-1:/data:ro" \
  modelos-fase4:1.0
```

**Windows (PowerShell):**
```powershell
docker run --rm -d --name churn-api -p 8000:8000 `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/logs:/app/logs" `
  -v "${PWD}/../fase-1:/data:ro" `
  modelos-fase4:1.0
```

| Montaje | Destino | Para qué |
|---|---|---|
| `fase-4/models` | `/app/models` | Versiones del modelo (lectura y escritura, como en la Fase 3). |
| `fase-4/logs` | `/app/logs` | **Registro de `/predict`**: queda en el equipo anfitrión y sobrevive a `docker stop`. |
| `fase-1` (solo lectura) | `/data` | `DatosCrudos.csv` para `/train`. |

`docker ps` debe mostrar `churn-api` como `(healthy)` a los pocos segundos. Después se envían cinco solicitudes y se calculan las métricas dentro del contenedor (en PowerShell, `curl.exe` y `-d "@archivo"`):

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud_2.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud_invalida.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @pytest.ini
docker exec churn-api python metricas.py
docker stop churn-api
```

**Resultado esperado:** `/health` responde `"version_activa": "v3"`. Las respuestas son 200, 200, 422, 200 y 400, y `metricas.py` muestra:

```
Registro: logs/predicciones.csv
Total de solicitudes: 5
Solicitudes fallidas (HTTP 4xx o 5xx): 2
Porcentaje de fallos: 40,0 %
```

**5.3. Ejecución 2: volver a la versión anterior (`v1`)**

Es el mismo `docker run` con `-e MODEL_VERSION=v1` antes del nombre de la imagen.

**Linux / macOS (Bash):**
```bash
docker run --rm -d --name churn-api -p 8000:8000 -e MODEL_VERSION=v1 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/logs:/app/logs" \
  -v "$PWD/../fase-1:/data:ro" \
  modelos-fase4:1.0
```

**Windows (PowerShell):**
```powershell
docker run --rm -d --name churn-api -p 8000:8000 -e MODEL_VERSION=v1 `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/logs:/app/logs" `
  -v "${PWD}/../fase-1:/data:ro" `
  modelos-fase4:1.0
```

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud_2.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/ejemplo_solicitud_2.json
docker exec churn-api python metricas.py
docker stop churn-api
```

**Resultado esperado:** `/health` responde `"version_activa": "v1"`, las tres predicciones traen `"version": "v1"` y las métricas acumulan las dos ejecuciones:

```
Registro: logs/predicciones.csv
Total de solicitudes: 8
Solicitudes fallidas (HTTP 4xx o 5xx): 2
Porcentaje de fallos: 25,0 %
```

**5.4. Comprobar el registro en el anfitrión**

Con el contenedor ya detenido:

```bash
cat logs/predicciones.csv      # PowerShell: Get-Content logs\predicciones.csv
python metricas.py
```

**Resultado esperado:** 8 filas de datos con un solo encabezado, las 5 primeras con `v3` (o vacías en los errores) y las 3 últimas con `v1`, como en `muestras/registro_docker.csv`. `metricas.py` vuelve a mostrar 8 solicitudes y 25,0 %. Las horas aparecen en UTC (`+00:00`), la zona del contenedor. En Linux el archivo pertenece a `root` (ver Limitaciones).

---

## Datos de entrada de `POST /predict`

El contrato de entrada es el de la Fase 3. La API recibe un objeto JSON con las **18 variables predictoras**; `customerID` es opcional y las claves adicionales se ignoran. Los casos especiales (nulos, categorías nuevas, claves adicionales) están descritos en `fase-3/README.md`.

| Campo | Tipo JSON | Valores admitidos | Unidad o significado |
|---|---|---|---|
| `gender` | texto | `Female`, `Male` | Género registrado del cliente. |
| `SeniorCitizen` | número | `0` o `1` | `1` si el cliente es adulto mayor, `0` si no (sin unidad). |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | texto | `Yes`, `No` | Si el cliente tiene pareja, dependientes, servicio telefónico y facturación sin papel, respectivamente. |
| `tenure` | número entero | `0` o más | **Meses** completos como cliente. |
| `MultipleLines` | texto | `Yes`, `No`, `No phone service` | Si tiene varias líneas telefónicas; `No phone service` si no tiene teléfono. |
| `InternetService` | texto | `DSL`, `Fiber optic`, `No` | Tipo de conexión a internet. |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | texto | `Yes`, `No`, `No internet service` | Si contrató cada servicio adicional; `No internet service` si no tiene internet. |
| `Contract` | texto | `Month-to-month`, `One year`, `Two year` | Tipo de contrato. |
| `PaymentMethod` | texto | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Método de pago. |
| `MonthlyCharges` | número | `0` o más | Cargo **mensual** facturado. La fuente del conjunto de datos no especifica la moneda: el valor debe enviarse en la misma unidad de los datos de entrenamiento. |

---

## Configuración (variables de entorno)

Las mismas de la Fase 3 (ver `fase-3/README.md`), con estos cambios:

| Variable | Valor por defecto | Para qué sirve |
|---|---|---|
| `MODEL_VERSION` | **`v3`** (en la Fase 3 era `v1`) | Versión que usa `POST /predict`. `v1` es la versión anterior. |
| `LOG_PATH` | `logs/predicciones.csv` | **(Nuevo)** CSV del registro de `/predict`. La carpeta se crea si no existe. `metricas.py` también lo lee. |

---

## Pruebas

```bash
python -m pytest
```

Se mantienen las pruebas de la Fase 3 (`tests/test_api.py` y `tests/test_versiones.py`, 79 en total). Hay dos diferencias: el valor por defecto esperado de `MODEL_VERSION` es `v3`, y las pruebas de versiones cubren también `v3`, con su referencia en `tests/referencia/predicciones_v3.csv`, tomada de `fase-2/outputs/predicciones_v3.csv`. Se agregan 19 en `tests/test_registro.py`:

| Grupo | Qué comprueba |
|---|---|
| Ensayo del enunciado | Dos válidas y una sin variable dan **3 filas** (200, 200, 422), con la versión y la predicción solo en las válidas; las métricas dan 3 y `33,3 %`. |
| Una fila por intento | 400, 422, 405, 503 y 500 generan exactamente una fila sin versión ni predicción; `/health` y otras rutas no escriben. |
| Persistencia | Dos ejecuciones de la API sobre el mismo archivo acumulan filas con un solo encabezado; 20 solicitudes simultáneas producen 20 filas. |
| `metricas.py` | Registro inexistente, vacío o solo con encabezado → `0` y `sin solicitudes`; 4xx y 5xx cuentan como fallos (200, 201, 399 y 600 no); un archivo con formato incorrecto se rechaza. |
| Muestras entregadas | `muestras/registro_prueba.csv` (3 solicitudes, `33,3 %`) y `muestras/registro_docker.csv` (8 solicitudes, `25,0 %`) tienen el formato esperado, solo las filas 200 traen versión y predicción, y reproducen los indicadores del Reporte. |

---

## Limitaciones conocidas

- **Solo métricas de operación:** el registro no tiene el resultado real de cada cliente, así que no mide la calidad predictiva (ver la política de reentrenamiento).
- **Un solo proceso:** el candado del registro es de memoria; es suficiente con la configuración del contenedor (1 proceso de gunicorn y 4 hilos). Con varios procesos haría falta un bloqueo de archivo o una base de datos.
- **Sin rotación:** `logs/predicciones.csv` crece indefinidamente.
- **Sin datos personales:** el registro no guarda el `customerID` ni las variables enviadas; solo los cuatro campos pedidos.
- **Archivos creados desde Docker en Linux:** `logs/predicciones.csv` pertenece al usuario `root` del contenedor.
- Las demás limitaciones de la Fase 3 siguen vigentes (sin autenticación en `/train`, umbral fijo de 0.5, datos ficticios).

---

## Fuentes consultadas

- Documentación de Flask sobre `after_request` y manejo de errores: https://flask.palletsprojects.com/
- Documentación del módulo `csv` de Python: https://docs.python.org/3/library/csv.html
- Documentación de Docker sobre montajes de carpetas (*bind mounts*): https://docs.docker.com/
