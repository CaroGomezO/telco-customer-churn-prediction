# Fase 2: script de entrenamiento

Aplicación simple y reproducible para entrenar un modelo de churn. Requiere Python
3.11. El entrenamiento muestra el avance por etapas en la consola.
El `Pipeline` de scikit-learn conserva el preprocesamiento junto con el modelo.
Se mantiene el tratamiento de Fase 1: `SeniorCitizen` se convierte de `0/1` a
`No/Yes` y pasa por one-hot; `TotalCharges` se excluye y `customerID` se trata
como identificador.

Como corrección de robustez de Fase 2, el pipeline incorpora imputación aprendida
solo con el conjunto de entrenamiento: mediana para variables numéricas y moda
para categóricas. Esta estrategia permite procesar valores faltantes sin fuga de
información.

El modelo usa explícitamente `GradientBoostingClassifier` con `learning_rate=0.1`,
`max_depth=3` y `n_estimators=100`, equivalentes a la configuración seleccionada
en Fase 1. Estos valores también se guardan en `metadata.json`.

## Preparar el entorno (PowerShell)

Desde `fase-2`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

El CSV etiquetado usado en el ejemplo está en `..\fase-1\DatosCrudos.csv`.

## Entrenar dos versiones

```powershell
python train.py --data ..\fase-1\DatosCrudos.csv --target Churn --output-dir models --version v1 --seed 42
python train.py --data ..\fase-1\DatosCrudos.csv --target Churn --output-dir models --version v2 --seed 7
```

Cada carpeta contiene `modelo.joblib`, `metrics.json` y `metadata.json`. Una
versión existente nunca se sobrescribe.

## Pruebas

```powershell
python -m pytest
```

# Predicción (`predict.py`)

`predict.py` carga una versión guardada y genera una predicción por cada fila de un CSV nuevo. **No entrena ni reajusta nada**: la imputación, el escalamiento y el One-Hot vienen guardados dentro de `modelo.joblib`, junto con el orden de las columnas, el nombre de la variable objetivo y el significado de las etiquetas. La lógica está en `inference.py` y reutiliza la lectura y limpieza de `pipeline.py`,
de modo que los datos nuevos se procesan igual que en el entrenamiento.

## Ejecutar (PowerShell, desde `fase-2`)

Cada comando va en una sola línea. Se puede indicar el archivo `.joblib` o la carpeta de la versión (`models\v1`):

```powershell
python predict.py --model models\v1\modelo.joblib --input data\ejemplo_prediccion.csv --output outputs\predicciones.csv
python predict.py --model models\v2 --input data\ejemplo_prediccion.csv --output outputs\predicciones_v2.csv
python predict.py --help
```

La carpeta `outputs\` se crea sola si no existe. El archivo de ejemplo
(`data\ejemplo_prediccion.csv`, 8 clientes reales sin la columna `Churn`) tiene el esquema completo del conjunto de datos.

## Esquema del CSV de entrada

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

## CSV de salida

| Columna | Significado |
|---|---|
| `customerID` | Solo si venía en la entrada. |
| `prediccion` | `1` = el cliente se va (churn), `0` = se queda. |
| `etiqueta` | `Yes` (se va) o `No` (se queda). |
| `probabilidad_churn` | Probabilidad entre 0 y 1 de que el cliente se vaya. |

Hay una fila de salida por cada fila de entrada y en el mismo orden, nunca se descartan filas en silencio.

## Qué pasa con los casos especiales

- **Orden de columnas:** no importa, se ordenan según el esquema del entrenamiento.

- **Columnas adicionales:** se ignoran.

- **Celdas vacías:** se imputan con la mediana (numéricas) o la moda (categóricas) aprendidas en el entrenamiento y el resumen indica cuántas fueron.

- **Categorías no vistas en el entrenamiento:** no producen error, se codifican con ceros y el script lo avisa por `stderr`.

- **Columna obligatoria ausente o texto en una columna numérica:** error claro.

- **Errores previsibles** (modelo o CSV inexistente, archivo dañado, columna ausente, tipo incompatible, CSV separado por punto y coma): mensaje `Error: ...` en `stderr`, código de salida `1` y **no se crea ni se modifica el archivo de salida**. Argumentos inválidos terminan con código `2`.

## Evidencia de ejecución

```text
> python predict.py --model models\v1\modelo.joblib --input data\ejemplo_prediccion.csv --output outputs\predicciones.csv
Predicción finalizada: 8 filas procesadas.
Resultado -> Yes: 2 | No: 6
Archivo de salida: outputs\predicciones.csv
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

Las pruebas de predicción están en `tests\test_predict.py` (se ejecutan con `python -m pytest`).