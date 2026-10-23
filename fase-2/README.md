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
