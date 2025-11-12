# app_abandonadores

Modelo de abandonadores (churn) basado en aprendizaje automático (XGBoost/RandomForest), con pipeline de features, entrenamiento, evaluación y exporte de predicciones y métricas.

## Requisitos
- Python 3.8+
- Paquetes: `pandas`, `numpy`, `scikit-learn`
- Opcional: `xgboost` (para `--model xgb`), `shap` (para `--shap`)

## Datos de entrada
- Colocar `sales_lms_clientes.csv` en `data/` con columnas:
  `FCH_FECHA, COD_FARMACIA_SAP, REGION, COD_NRO_TARJETA, EDAD, SEXO, CNT_UNIDADES, MNT_VENTA_BRUTA, MNT_VENTA_NETA, TICKETS`.

## Uso rápido
### Pipeline en un solo paso
```
python modelo_abandonadores.py \
  --csv data/sales_lms_clientes.csv \
  --out data \
  --umbral 90 \
  --model xgb
```

### Pipeline en dos pasos (recomendado para datasets grandes)
1. Preparar features y dataset del modelo  
   ```
   python preparar_datos.py \
     --csv data/sales_lms_clientes.csv \
     --out data \
     --umbral 90
   ```
2. Entrenar/predicir usando los archivos generados  
   ```
   python entrenar_modelo.py \
     --dataset data/dataset_modelo.csv \
     --features data/features_clientes.csv \
     --out data \
     --model xgb
   ```

Opciones útiles:
- `--model {xgb,rf}`: selecciona modelo (por defecto `xgb`).
- `--umbral`: días sin compra para marcar abandono (p.ej., 90/120/180).
- `--thr 0.2,0.4,0.6`: umbrales de riesgo para Bajo/Medio/Alto/Crítico.
- `--no-train`: solo prepara datasets; no entrena ni predice (solo en `modelo_abandonadores.py`).
- `--chunk-size`: controla el tamaño del chunk cuando se usa el modo de bajo consumo (preparación).
- `--shap`: calcula importancia SHAP si está disponible.

## Salidas
- `features_clientes.csv`: métricas y agregaciones por cliente.
- `dataset_modelo.csv`: dataset final listo para modelar.
- `predicciones_abandono.csv`: `COD_NRO_TARJETA`, `p_abandono`, `riesgo` y señales básicas.
- `importancias_modelo.csv`: importancia de variables del modelo entrenado.
- `metricas_modelo.json`: reporte de métricas (ROC AUC, PR AUC, classification report).
- `shap_importance.csv`: (si `--shap`) impacto medio |SHAP| por variable.
