"""
Pipeline de Churn (abandono) con modelo configurable (XGBoost/RandomForest),
explicabilidad opcional (SHAP) y exportes para visibilidad.

Entradas esperadas en `data/sales_lms_clientes.csv`:
FCH_FECHA, COD_FARMACIA_SAP, REGION, COD_NRO_TARJETA, EDAD, SEXO,
CNT_UNIDADES, MNT_VENTA_BRUTA, MNT_VENTA_NETA, TICKETS

Salidas principales en carpeta de datos:
- features_clientes.csv       -> metricas por cliente (una fila por cliente)
- dataset_modelo.csv          -> listo para entrenar (features + label)
- predicciones_abandono.csv   -> probabilidad p_abandono, riesgo y senales basicas
- importancias_modelo.csv     -> importancia de variables (segun modelo)
- shap_importance.csv         -> (si `--shap`) impacto medio |SHAP| por variable
"""

import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import timedelta


# =============== Utilidades ===============
def moda_robusta(s: pd.Series):
    try:
        m = s.mode(dropna=True)
        return m.iloc[0] if len(m) else np.nan
    except Exception:
        return np.nan


def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


@dataclass
class Config:
    csv_path: str
    output_dir: str
    umbral_abandono_dias: int = 90
    test_size: float = 0.30
    random_state: int = 42
    model_type: str = "xgb"  # "xgb" | "rf"
    train_model: bool = True
    shap_enabled: bool = False
    risk_thresholds: Tuple[float, float, float] = (0.2, 0.4, 0.6)  # Bajo/Medio/Alto/Critico
    chunk_size: int = 200_000


def cargar_y_limpiar(csv_path: str) -> pd.DataFrame:
    # Lectura robusta de CSV (intenta varias codificaciones comunes) y reduce memoria
    last_err = None
    cols = ['FCH_FECHA','COD_FARMACIA_SAP','REGION','COD_NRO_TARJETA','EDAD','SEXO',
            'CNT_UNIDADES','MNT_VENTA_BRUTA','MNT_VENTA_NETA','TICKETS']
    dtypes = {
        'COD_FARMACIA_SAP': 'string',
        'REGION': 'category',
        'COD_NRO_TARJETA': 'string',
        'EDAD': 'float32',
        'SEXO': 'category',
        'CNT_UNIDADES': 'float32',
        'MNT_VENTA_BRUTA': 'float32',
        'MNT_VENTA_NETA': 'float32',
        'TICKETS': 'float32',
    }
    for enc in ('utf-8-sig', 'utf-8', 'latin1', 'cp1252'):
        try:
            df = pd.read_csv(
                csv_path,
                encoding=enc,
                usecols=cols,
                dtype=dtypes,
                parse_dates=['FCH_FECHA'],
                infer_datetime_format=True,
                engine='python',
            )
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err
    df.columns = [c.strip().upper() for c in df.columns]
    requeridas = set(cols)
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")
    # Limpieza
    df = df.dropna(subset=['FCH_FECHA','COD_NRO_TARJETA']).copy()
    return df


def construir_features_y_label(df: pd.DataFrame, umbral_abandono_dias: int):
    ref_date = df['FCH_FECHA'].max()
    grp = df.sort_values('FCH_FECHA').groupby('COD_NRO_TARJETA', as_index=False)
    agg_df = grp.agg({
        'FCH_FECHA': ['min','max','nunique'],
        'CNT_UNIDADES': 'sum',
        'MNT_VENTA_BRUTA': ['mean','sum'],
        'MNT_VENTA_NETA': ['mean','sum'],
        'TICKETS': 'sum',
        'COD_FARMACIA_SAP': pd.Series.nunique,
        'REGION': moda_robusta,
        'EDAD': 'last',
        'SEXO': 'last'
    })
    agg_df.columns = [
        'COD_NRO_TARJETA',
        'FECHA_PRIMERA_COMPRA','FECHA_ULTIMA_COMPRA','DIAS_CON_COMPRA',
        'TOTAL_UNIDADES','MNT_BRUTO_PROM','MNT_BRUTO_TOT',
        'MNT_NETO_PROM','MNT_NETO_TOT','TOTAL_TICKETS',
        'FARMACIAS_DISTINTAS','REGION_MAS_FRECUENTE','EDAD','SEXO'
    ]

    agg_df['DIAS_DESDE_ULTIMA'] = (ref_date - pd.to_datetime(agg_df['FECHA_ULTIMA_COMPRA'])).dt.days
    agg_df['ANTIGUEDAD_CLIENTE'] = (ref_date - pd.to_datetime(agg_df['FECHA_PRIMERA_COMPRA'])).dt.days
    agg_df['FRECUENCIA_MENSUAL'] = agg_df['TOTAL_TICKETS'] / (agg_df['ANTIGUEDAD_CLIENTE'] / 30.0 + 1)
    agg_df['TICKET_PROMEDIO'] = np.where(
        agg_df['TOTAL_TICKETS'] > 0,
        agg_df['MNT_NETO_TOT'] / agg_df['TOTAL_TICKETS'],
        0.0,
    )

    df['MES'] = df['FCH_FECHA'].dt.month
    df['DOW'] = df['FCH_FECHA'].dt.dayofweek
    tmp_mode = df.groupby('COD_NRO_TARJETA').agg({'MES': moda_robusta, 'DOW': moda_robusta}).reset_index().rename(
        columns={'MES':'MES_FRECUENTE','DOW':'DOW_FRECUENTE'}
    )
    agg_df = agg_df.merge(tmp_mode, on='COD_NRO_TARJETA', how='left')

    ventana = 90
    corte = ref_date - timedelta(days=ventana)
    df['ES_ULT_90'] = (df['FCH_FECHA'] > corte).astype(int)
    g90 = df.groupby(['COD_NRO_TARJETA','ES_ULT_90']).agg(
        MNT_90=('MNT_VENTA_NETA','sum')
    ).reset_index()
    g90_w = g90.pivot(index='COD_NRO_TARJETA', columns='ES_ULT_90', values='MNT_90').fillna(0.0)
    if 0 not in g90_w.columns:
        g90_w[0] = 0.0
    if 1 not in g90_w.columns:
        g90_w[1] = 0.0
    g90_w.columns = ['MNT_ANT','MNT_ULT90']
    g90_w = g90_w.reset_index()
    g90_w['VAR_GASTO_90'] = (g90_w['MNT_ULT90'] - g90_w['MNT_ANT']) / (g90_w['MNT_ANT'] + 1e-6)
    agg_df = agg_df.merge(g90_w[['COD_NRO_TARJETA','VAR_GASTO_90']], on='COD_NRO_TARJETA', how='left').fillna({'VAR_GASTO_90':0.0})

    agg_df['ABANDONO'] = (agg_df['DIAS_DESDE_ULTIMA'] > umbral_abandono_dias).astype(int)

    feature_cols = [
        'EDAD','SEXO','REGION_MAS_FRECUENTE','TOTAL_TICKETS','FRECUENCIA_MENSUAL',
        'TICKET_PROMEDIO','MNT_NETO_TOT','MNT_NETO_PROM','DIAS_DESDE_ULTIMA',
        'ANTIGUEDAD_CLIENTE','FARMACIAS_DISTINTAS','MES_FRECUENTE','DOW_FRECUENTE',
        'VAR_GASTO_90'
    ]
    df_features = agg_df[['COD_NRO_TARJETA'] + feature_cols + ['ABANDONO']].copy()
    num_cols = [c for c in feature_cols if c not in ['SEXO','REGION_MAS_FRECUENTE']]
    df_features[num_cols] = df_features[num_cols].fillna(0)
    df_model = pd.get_dummies(
        df_features,
        columns=['SEXO','REGION_MAS_FRECUENTE'],
        drop_first=True
    )
    return agg_df, df_model


def construir_features_y_label_en_chunks(csv_path: str, umbral_abandono_dias: int,
                                         chunksize: int = 200_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ['FCH_FECHA','COD_FARMACIA_SAP','REGION','COD_NRO_TARJETA','EDAD','SEXO',
            'CNT_UNIDADES','MNT_VENTA_BRUTA','MNT_VENTA_NETA','TICKETS']
    dtypes = {
        'COD_FARMACIA_SAP': 'string',
        'REGION': 'category',
        'COD_NRO_TARJETA': 'string',
        'EDAD': 'float32',
        'SEXO': 'category',
        'CNT_UNIDADES': 'float32',
        'MNT_VENTA_BRUTA': 'float32',
        'MNT_VENTA_NETA': 'float32',
        'TICKETS': 'float32',
    }

    # Detectar encoding con una lectura corta
    enc_found = None
    for enc in ('utf-8-sig', 'utf-8', 'latin1', 'cp1252'):
        try:
            _ = pd.read_csv(csv_path, encoding=enc, nrows=100, usecols=cols, engine='python')
            enc_found = enc
            break
        except Exception:
            continue
    if enc_found is None:
        enc_found = 'utf-8'

    # 1) Obtener ref_date
    ref_date = None
    for chunk in pd.read_csv(
        csv_path,
        encoding=enc_found,
        usecols=cols,
        dtype=dtypes,
        parse_dates=['FCH_FECHA'],
        engine='python',
        chunksize=chunksize,
    ):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        chunk = chunk.dropna(subset=['FCH_FECHA','COD_NRO_TARJETA'])
        mx = chunk['FCH_FECHA'].max()
        ref_date = mx if ref_date is None or mx > ref_date else ref_date

    if ref_date is None:
        raise ValueError('No se encontraron filas válidas en el CSV')

    corte_90 = ref_date - timedelta(days=90)

    # 2) Acumular agregaciones por cliente
    from collections import defaultdict
    acc = {}

    def _ensure(cid):
        if cid not in acc:
            acc[cid] = {
                'fecha_min': None,
                'fecha_max': None,
                'total_unidades': 0.0,
                'mnt_bruto_sum': 0.0,
                'mnt_bruto_cnt': 0,
                'mnt_neto_sum': 0.0,
                'mnt_neto_cnt': 0,
                'total_tickets': 0.0,
                'farmacias': set(),
                'region_last': None,
                'edad_last': np.nan,
                'sexo_last': None,
                'mes_last': np.nan,
                'dow_last': np.nan,
                'mnt_ult90_sum': 0.0,
                'mnt_ant_sum': 0.0,
            }
        return acc[cid]

    for chunk in pd.read_csv(
        csv_path,
        encoding=enc_found,
        usecols=cols,
        dtype=dtypes,
        parse_dates=['FCH_FECHA'],
        engine='python',
        chunksize=chunksize,
    ):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        chunk = chunk.dropna(subset=['FCH_FECHA','COD_NRO_TARJETA']).sort_values('FCH_FECHA')

        # Simple aggregates by group
        g = chunk.groupby('COD_NRO_TARJETA', as_index=False)
        g_basic = g.agg({
            'FCH_FECHA': ['min','max','last'],
            'CNT_UNIDADES': 'sum',
            'MNT_VENTA_BRUTA': ['sum','count'],
            'MNT_VENTA_NETA': ['sum','count'],
            'TICKETS': 'sum',
            'REGION': 'last',
            'EDAD': 'last',
            'SEXO': 'last',
        })
        g_basic.columns = [
            'COD_NRO_TARJETA','fecha_min_c','fecha_max_c','fecha_last_c',
            'total_unidades_c','mnt_bruto_sum_c','mnt_bruto_cnt_c',
            'mnt_neto_sum_c','mnt_neto_cnt_c','total_tickets_c',
            'region_last_c','edad_last_c','sexo_last_c'
        ]

        # Farmacias distintas en el chunk (unir como set luego)
        g_ph = (
            g['COD_FARMACIA_SAP']
            .agg(lambda s: set(pd.unique(s.dropna())))
            .reset_index(name='farm_set')
        )

        # Merge temp frames
        merged = pd.merge(g_basic, g_ph, on='COD_NRO_TARJETA', how='left')

        # Sumas 90
        chunk['ES_ULT_90'] = (chunk['FCH_FECHA'] > corte_90).astype(int)
        g90 = chunk.groupby(['COD_NRO_TARJETA','ES_ULT_90'])['MNT_VENTA_NETA'].sum().unstack(fill_value=0.0)
        g90 = g90.rename(columns={0:'ant',1:'ult'}).reset_index()
        merged = merged.merge(g90, on='COD_NRO_TARJETA', how='left').fillna({'ant':0.0,'ult':0.0})

        # Mes/DOW de la última compra en el chunk
        last_k = merged['fecha_last_c']
        merged['mes_last_c'] = pd.to_datetime(last_k).dt.month.astype('float32')
        merged['dow_last_c'] = pd.to_datetime(last_k).dt.dayofweek.astype('float32')

        # Acumular en dict
        for row in merged.itertuples(index=False):
            cid = row.COD_NRO_TARJETA
            st = _ensure(cid)
            # fechas
            st['fecha_min'] = row.fecha_min_c if st['fecha_min'] is None or row.fecha_min_c < st['fecha_min'] else st['fecha_min']
            st['fecha_max'] = row.fecha_max_c if st['fecha_max'] is None or row.fecha_max_c > st['fecha_max'] else st['fecha_max']
            # sumas/cuentas
            st['total_unidades'] += float(row.total_unidades_c)
            st['mnt_bruto_sum'] += float(row.mnt_bruto_sum_c)
            st['mnt_bruto_cnt'] += int(row.mnt_bruto_cnt_c)
            st['mnt_neto_sum'] += float(row.mnt_neto_sum_c)
            st['mnt_neto_cnt'] += int(row.mnt_neto_cnt_c)
            st['total_tickets'] += float(row.total_tickets_c)
            # farmacias
            try:
                for v in row.farm_set:
                    st['farmacias'].add(v)
            except Exception:
                pass
            # ultimos valores
            st['region_last'] = row.region_last_c if pd.notna(row.region_last_c) else st['region_last']
            st['edad_last'] = float(row.edad_last_c) if pd.notna(row.edad_last_c) else st['edad_last']
            st['sexo_last'] = row.sexo_last_c if pd.notna(row.sexo_last_c) else st['sexo_last']
            st['mes_last'] = float(row.mes_last_c) if pd.notna(row.mes_last_c) else st['mes_last']
            st['dow_last'] = float(row.dow_last_c) if pd.notna(row.dow_last_c) else st['dow_last']
            # 90d
            st['mnt_ult90_sum'] += float(row.ult)
            st['mnt_ant_sum'] += float(row.ant)

    # Construir DataFrame final
    records = []
    for cid, st in acc.items():
        records.append({
            'COD_NRO_TARJETA': cid,
            'FECHA_PRIMERA_COMPRA': st['fecha_min'],
            'FECHA_ULTIMA_COMPRA': st['fecha_max'],
            'TOTAL_UNIDADES': st['total_unidades'],
            'MNT_BRUTO_PROM': (st['mnt_bruto_sum'] / st['mnt_bruto_cnt']) if st['mnt_bruto_cnt'] else 0.0,
            'MNT_BRUTO_TOT': st['mnt_bruto_sum'],
            'MNT_NETO_PROM': (st['mnt_neto_sum'] / st['mnt_neto_cnt']) if st['mnt_neto_cnt'] else 0.0,
            'MNT_NETO_TOT': st['mnt_neto_sum'],
            'TOTAL_TICKETS': st['total_tickets'],
            'FARMACIAS_DISTINTAS': len(st['farmacias']) if isinstance(st['farmacias'], set) else 0,
            'REGION_MAS_FRECUENTE': st['region_last'],  # aproximación: última
            'EDAD': st['edad_last'],
            'SEXO': st['sexo_last'],
            'MES_FRECUENTE': st['mes_last'],            # aproximación: última
            'DOW_FRECUENTE': st['dow_last'],            # aproximación: última
            'VAR_GASTO_90': ((st['mnt_ult90_sum'] - st['mnt_ant_sum']) / (st['mnt_ant_sum'] + 1e-6)),
        })

    agg_df = pd.DataFrame.from_records(records)

    # Derivadas y label
    ref_date = agg_df['FECHA_ULTIMA_COMPRA'].max()
    agg_df['DIAS_DESDE_ULTIMA'] = (ref_date - pd.to_datetime(agg_df['FECHA_ULTIMA_COMPRA'])).dt.days
    agg_df['ANTIGUEDAD_CLIENTE'] = (ref_date - pd.to_datetime(agg_df['FECHA_PRIMERA_COMPRA'])).dt.days
    agg_df['FRECUENCIA_MENSUAL'] = agg_df['TOTAL_TICKETS'] / (agg_df['ANTIGUEDAD_CLIENTE'] / 30.0 + 1)
    agg_df['TICKET_PROMEDIO'] = np.where(
        agg_df['TOTAL_TICKETS'] > 0,
        agg_df['MNT_NETO_TOT'] / agg_df['TOTAL_TICKETS'],
        0.0,
    )
    agg_df['ABANDONO'] = (agg_df['DIAS_DESDE_ULTIMA'] > umbral_abandono_dias).astype(int)

    feature_cols = [
        'EDAD','SEXO','REGION_MAS_FRECUENTE','TOTAL_TICKETS','FRECUENCIA_MENSUAL',
        'TICKET_PROMEDIO','MNT_NETO_TOT','MNT_NETO_PROM','DIAS_DESDE_ULTIMA',
        'ANTIGUEDAD_CLIENTE','FARMACIAS_DISTINTAS','MES_FRECUENTE','DOW_FRECUENTE',
        'VAR_GASTO_90'
    ]
    df_features = agg_df[['COD_NRO_TARJETA'] + feature_cols + ['ABANDONO']].copy()
    num_cols = [c for c in feature_cols if c not in ['SEXO','REGION_MAS_FRECUENTE']]
    df_features[num_cols] = df_features[num_cols].fillna(0)
    df_model = pd.get_dummies(
        df_features,
        columns=['SEXO','REGION_MAS_FRECUENTE'],
        drop_first=True
    )
    return agg_df, df_model


def entrenar_y_evaluar(model_type: str, X: pd.DataFrame, y: np.ndarray, test_size: float, random_state: int):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    model = None
    feature_importances = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_type == 'xgb':
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=-1,
            )
        except Exception as e:
            raise RuntimeError("xgboost no esta disponible. Instalala o usa --model rf") from e
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced_subsample'
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'classification_report': classification_report(y_test, y_pred, digits=3, output_dict=True),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'pr_auc': float(average_precision_score(y_test, y_proba)),
    }

    # Importancias
    try:
        if model_type == 'xgb' and hasattr(model, 'get_booster'):
            booster = model.get_booster()
            gain = booster.get_score(importance_type='gain')
            fi = pd.DataFrame({'feature': list(X.columns)})
            fi['gain'] = fi['feature'].map(gain).fillna(0.0)
            fi = fi.sort_values('gain', ascending=False).reset_index(drop=True)
            feature_importances = fi.rename(columns={'gain': 'importance'})
        elif hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
            feature_importances = fi.sort_values('importance', ascending=False).reset_index(drop=True)
    except Exception:
        feature_importances = None

    return model, metrics, feature_importances


def bucket_riesgo(p: float, thr: Tuple[float, float, float]) -> str:
    t1, t2, t3 = thr
    if p < t1:
        return 'Bajo'
    if p < t2:
        return 'Medio'
    if p < t3:
        return 'Alto'
    return 'Critico'


def exportar_salidas(agg_df: pd.DataFrame, df_model: pd.DataFrame, X: pd.DataFrame, model, metrics: Dict,
                     feature_importances: Optional[pd.DataFrame], cfg: Config, rewrite_datasets: bool = True):
    ensure_dir(cfg.output_dir)
    features_path = os.path.join(cfg.output_dir, 'features_clientes.csv')
    model_path = os.path.join(cfg.output_dir, 'dataset_modelo.csv')
    if rewrite_datasets:
        agg_df.to_csv(features_path, index=False, encoding='utf-8-sig')
        df_model.to_csv(model_path, index=False, encoding='utf-8-sig')
        print(f"[OK] Features por cliente guardadas en: {features_path}")
        print(f"[OK] Dataset para modelo guardado en:    {model_path}")

    if model is not None:
        # Predicciones para todos los clientes
        proba_all = model.predict_proba(X)[:, 1]
        out_pred = df_model[['COD_NRO_TARJETA']].copy()
        out_pred['p_abandono'] = proba_all
        out_pred['riesgo'] = [bucket_riesgo(p, cfg.risk_thresholds) for p in proba_all]
        extra = agg_df[['COD_NRO_TARJETA','DIAS_DESDE_ULTIMA','ABANDONO','FRECUENCIA_MENSUAL','TICKET_PROMEDIO']]
        out_pred = out_pred.merge(extra, on='COD_NRO_TARJETA', how='left')
        pred_path = os.path.join(cfg.output_dir, 'predicciones_abandono.csv')
        out_pred.to_csv(pred_path, index=False, encoding='utf-8-sig')
        print(f"[OK] Predicciones guardadas en:         {pred_path}")

        # Importancias del modelo
        if feature_importances is not None:
            fi_path = os.path.join(cfg.output_dir, 'importancias_modelo.csv')
            feature_importances.to_csv(fi_path, index=False, encoding='utf-8-sig')
            print(f"[OK] Importancias guardadas en:       {fi_path}")

        # Metricas
        metrics_path = os.path.join(cfg.output_dir, 'metricas_modelo.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[OK] Metricas guardadas en:            {metrics_path}")


def preparar_datasets(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print('[INFO] Cargando y preparando datos...')
    try:
        df = cargar_y_limpiar(cfg.csv_path)
        agg_df, df_model = construir_features_y_label(df, cfg.umbral_abandono_dias)
    except Exception as e:
        print('[AVISO] Carga completa falló (', type(e).__name__, '). Usando procesamiento por chunks...', sep='')
        agg_df, df_model = construir_features_y_label_en_chunks(
            cfg.csv_path,
            cfg.umbral_abandono_dias,
            chunksize=cfg.chunk_size,
        )
    ensure_dir(cfg.output_dir)
    features_path = os.path.join(cfg.output_dir, 'features_clientes.csv')
    model_path = os.path.join(cfg.output_dir, 'dataset_modelo.csv')
    agg_df.to_csv(features_path, index=False, encoding='utf-8-sig')
    df_model.to_csv(model_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Features por cliente guardadas en: {features_path}")
    print(f"[OK] Dataset para modelo guardado en:    {model_path}")
    return agg_df, df_model


def entrenar_pipeline(agg_df: pd.DataFrame, df_model: pd.DataFrame, cfg: Config):
    print('[INFO] Entrenando y evaluando modelo...')
    y = df_model['ABANDONO'].astype(int).values
    X = df_model.drop(columns=['ABANDONO', 'COD_NRO_TARJETA'])
    model, metrics, feature_importances = entrenar_y_evaluar(
        cfg.model_type,
        X,
        y,
        cfg.test_size,
        cfg.random_state,
    )
    exportar_salidas(agg_df, df_model, X, model, metrics, feature_importances, cfg, rewrite_datasets=False)
    if cfg.shap_enabled:
        print('[INFO] Calculando SHAP...')
        intentar_shap(model, X, cfg.output_dir)
    return model, metrics, feature_importances


def entrenar_desde_archivos(features_path: str, dataset_path: str, cfg: Config):
    print('[INFO] Leyendo datasets preparados...')
    agg_df = pd.read_csv(features_path, encoding='utf-8-sig')
    df_model = pd.read_csv(dataset_path, encoding='utf-8-sig')
    entrenar_pipeline(agg_df, df_model, cfg)


def intentar_shap(model, X: pd.DataFrame, output_dir: str):
    try:
        import shap
        warnings.filterwarnings('ignore', category=UserWarning)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        shap_imp = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
        shap_path = os.path.join(output_dir, 'shap_importance.csv')
        shap_imp.to_csv(shap_path, index=False, encoding='utf-8-sig')
        print(f"[OK] SHAP importance guardado en:      {shap_path}")
    except Exception as e:
        print("[AVISO] SHAP no disponible o fallo el calculo.")
        print("Detalle:", e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Modelo de abandono: features, entrenamiento y prediccion')
    default_csv = os.path.join(os.path.dirname(__file__), 'data', 'sales_lms_clientes.csv')
    default_out = os.path.dirname(default_csv)
    parser.add_argument('--csv', dest='csv_path', default=default_csv, help='Ruta del CSV de ventas por cliente')
    parser.add_argument('--out', dest='output_dir', default=default_out, help='Carpeta de salida para archivos generados')
    parser.add_argument('--umbral', dest='umbral', type=int, default=90, help='Dias sin compra para marcar abandono')
    parser.add_argument('--model', dest='model', choices=['xgb','rf'], default='xgb', help='Modelo: xgb (XGBoost) o rf (RandomForest)')
    parser.add_argument('--no-train', dest='no_train', action='store_true', help='Solo preparar datasets, no entrenar')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.30, help='Tamano de test (0-1)')
    parser.add_argument('--seed', dest='seed', type=int, default=42, help='Semilla aleatoria')
    parser.add_argument('--shap', dest='shap', action='store_true', help='Calcular explicabilidad SHAP (si disponible)')
    parser.add_argument('--thr', dest='thr', default='0.2,0.4,0.6', help='Umbrales de riesgo bajo,medio,alto (tres valores)')
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=200_000,
                        help='Filas por chunk para el modo de bajo consumo (solo preparación)')

    args = parser.parse_args()
    try:
        thr_tuple = tuple(float(x) for x in args.thr.split(','))
        if len(thr_tuple) != 3 or not all(0.0 <= t <= 1.0 for t in thr_tuple):
            raise ValueError
    except Exception:
        raise ValueError('Parametro --thr invalido. Ejemplo: 0.2,0.4,0.6')

    cfg = Config(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        umbral_abandono_dias=args.umbral,
        test_size=args.test_size,
        random_state=args.seed,
        model_type=args.model,
        train_model=(not args.no_train),
        shap_enabled=args.shap,
        risk_thresholds=thr_tuple,
        chunk_size=args.chunk_size,
    )

    agg_df, df_model = preparar_datasets(cfg)

    if cfg.train_model:
        entrenar_pipeline(agg_df, df_model, cfg)
    else:
        print('[INFO] Datasets preparados. Ejecuta con entrenamiento para generar predicciones.')


if __name__ == '__main__':
    main()

"""
Notas:
- `--umbral` controla cuando consideras abandono (90/120/180 dias).
- Puedes usar `--model rf` si no tienes XGBoost disponible.
- `--thr` define cortes de riesgo para segmentar en Bajo/Medio/Alto/Critico.
- Para validacion temporal, arma un split por fecha en vez de aleatorio.
"""
