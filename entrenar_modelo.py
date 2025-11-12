"""
Entrada dedicada para entrenar/predicir usando los datasets ya preparados.
"""

import argparse
import os
from typing import Tuple, cast

from modelo_abandonadores import Config, entrenar_desde_archivos


def build_parser() -> argparse.ArgumentParser:
    base_dir = os.path.dirname(__file__)
    default_out = os.path.join(base_dir, 'data')
    default_dataset = os.path.join(default_out, 'dataset_modelo.csv')
    default_features = os.path.join(default_out, 'features_clientes.csv')

    parser = argparse.ArgumentParser(
        description='Entrena el modelo de abandono usando datasets ya preparados'
    )
    parser.add_argument('--dataset', dest='dataset_path', default=default_dataset,
                        help='Ruta a dataset_modelo.csv')
    parser.add_argument('--features', dest='features_path', default=default_features,
                        help='Ruta a features_clientes.csv')
    parser.add_argument('--out', dest='output_dir', default=default_out,
                        help='Carpeta de salida para archivos generados')
    parser.add_argument('--model', dest='model', choices=['xgb', 'rf'], default='xgb',
                        help='Modelo: xgb (XGBoost) o rf (RandomForest)')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.30,
                        help='Tamaño de test (0-1)')
    parser.add_argument('--seed', dest='seed', type=int, default=42, help='Semilla aleatoria')
    parser.add_argument('--shap', dest='shap', action='store_true',
                        help='Calcular explicabilidad SHAP (si disponible)')
    parser.add_argument('--thr', dest='thr', default='0.2,0.4,0.6',
                        help='Umbrales de riesgo bajo,medio,alto (tres valores)')
    parser.add_argument('--umbral', dest='umbral', type=int, default=90,
                        help='Metadato: umbral de dias usado al preparar el dataset')
    return parser


def parse_thresholds(raw_thr: str) -> Tuple[float, float, float]:
    thr_tuple = tuple(float(x) for x in raw_thr.split(','))
    if len(thr_tuple) != 3 or not all(0.0 <= t <= 1.0 for t in thr_tuple):
        raise ValueError('Parametro --thr invalido. Ejemplo: 0.2,0.4,0.6')
    return cast(Tuple[float, float, float], thr_tuple)


def main():
    parser = build_parser()
    args = parser.parse_args()
    thr_tuple = parse_thresholds(args.thr)
    cfg = Config(
        csv_path=args.dataset_path,
        output_dir=args.output_dir,
        umbral_abandono_dias=args.umbral,
        test_size=args.test_size,
        random_state=args.seed,
        model_type=args.model,
        train_model=True,
        shap_enabled=args.shap,
        risk_thresholds=thr_tuple,
    )
    entrenar_desde_archivos(args.features_path, args.dataset_path, cfg)


if __name__ == '__main__':
    main()
