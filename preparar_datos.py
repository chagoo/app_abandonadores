"""
Entrada dedicada para preparar features_clientes.csv y dataset_modelo.csv
sin ejecutar el entrenamiento del modelo.
"""

import argparse
import os

from modelo_abandonadores import Config, preparar_datasets


def build_parser() -> argparse.ArgumentParser:
    base_dir = os.path.dirname(__file__)
    default_csv = os.path.join(base_dir, 'data', 'sales_lms_clientes.csv')
    default_out = os.path.join(base_dir, 'data')

    parser = argparse.ArgumentParser(
        description='Prepara features y dataset del modelo de abandono'
    )
    parser.add_argument('--csv', dest='csv_path', default=default_csv,
                        help='Ruta del CSV de ventas por cliente')
    parser.add_argument('--out', dest='output_dir', default=default_out,
                        help='Carpeta de salida para archivos generados')
    parser.add_argument('--umbral', dest='umbral', type=int, default=90,
                        help='Dias sin compra para marcar abandono')
    parser.add_argument('--chunk-size', dest='chunk_size', type=int, default=200_000,
                        help='Filas por chunk en el modo de bajo consumo de memoria')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = Config(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        umbral_abandono_dias=args.umbral,
        train_model=False,
        shap_enabled=False,
        chunk_size=args.chunk_size,
    )
    preparar_datasets(cfg)


if __name__ == '__main__':
    main()
