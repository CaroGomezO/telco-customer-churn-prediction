from __future__ import annotations

import argparse
import sys

from pipeline import train_and_save


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena y versiona el modelo de churn.")
    parser.add_argument("--data", required=True, help="CSV etiquetado.")
    parser.add_argument("--target", default="Churn", help="Columna objetivo (por defecto: Churn).")
    parser.add_argument("--output-dir", default="models", help="Directorio de versiones.")
    parser.add_argument("--version", required=True, help="Identificador, por ejemplo v1.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    try:
        version_dir = train_and_save(
            args.data, args.target, args.output_dir, args.version, args.seed, args.test_size
        )
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Modelo guardado en: {version_dir}")
    print(f"Métricas: {version_dir / 'metrics.json'}")
    print(f"Metadatos: {version_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
