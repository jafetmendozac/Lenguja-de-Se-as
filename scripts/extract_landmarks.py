"""
ETAPA 4 — Extracción de landmarks a CSV.

Lee todas las imágenes de data/raw/{clase}/, extrae los 21 landmarks
con MediaPipe (modo IMAGE), normaliza con FeatureExtractor y guarda
el resultado en data/processed/landmarks.csv.

Ejecutar DESPUÉS de collect_data.py (cuando ya existen imágenes capturadas).

Uso:
    python scripts/extract_landmarks.py
    python scripts/extract_landmarks.py --show-stats
"""

import argparse
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, ".")

from src.detection.hand_detector import HandDetector
from src.features.extractor import FeatureExtractor
from config.settings import CLASSES, DATA_DIR, PROCESSED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extracción de landmarks a CSV")
    parser.add_argument("--show-stats", action="store_true",
                        help="Mostrar estadísticas del dataset al finalizar")
    return parser.parse_args()


def extract_all(detector: HandDetector, extractor: FeatureExtractor) -> pd.DataFrame:
    """
    Recorre data/raw/ y extrae features de cada imagen.
    Devuelve un DataFrame con 63 features + columna 'label'.
    """
    rows    = []
    skipped = 0
    feature_cols = extractor.feature_names()

    for sign_class in CLASSES:
        class_dir = Path(DATA_DIR) / sign_class
        if not class_dir.exists():
            print(f"  [AVISO] No existe directorio: {class_dir}")
            continue

        images = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        if not images:
            print(f"  [AVISO] Sin imágenes en: {class_dir}")
            continue

        print(f"\n  Procesando '{sign_class}' ({len(images)} imágenes)...")
        for img_path in tqdm(images, desc=f"  {sign_class}", unit="img"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                skipped += 1
                continue

            result    = detector.detect(frame)
            landmarks = detector.get_landmarks_array(result)
            features  = extractor.extract(landmarks)

            if features is None:
                # No se detectó mano en esta imagen — la descartamos
                skipped += 1
                continue

            row = dict(zip(feature_cols, features))
            row["label"] = sign_class
            rows.append(row)

    print(f"\n  Imágenes descartadas (sin mano detectada): {skipped}")
    return pd.DataFrame(rows)


def show_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("  Estadísticas del dataset")
    print("=" * 50)
    print(f"  Total de muestras: {len(df)}")
    print(f"  Features por muestra: {len(df.columns) - 1}")
    print("\n  Distribución por clase:")
    counts = df["label"].value_counts().sort_index()
    for label, n in counts.items():
        bar = "█" * (n // 5)
        print(f"    {label:<10} {n:>4}  {bar}")
    print()


def main() -> None:
    args = parse_args()

    output_path = Path(PROCESSED)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n[ETAPA 4] Extracción de landmarks → CSV")
    print(f"Origen:  data/raw/{{clase}}/")
    print(f"Destino: {output_path}\n")

    with HandDetector(mode="image") as detector:
        extractor = FeatureExtractor()
        df = extract_all(detector, extractor)

    if df.empty:
        print("[ERROR] No se extrajeron features. Verifica que existan imágenes en data/raw/")
        sys.exit(1)

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Dataset guardado: {output_path}  ({len(df)} filas)")

    if args.show_stats:
        show_stats(df)


if __name__ == "__main__":
    main()
