"""
ETAPA 5 — Entrenamiento del modelo LSP.

Flujo:
  1. Carga data/processed/landmarks.csv
  2. Divide en train (80%) / test (20%) estratificado por clase
  3. Entrena Random Forest
  4. Validación cruzada 5-fold sobre el train set
  5. Evalúa sobre el test set
  6. Guarda el modelo en models/sign_classifier.pkl

Uso:
    python scripts/train_model.py
    python scripts/train_model.py --no-cv     # Saltar validación cruzada (más rápido)
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, ".")

from src.recognition.model import SignClassifier
from src.features.extractor import FeatureExtractor
from config.settings import PROCESSED, MODEL_PATH, CLASSES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento del modelo LSP")
    p.add_argument("--no-cv", action="store_true", help="Omitir validación cruzada")
    return p.parse_args()


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga el CSV y separa features (X) de etiquetas (y)."""
    if not csv_path.exists():
        print(f"[ERROR] No se encontró el dataset: {csv_path}")
        print("  → Ejecuta primero: python scripts/collect_data.py --class <seña>")
        print("  → Luego:           python scripts/extract_landmarks.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[OK] Dataset cargado: {len(df)} muestras, {df['label'].nunique()} clases")

    # Verificar clases presentes
    present  = sorted(df["label"].unique())
    missing  = [c for c in CLASSES if c not in present]
    if missing:
        print(f"[AVISO] Clases sin datos: {missing}")
        print("  El modelo solo reconocerá las clases con datos disponibles.")

    # Distribución
    print("\n  Muestras por clase:")
    for cls, n in df["label"].value_counts().sort_index().items():
        bar = "█" * (n // 10)
        print(f"    {cls:<10} {n:>4}  {bar}")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    return X, y


def cross_validate_model(X: np.ndarray, y: np.ndarray) -> None:
    """Validación cruzada estratificada 5-fold para estimar la precisión real."""
    print("\n  Ejecutando validación cruzada 5-fold...")
    clf = SignClassifier()
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = cross_validate(
        clf._clf, X, y,
        cv=cv,
        scoring=["accuracy", "f1_weighted"],
        return_train_score=True,
        n_jobs=-1,
    )

    acc_train = results["train_accuracy"]
    acc_val   = results["test_accuracy"]
    f1_val    = results["test_f1_weighted"]

    print(f"\n  {'Fold':<6} {'Train Acc':>10} {'Val Acc':>10} {'Val F1':>10}")
    print("  " + "-" * 40)
    for i, (tr, va, f1) in enumerate(zip(acc_train, acc_val, f1_val), 1):
        print(f"  {i:<6} {tr:>10.4f} {va:>10.4f} {f1:>10.4f}")
    print("  " + "-" * 40)
    print(f"  {'Media':<6} {acc_train.mean():>10.4f} {acc_val.mean():>10.4f} {f1_val.mean():>10.4f}")
    print(f"  {'±Std':<6} {acc_train.std():>10.4f} {acc_val.std():>10.4f} {f1_val.std():>10.4f}")

    # Advertir si hay sobreajuste significativo
    gap = acc_train.mean() - acc_val.mean()
    if gap > 0.05:
        print(f"\n  [AVISO] Brecha train-val = {gap:.3f}. Considera más datos o min_samples_leaf mayor.")


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 55)
    print("  Entrenamiento — LSP Recognition System")
    print("=" * 55)

    # ── 1. Cargar dataset ────────────────────────────────────────────────────
    X, y = load_dataset(Path(PROCESSED))

    if len(np.unique(y)) < 2:
        print("[ERROR] Se necesitan al menos 2 clases para entrenar.")
        sys.exit(1)

    # ── 2. División train / test ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,        # Mantiene la proporción de clases en ambos conjuntos
    )
    print(f"\n  Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

    # ── 3. Validación cruzada (opcional) ────────────────────────────────────
    if not args.no_cv:
        cross_validate_model(X_train, y_train)

    # ── 4. Entrenamiento final ───────────────────────────────────────────────
    print("\n  Entrenando modelo final sobre el train set completo...")
    clf = SignClassifier()
    clf.fit(X_train, y_train)
    print("  Entrenamiento completado.")

    # ── 5. Evaluación en test set ────────────────────────────────────────────
    y_pred, confs = clf.predict_batch(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 55}")
    print(f"  Precisión en test set: {acc * 100:.2f}%")
    print(f"  Confianza media:       {confs.mean() * 100:.2f}%")
    print(f"{'=' * 55}")

    print("\n  Reporte por clase (test set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── 6. Guardar modelo ────────────────────────────────────────────────────
    clf.save(MODEL_PATH)

    print(f"\n  Clases en el modelo: {list(clf.classes)}")
    print("\n  Siguiente paso: python scripts/evaluate_model.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
