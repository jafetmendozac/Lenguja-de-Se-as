"""
ETAPA 5 — Evaluación detallada del modelo LSP.

Genera:
  - Matriz de confusión (gráfica y en texto)
  - Importancia de features
  - Reporte completo por clase
  - Guarda gráficas en reports/

Uso:
    python scripts/evaluate_model.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, ".")

from src.recognition.model import SignClassifier
from config.settings import PROCESSED, MODEL_PATH, CLASSES_DISPLAY


REPORTS_DIR = Path("reports")


def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path(PROCESSED)
    if not csv_path.exists():
        print(f"[ERROR] Dataset no encontrado: {csv_path}")
        sys.exit(1)

    df           = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "label"]
    X            = df[feature_cols].values.astype(np.float32)
    y            = df["label"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, classes: list[str]) -> None:
    """Genera y guarda la matriz de confusión normalizada."""
    cm      = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    labels  = [CLASSES_DISPLAY.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title("Matriz de confusión — LSP (normalizada)", fontsize=13, pad=12)
    plt.tight_layout()

    out = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [OK] Matriz de confusión → {out}")


def plot_feature_importance(importances: np.ndarray, n_top: int = 20) -> None:
    """
    Muestra los N landmarks más relevantes para la clasificación.

    Cada feature es lm{i}_{eje}: lm0_x = coordenada x del landmark 0 (muñeca), etc.
    """
    axes   = ["x", "y", "z"]
    names  = [f"lm{i}_{ax}" for i in range(21) for ax in axes]

    # Agrupa por landmark sumando x+y+z
    lm_importance = np.zeros(21)
    for idx, imp in enumerate(importances):
        lm_idx = idx // 3
        lm_importance[lm_idx] += imp

    lm_names   = [f"Punto {i}" for i in range(21)]
    sorted_idx = np.argsort(lm_importance)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [lm_names[i] for i in sorted_idx][::-1],
        lm_importance[sorted_idx][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importancia relativa", fontsize=11)
    ax.set_title(f"Top {n_top} landmarks más importantes — LSP", fontsize=12, pad=10)
    plt.tight_layout()

    out = REPORTS_DIR / "feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [OK] Importancia de features → {out}")


def print_detailed_report(y_true, y_pred, confs, classes) -> None:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Precisión global:  {acc * 100:.2f}%")
    print(f"  Confianza media:   {confs.mean() * 100:.2f}%")
    print(f"  Confianza mínima:  {confs.min() * 100:.2f}%\n")

    labels = [CLASSES_DISPLAY.get(c, c) for c in classes]
    print(classification_report(y_true, y_pred, labels=classes,
                                 target_names=labels, zero_division=0))


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 55)
    print("  Evaluación del modelo — LSP Recognition System")
    print("=" * 55)

    # Cargar modelo
    clf = SignClassifier.load(MODEL_PATH)
    print(f"\n  Modelo cargado: {MODEL_PATH}")
    print(f"  Clases: {list(clf.classes)}")

    # Cargar test set (misma semilla que en train_model.py → mismo split)
    X_test, y_test = load_test_set()
    print(f"  Muestras de test: {len(X_test)}")

    # Predicción
    y_pred, confs = clf.predict_batch(X_test)
    classes       = list(clf.classes)

    # Reporte en consola
    print_detailed_report(y_test, y_pred, confs, classes)

    # Gráficas
    print("  Generando gráficas...")
    plot_confusion_matrix(y_test, y_pred, classes)
    plot_feature_importance(clf.feature_importances, n_top=15)

    print(f"\n  Gráficas guardadas en: {REPORTS_DIR}/")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
