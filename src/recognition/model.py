"""
Módulo del clasificador de señas LSP.

Por qué Random Forest sobre otras opciones:
  - Con solo 63 features (landmarks normalizados), los árboles de decisión
    funcionan excelentemente sin GPU ni datos masivos.
  - Invariante a la escala de features (no necesita StandardScaler).
  - predict_proba() da probabilidades calibradas → score de confianza confiable.
  - n_jobs=-1 usa todos los núcleos del M2 para inferencia paralela.
  - Fácil de sustituir: cambiar _clf por MLPClassifier o SVC sin tocar el resto.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from config.settings import MODEL_PATH, CLASSES


class SignClassifier:
    """
    Clasificador de señas LSP basado en Random Forest.

    Uso — entrenamiento:
        clf = SignClassifier()
        clf.fit(X_train, y_train)
        clf.save()

    Uso — inferencia en tiempo real:
        clf = SignClassifier.load()
        label, confidence = clf.predict(features)   # features: (63,)
    """

    def __init__(self) -> None:
        self._clf = RandomForestClassifier(
            n_estimators=300,       # Más árboles = más estable; costo asumible en M2
            max_depth=None,         # Árboles profundos; con 63 features el sobreajuste es bajo
            min_samples_split=4,    # Evita hojas con 1-2 muestras ruidosas
            min_samples_leaf=2,
            class_weight="balanced",# Compensa si alguna clase tiene menos muestras
            random_state=42,
            n_jobs=-1,              # Paralelismo total en Apple M2 (8 cores)
        )
        self._is_trained = False

    # ── Entrenamiento ────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SignClassifier":
        """
        Entrena el clasificador.

        Args:
            X: Array (n_samples, 63) de features normalizados.
            y: Array (n_samples,) de etiquetas string (nombre de la seña).
        """
        self._clf.fit(X, y)
        self._is_trained = True
        return self

    # ── Inferencia ───────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """
        Predice la seña LSP para un vector de features.

        Args:
            features: Array (63,) de features normalizados.

        Returns:
            Tupla (label, confidence) donde confidence ∈ [0, 1].
        """
        self._check_trained()
        x = features.reshape(1, -1)
        probas    = self._clf.predict_proba(x)[0]
        class_idx = int(np.argmax(probas))
        label     = self._clf.classes_[class_idx]
        confidence = float(probas[class_idx])
        return label, confidence

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predice sobre múltiples muestras. Usado en evaluación.

        Returns:
            (labels_array, confidences_array)
        """
        self._check_trained()
        probas    = self._clf.predict_proba(X)
        indices   = np.argmax(probas, axis=1)
        labels    = self._clf.classes_[indices]
        confs     = probas[np.arange(len(probas)), indices]
        return labels, confs

    @property
    def classes(self) -> np.ndarray:
        self._check_trained()
        return self._clf.classes_

    @property
    def feature_importances(self) -> np.ndarray:
        self._check_trained()
        return self._clf.feature_importances_

    # ── Persistencia ─────────────────────────────────────────────────────────

    def save(self, path: str | Path = MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)
        print(f"[OK] Modelo guardado: {path}")

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "SignClassifier":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {path}\n"
                "Ejecuta primero: python scripts/train_model.py"
            )
        instance = cls.__new__(cls)
        instance._clf        = joblib.load(path)
        instance._is_trained = True
        return instance

    # ── Utilidades ───────────────────────────────────────────────────────────

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
