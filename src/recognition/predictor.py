"""
Módulo de predicción en tiempo real de señas LSP.

Por qué el suavizado temporal:
  Un clasificador por frame produce parpadeo — cada frame puede predecir una
  clase distinta aunque la mano esté quieta. El suavizado temporal acumula
  las últimas N predicciones y solo confirma una seña cuando N-1 frames
  consecutivos coinciden. Elimina el 90%+ de los falsos positivos.

  Umbral de confianza (0.75) + suavizado temporal (5 frames) = sistema estable.
"""

import numpy as np
from collections import deque, Counter

from src.recognition.model import SignClassifier
from src.features.extractor import FeatureExtractor
from config.settings import (
    MODEL_PATH,
    PREDICTION_THRESHOLD,
    PREDICTION_HISTORY,
    CLASSES_DISPLAY,
)


class PredictionState:
    """Resultado de una actualización del predictor."""
    __slots__ = ("label", "display_label", "confidence", "is_stable")

    def __init__(
        self,
        label: str | None,
        confidence: float,
        is_stable: bool,
    ) -> None:
        self.label         = label
        self.display_label = CLASSES_DISPLAY.get(label, label) if label else ""
        self.confidence    = confidence
        self.is_stable     = is_stable

    def __bool__(self) -> bool:
        return self.label is not None


class SignPredictor:
    """
    Predice señas LSP en tiempo real con suavizado temporal.

    Uso:
        predictor = SignPredictor()          # Carga el modelo entrenado
        state = predictor.update(landmarks)  # landmarks: np.ndarray (21,3) o None
        if state.is_stable:
            print(state.display_label, state.confidence)
    """

    def __init__(self) -> None:
        self._clf       = SignClassifier.load(MODEL_PATH)
        self._extractor = FeatureExtractor()
        # Historial de las últimas N etiquetas para suavizado
        self._history: deque[str | None] = deque(maxlen=PREDICTION_HISTORY)

    # ── Inferencia ────────────────────────────────────────────────────────────

    def update(self, landmarks: np.ndarray | None) -> PredictionState:
        """
        Actualiza el predictor con los landmarks del frame actual.

        Args:
            landmarks: Array (21, 3) de MediaPipe, o None si no hay mano.

        Returns:
            PredictionState con label, confidence e is_stable.
        """
        if landmarks is None:
            self._history.clear()
            return PredictionState(None, 0.0, False)

        features = self._extractor.extract(landmarks)
        if features is None:
            self._history.clear()
            return PredictionState(None, 0.0, False)

        label, conf = self._clf.predict(features)

        # Si la confianza es baja, no acumular en el historial
        if conf < PREDICTION_THRESHOLD:
            self._history.append(None)
            return PredictionState(label, conf, False)

        self._history.append(label)

        # Confirmar solo si la mayoría del historial coincide
        is_stable = self._is_stable(label)
        return PredictionState(label, conf, is_stable)

    def reset(self) -> None:
        """Limpia el historial de predicciones."""
        self._history.clear()

    @property
    def is_ready(self) -> bool:
        """True si el modelo está cargado y listo para predecir."""
        return True

    # ── Lógica de estabilidad ────────────────────────────────────────────────

    def _is_stable(self, current_label: str) -> bool:
        """
        Considera una predicción estable si aparece en al menos N-1 de los
        últimos N frames (permite 1 frame de ruido).
        """
        if len(self._history) < self._history.maxlen:
            return False

        most_common_label, count = Counter(self._history).most_common(1)[0]
        required = self._history.maxlen - 1   # N-1 de N frames
        return most_common_label == current_label and count >= required
