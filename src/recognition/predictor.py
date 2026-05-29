"""
Módulo de predicción en tiempo real de señas LSP.

Optimizaciones implementadas (ETAPA 8):
  1. Suavizado temporal (mayoría N-1/N) — reduce parpadeo de etiqueta.
  2. Confidence smoothing (promedio deslizante) — suaviza la barra de confianza.
  3. Landmark delta — si la mano no se movió lo suficiente, reutiliza la última
     predicción sin llamar al modelo. Ahorra ~1ms por frame cuando la mano está quieta
     y elimina predicciones ruidosas por micro-tembleos.

Por qué NO se implementó resolución reducida:
  Benchmark en M2 mostró que MediaPipe tarda ~35ms a 640x480 y ~34ms a 320x240.
  El cuello de botella es la red neuronal (XNNPACK), no la resolución del input.
  Reducir resolución agregaría complejidad sin beneficio medible en este hardware.
"""

import numpy as np
from collections import deque, Counter

from src.recognition.model import SignClassifier
from src.features.extractor import FeatureExtractor
from config.settings import (
    MODEL_PATH,
    PREDICTION_THRESHOLD,
    PREDICTION_HISTORY,
    CONFIDENCE_SMOOTH_WINDOW,
    LANDMARK_CHANGE_THRESHOLD,
    CLASSES_DISPLAY,
)


class PredictionState:
    """Resultado de una actualización del predictor."""
    __slots__ = ("label", "display_label", "confidence", "is_stable")

    def __init__(self, label: str | None, confidence: float, is_stable: bool) -> None:
        self.label         = label
        self.display_label = CLASSES_DISPLAY.get(label, label) if label else ""
        self.confidence    = confidence
        self.is_stable     = is_stable

    def __bool__(self) -> bool:
        return self.label is not None


class SignPredictor:
    """
    Predice señas LSP en tiempo real con suavizado temporal y optimizaciones.

    Uso:
        predictor = SignPredictor()
        state = predictor.update(landmarks)   # landmarks: (21,3) o None
        if state.is_stable:
            print(state.display_label, state.confidence)
    """

    def __init__(self) -> None:
        self._clf       = SignClassifier.load(MODEL_PATH)
        self._extractor = FeatureExtractor()

        # Suavizado temporal: últimas N etiquetas para majority vote
        self._label_history: deque[str | None] = deque(maxlen=PREDICTION_HISTORY)

        # Confidence smoothing: promedio deslizante de las últimas N confianzas
        self._conf_history: deque[float] = deque(maxlen=CONFIDENCE_SMOOTH_WINDOW)

        # Landmark delta: guardar landmarks del frame anterior
        self._prev_landmarks: np.ndarray | None = None

        # Caché de la última predicción para reutilizar cuando la mano está quieta
        self._cached_label: str | None = None
        self._cached_conf:  float      = 0.0

    # ── Inferencia ────────────────────────────────────────────────────────────

    def update(self, landmarks: np.ndarray | None) -> PredictionState:
        """
        Actualiza el predictor con los landmarks del frame actual.

        Args:
            landmarks: Array (21, 3) de MediaPipe, o None si no hay mano.

        Returns:
            PredictionState con label, confidence suavizada e is_stable.
        """
        if landmarks is None:
            self._reset_state()
            return PredictionState(None, 0.0, False)

        features = self._extractor.extract(landmarks)
        if features is None:
            self._reset_state()
            return PredictionState(None, 0.0, False)

        # ── Optimización: landmark delta ─────────────────────────────────────
        # Si la mano no se movió significativamente, reutilizar la predicción anterior.
        if self._hand_is_static(landmarks):
            label = self._cached_label
            conf  = self._cached_conf
        else:
            label, conf = self._clf.predict(features)
            self._cached_label = label
            self._cached_conf  = conf
            self._prev_landmarks = landmarks.copy()

        # ── Umbral de confianza ───────────────────────────────────────────────
        if conf < PREDICTION_THRESHOLD:
            self._label_history.append(None)
            self._conf_history.append(conf)
            smooth_conf = float(np.mean(self._conf_history))
            return PredictionState(label, smooth_conf, False)

        # ── Suavizado temporal (majority vote) ───────────────────────────────
        self._label_history.append(label)
        self._conf_history.append(conf)
        smooth_conf = float(np.mean(self._conf_history))
        is_stable   = self._is_stable(label)

        return PredictionState(label, smooth_conf, is_stable)

    def reset(self) -> None:
        """Limpia todo el estado interno. Útil al presionar R."""
        self._reset_state()

    @property
    def is_ready(self) -> bool:
        return True

    # ── Lógica interna ────────────────────────────────────────────────────────

    def _hand_is_static(self, landmarks: np.ndarray) -> bool:
        """
        Devuelve True si la mano no se movió suficiente desde el frame anterior.
        Se usa para evitar llamadas innecesarias al modelo.
        """
        if self._prev_landmarks is None or self._cached_label is None:
            return False
        delta = float(np.mean(np.abs(landmarks - self._prev_landmarks)))
        return delta < LANDMARK_CHANGE_THRESHOLD

    def _is_stable(self, current_label: str) -> bool:
        """Predicción estable si aparece en N-1 de los últimos N frames."""
        if len(self._label_history) < self._label_history.maxlen:
            return False
        most_common_label, count = Counter(self._label_history).most_common(1)[0]
        required = self._label_history.maxlen - 1
        return most_common_label == current_label and count >= required

    def _reset_state(self) -> None:
        self._label_history.clear()
        self._conf_history.clear()
        self._prev_landmarks  = None
        self._cached_label    = None
        self._cached_conf     = 0.0
