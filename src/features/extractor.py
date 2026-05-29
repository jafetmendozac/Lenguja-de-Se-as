"""
Módulo de extracción y normalización de features a partir de landmarks de mano.

Por qué normalizar:
  Los landmarks crudos dependen de dónde está la mano en el frame y qué tan
  cerca está de la cámara. Si entrenamos con esos valores, el modelo aprendería
  posición/tamaño en vez de forma de la seña.

Estrategia de normalización:
  1. Translación  → restar la posición de la muñeca (punto 0) a todos los puntos.
  2. Escalado     → dividir por la distancia máxima entre la muñeca y cualquier
                    otro punto.

  Resultado: vector de 63 valores en [-1, 1] que representa SOLO la forma
  de la mano, independiente de posición y escala.

  Invariancias obtenidas:   posición ✓   escala ✓
  Invariancias NO buscadas: rotación ✗   (distintas señas tienen distinta orientación)
"""

import numpy as np


class FeatureExtractor:
    """
    Convierte un array de landmarks (21, 3) en un vector de features normalizado (63,).

    Uso:
        extractor = FeatureExtractor()
        features  = extractor.extract(landmarks_array)   # → np.ndarray (63,)
    """

    N_LANDMARKS = 21
    N_COORDS    = 3                         # x, y, z
    N_FEATURES  = N_LANDMARKS * N_COORDS    # 63

    def extract(self, landmarks: np.ndarray) -> np.ndarray | None:
        """
        Normaliza los landmarks y devuelve el vector de features.

        Args:
            landmarks: Array de forma (21, 3) con coordenadas normalizadas [0, 1]
                       directamente desde MediaPipe.

        Returns:
            Array 1D de 63 features en [-1, 1], o None si la entrada es inválida.
        """
        if landmarks is None:
            return None

        if landmarks.shape != (self.N_LANDMARKS, self.N_COORDS):
            return None

        # Paso 1: trasladar al origen (muñeca = punto 0)
        wrist      = landmarks[0].copy()
        translated = landmarks - wrist

        # Paso 2: escalar por la distancia máxima desde la muñeca
        distances = np.linalg.norm(translated, axis=1)
        max_dist  = distances.max()
        if max_dist > 1e-6:
            normalized = translated / max_dist
        else:
            # Mano demasiado pequeña o datos corruptos
            normalized = translated

        return normalized.flatten().astype(np.float32)

    def feature_names(self) -> list[str]:
        """Devuelve nombres de columna para el CSV del dataset."""
        axes = ["x", "y", "z"]
        return [f"lm{i}_{ax}" for i in range(self.N_LANDMARKS) for ax in axes]
