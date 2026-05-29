"""
Módulo de visualización.

Todas las operaciones de dibujo sobre el frame de OpenCV están aquí.
Ningún otro módulo debe llamar a cv2.putText / cv2.circle / cv2.line directamente.
"""

import cv2
import numpy as np

from config.settings import (
    COLOR_LANDMARKS,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_CONF,
    COLOR_BG_PANEL,
    COLOR_ALERT,
    FONT,
)
from src.detection.hand_detector import HAND_CONNECTIONS


class Display:
    """Renderiza landmarks, FPS, predicciones y paneles de estado sobre frames."""

    # Parámetros de dibujo
    LANDMARK_RADIUS      = 5
    LANDMARK_THICKNESS   = -1          # Relleno
    CONNECTION_THICKNESS = 2
    FONT_SCALE_LARGE     = 0.9
    FONT_SCALE_SMALL     = 0.55
    FONT_THICKNESS       = 2
    PANEL_HEIGHT         = 110
    PANEL_ALPHA          = 0.65        # Transparencia del panel inferior

    def draw_landmarks(self, frame: np.ndarray, result) -> None:
        """
        Dibuja los 21 landmarks y las conexiones del esqueleto de la mano.

        Los landmarks están normalizados [0, 1]; se multiplican por el tamaño
        del frame para obtener coordenadas de píxel.
        """
        if not result or not result.hand_landmarks:
            return

        h, w = frame.shape[:2]

        for hand_landmarks in result.hand_landmarks:
            # Convertir a coordenadas de píxel
            points = [
                (int(lm.x * w), int(lm.y * h))
                for lm in hand_landmarks
            ]

            # Dibujar conexiones primero (quedan debajo de los puntos)
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start_idx],
                    points[end_idx],
                    COLOR_LANDMARKS,
                    self.CONNECTION_THICKNESS,
                )

            # Dibujar círculos en cada landmark
            for px, py in points:
                cv2.circle(frame, (px, py), self.LANDMARK_RADIUS, COLOR_LANDMARKS, self.LANDMARK_THICKNESS)
                # Punto interior negro para resaltar
                cv2.circle(frame, (px, py), 2, (0, 0, 0), -1)

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Muestra el FPS en la esquina superior derecha."""
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, self.FONT_SCALE_SMALL, self.FONT_THICKNESS)
        x = w - tw - 12
        y = th + 12
        cv2.putText(frame, text, (x, y), FONT, self.FONT_SCALE_SMALL, (0, 0, 0), self.FONT_THICKNESS + 1)
        cv2.putText(frame, text, (x, y), FONT, self.FONT_SCALE_SMALL, COLOR_TEXT_CONF, self.FONT_THICKNESS)

    def draw_info_panel(
        self,
        frame: np.ndarray,
        sign: str = "",
        confidence: float = 0.0,
        status: str = "Buscando mano...",
        is_stable: bool = False,
    ) -> None:
        """
        Panel semitransparente en la parte inferior con seña, confianza y estado.

        is_stable=True  → seña confirmada (texto brillante + borde verde)
        is_stable=False → detección tentativa (texto amarillo más pequeño)
        """
        h, w = frame.shape[:2]
        panel_y = h - self.PANEL_HEIGHT

        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, self.PANEL_ALPHA, frame, 1 - self.PANEL_ALPHA, 0, frame)

        # Línea separadora: verde si estable, blanca si tentativa
        line_color = (0, 220, 0) if is_stable else (180, 180, 180)
        cv2.line(frame, (0, panel_y), (w, panel_y), line_color, 2 if is_stable else 1)

        if sign:
            if is_stable:
                # Seña CONFIRMADA — texto grande y brillante
                sign_text = sign.upper()
                cv2.putText(
                    frame, sign_text,
                    (16, panel_y + 48),
                    FONT, self.FONT_SCALE_LARGE * 1.4,
                    (0, 0, 0), self.FONT_THICKNESS + 2,
                )
                cv2.putText(
                    frame, sign_text,
                    (16, panel_y + 48),
                    FONT, self.FONT_SCALE_LARGE * 1.4,
                    COLOR_TEXT_MAIN, self.FONT_THICKNESS,
                )
                label_tag = "CONFIRMADA"
                tag_color = (0, 220, 0)
            else:
                # Seña TENTATIVA — texto más pequeño en amarillo
                sign_text = sign.upper()
                cv2.putText(
                    frame, sign_text,
                    (16, panel_y + 45),
                    FONT, self.FONT_SCALE_LARGE * 1.1,
                    (0, 200, 200), self.FONT_THICKNESS,
                )
                label_tag = "Analizando..."
                tag_color = (150, 150, 150)

            # Etiqueta de estado (CONFIRMADA / Analizando...)
            cv2.putText(frame, label_tag, (16, panel_y + 65),
                        FONT, 0.42, tag_color, 1)

            # Porcentaje de confianza
            conf_text = f"Confianza: {confidence * 100:.1f}%"
            cv2.putText(frame, conf_text, (16, panel_y + 82),
                        FONT, self.FONT_SCALE_SMALL, COLOR_TEXT_CONF, 1)

            # Barra de confianza
            bar_x, bar_y = 16, panel_y + 92
            bar_w = int((w - 32) * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w - 32, bar_y + 8), (50, 50, 50), -1)
            color_bar = (0, 200, 0) if is_stable else (COLOR_LANDMARKS if confidence >= 0.75 else COLOR_ALERT)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), color_bar, -1)
        else:
            # Sin predicción: mostrar estado
            cv2.putText(
                frame, status,
                (16, panel_y + 55),
                FONT, self.FONT_SCALE_SMALL,
                (160, 160, 160), 1,
            )

    def draw_hand_label(self, frame: np.ndarray, result) -> None:
        """Muestra si la mano detectada es izquierda o derecha."""
        if not result or not result.handedness:
            return

        h, w = frame.shape[:2]
        for i, handedness in enumerate(result.handedness):
            label = handedness[0].category_name
            score = handedness[0].score

            if result.hand_landmarks and i < len(result.hand_landmarks):
                # Posicionar el texto cerca de la muñeca (punto 0)
                wrist = result.hand_landmarks[i][0]
                px = int(wrist.x * w)
                py = max(int(wrist.y * h) - 15, 20)
                text = f"{label} ({score:.0%})"
                cv2.putText(frame, text, (px, py), FONT, self.FONT_SCALE_SMALL, COLOR_TEXT_CONF, 1)

    def draw_no_hand_warning(self, frame: np.ndarray) -> None:
        """Muestra aviso cuando no se detecta mano."""
        h, w = frame.shape[:2]
        text = "Sin mano detectada"
        (tw, _), _ = cv2.getTextSize(text, FONT, self.FONT_SCALE_SMALL, 1)
        x = (w - tw) // 2
        cv2.putText(frame, text, (x, 30), FONT, self.FONT_SCALE_SMALL, (100, 100, 255), 1)
