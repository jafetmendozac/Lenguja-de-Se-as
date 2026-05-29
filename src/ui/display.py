"""
Módulo de visualización — Interfaz final (ETAPA 9).

Layout del frame renderizado:
  ┌──────────────────────────────────────────────────┐
  │ ● ESTADO   Lengua de Señas Peruana (LSP)  FPS:XX │  ← barra superior (45px)
  ├──────────────────────────────────────────────────┤
  │                                                  │
  │          VIDEO + LANDMARKS DE LA MANO            │
  │                                                  │
  ├──────────────────────────────────────────────────┤
  │  SEÑA GRANDE          CONFIANZA%  ████████████   │  ← panel inferior (130px)
  │  ✓ CONFIRMADA / Analizando...                    │
  │  Historial: Hola · Gracias · Sí                  │
  │  [Q] Salir  [R] Reiniciar  [S] Screenshot        │
  └──────────────────────────────────────────────────┘

API pública:
  display.render(frame, result, fps, state, history)  ← método principal
  display.draw_landmarks(frame, result)               ← compatible con test scripts
"""

import cv2
import numpy as np
from collections import deque

from config.settings import (
    COLOR_LANDMARKS,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_CONF,
    COLOR_BG_PANEL,
    COLOR_ALERT,
    FONT,
)
from src.detection.hand_detector import HAND_CONNECTIONS


# ── Paleta de colores ────────────────────────────────────────────────────────
_C_GREEN   = (0, 210, 0)
_C_YELLOW  = (0, 210, 210)
_C_RED     = (0, 0, 200)
_C_WHITE   = (255, 255, 255)
_C_GRAY    = (140, 140, 140)
_C_DARK    = (20, 20, 20)
_C_ACCENT  = (0, 180, 255)   # Naranja para "Analizando"


def _conf_color(confidence: float) -> tuple[int, int, int]:
    """Devuelve un color BGR que va de rojo → amarillo → verde según la confianza."""
    if confidence < 0.60:
        return _C_RED
    if confidence < 0.75:
        return _C_YELLOW
    return _C_GREEN


def _alpha_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: tuple, alpha: float) -> None:
    """Dibuja un rectángulo semitransparente sobre el frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class Display:
    """
    Renderiza la interfaz completa del sistema LSP sobre frames de OpenCV.

    Uso principal:
        display = Display()
        display.render(frame, result, fps=35.2, state=pred_state, history=deque)

    Los métodos individuales (draw_landmarks, etc.) se mantienen para
    compatibilidad con los scripts de prueba de etapas anteriores.
    """

    TOP_BAR_H  = 45
    BOT_PANEL_H = 130
    LM_RADIUS  = 5
    CONN_W     = 2
    ALPHA_PANEL = 0.72

    # ── API principal ─────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        result,
        fps: float,
        state,
        history: deque,
    ) -> None:
        """
        Renderiza la interfaz completa en un solo llamado.

        Args:
            frame:   Frame BGR de la webcam (modificado in-place).
            result:  HandLandmarkerResult de MediaPipe (puede ser None).
            fps:     FPS calculado por CameraCapture.
            state:   PredictionState del SignPredictor.
            history: deque de strings con los últimos N signos confirmados.
        """
        self.draw_landmarks(frame, result)
        self._draw_top_bar(frame, fps, result, state)
        self._draw_bottom_panel(frame, state, history)
        self.draw_hand_label(frame, result)

    # ── Barra superior ────────────────────────────────────────────────────────

    def _draw_top_bar(self, frame: np.ndarray, fps: float, result, state) -> None:
        h, w = frame.shape[:2]
        _alpha_rect(frame, 0, 0, w, self.TOP_BAR_H, _C_DARK, 0.75)
        cv2.line(frame, (0, self.TOP_BAR_H), (w, self.TOP_BAR_H), (60, 60, 60), 1)

        # ── Indicador de estado (dot + texto) ────────────────────────────────
        hand_visible = bool(result and result.hand_landmarks)
        if state and state.is_stable:
            dot_color, status_text = _C_GREEN,  "SEÑA DETECTADA"
        elif hand_visible:
            dot_color, status_text = _C_ACCENT, "ANALIZANDO"
        else:
            dot_color, status_text = _C_GRAY,   "EN ESPERA"

        cv2.circle(frame, (16, self.TOP_BAR_H // 2), 7, dot_color, -1)
        cv2.putText(frame, status_text, (30, 29), FONT, 0.48, dot_color, 1)

        # ── Título centrado ───────────────────────────────────────────────────
        title = "Lengua de Senas Peruana (LSP)"
        (tw, _), _ = cv2.getTextSize(title, FONT, 0.52, 1)
        cv2.putText(frame, title, ((w - tw) // 2, 29), FONT, 0.52, _C_GRAY, 1)

        # ── FPS (derecha) ─────────────────────────────────────────────────────
        fps_text = f"FPS {fps:4.1f}"
        (fw, _), _ = cv2.getTextSize(fps_text, FONT, 0.52, 1)
        fps_color = _C_GREEN if fps >= 25 else (_C_YELLOW if fps >= 15 else _C_RED)
        cv2.putText(frame, fps_text, (w - fw - 12, 29), FONT, 0.52, fps_color, 1)

    # ── Panel inferior ────────────────────────────────────────────────────────

    def _draw_bottom_panel(self, frame: np.ndarray, state, history: deque) -> None:
        h, w = frame.shape[:2]
        panel_y = h - self.BOT_PANEL_H
        _alpha_rect(frame, 0, panel_y, w, h, _C_DARK, self.ALPHA_PANEL)

        if state and state.label:
            self._draw_sign_section(frame, state, panel_y, w)
        else:
            self._draw_idle_section(frame, state, panel_y, w)

        self._draw_history_row(frame, history, panel_y, w, h)
        self._draw_shortcuts(frame, w, h)

    def _draw_sign_section(self, frame, state, panel_y: int, w: int) -> None:
        """Muestra la seña actual con su estado y barra de confianza."""
        if state.is_stable:
            # Seña CONFIRMADA
            sep_color = _C_GREEN
            sign_color = _C_WHITE
            tag_text   = "CONFIRMADA"
            tag_color  = _C_GREEN
            sign_scale = 1.55
        else:
            # Seña TENTATIVA
            sep_color = _C_ACCENT
            sign_color = _C_YELLOW
            tag_text   = "Analizando..."
            tag_color  = _C_ACCENT
            sign_scale = 1.25

        # Línea separadora superior del panel
        cv2.line(frame, (0, panel_y), (w, panel_y), sep_color, 2)

        # Nombre de la seña (grande, con sombra para legibilidad)
        sign_text = state.display_label.upper()
        cv2.putText(frame, sign_text, (14, panel_y + 50),
                    FONT, sign_scale, _C_DARK, 4)
        cv2.putText(frame, sign_text, (14, panel_y + 50),
                    FONT, sign_scale, sign_color, 2)

        # Etiqueta de estado
        cv2.putText(frame, tag_text, (14, panel_y + 70),
                    FONT, 0.44, tag_color, 1)

        # Confianza (derecha) + barra
        conf_pct  = state.confidence * 100
        conf_text = f"{conf_pct:.1f}%"
        (cw, _), _ = cv2.getTextSize(conf_text, FONT, 0.72, 2)
        cv2.putText(frame, conf_text, (w - cw - 14, panel_y + 52),
                    FONT, 0.72, _conf_color(state.confidence), 2)

        # Barra de confianza
        bx, by = 14, panel_y + 78
        bw     = w - 28
        filled = int(bw * state.confidence)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 7), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, by), (bx + filled, by + 7),
                      _conf_color(state.confidence), -1)

    def _draw_idle_section(self, frame, state, panel_y: int, w: int) -> None:
        """Muestra mensaje de espera cuando no hay seña activa."""
        cv2.line(frame, (0, panel_y), (w, panel_y), (60, 60, 60), 1)
        idle_text = "Muestra tu mano a la camara"
        (tw, _), _ = cv2.getTextSize(idle_text, FONT, 0.6, 1)
        cv2.putText(frame, idle_text, ((w - tw) // 2, panel_y + 52),
                    FONT, 0.6, _C_GRAY, 1)

    def _draw_history_row(self, frame, history: deque, panel_y: int, w: int, h: int) -> None:
        """
        Fila de historial: muestra los últimos N signos confirmados.
        Las entradas más recientes aparecen más brillantes.
        """
        if not history:
            return

        items    = list(history)     # Más antiguo primero
        n        = len(items)
        base_y   = h - 32

        cv2.putText(frame, "Historial:", (14, base_y), FONT, 0.40, _C_GRAY, 1)

        x = 96
        for i, label in enumerate(items):
            # Más reciente = más brillante (último item)
            brightness = int(80 + 175 * (i / max(n - 1, 1)))
            color = (brightness, brightness, brightness)

            separator = "  " if i == 0 else " · "
            sep_w = 0
            if i > 0:
                (sep_w, _), _ = cv2.getTextSize(" · ", FONT, 0.40, 1)
                cv2.putText(frame, " · ", (x, base_y), FONT, 0.40, (60, 60, 60), 1)
            x += sep_w

            cv2.putText(frame, label, (x, base_y), FONT, 0.40, color, 1)
            (lw, _), _ = cv2.getTextSize(label, FONT, 0.40, 1)
            x += lw + 2

    def _draw_shortcuts(self, frame: np.ndarray, w: int, h: int) -> None:
        """Fila de atajos de teclado en la parte inferior del panel."""
        shortcuts = "[Q] Salir    [R] Reiniciar    [S] Screenshot"
        (sw, _), _ = cv2.getTextSize(shortcuts, FONT, 0.36, 1)
        cv2.putText(frame, shortcuts, ((w - sw) // 2, h - 10),
                    FONT, 0.36, (70, 70, 70), 1)

    # ── Landmarks ────────────────────────────────────────────────────────────

    def draw_landmarks(self, frame: np.ndarray, result) -> None:
        """Dibuja los 21 landmarks y el esqueleto de la mano."""
        if not result or not result.hand_landmarks:
            return

        h, w = frame.shape[:2]
        for hand_landmarks in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, pts[s], pts[e], COLOR_LANDMARKS, self.CONN_W)

            for px, py in pts:
                cv2.circle(frame, (px, py), self.LM_RADIUS, COLOR_LANDMARKS, -1)
                cv2.circle(frame, (px, py), 2, _C_DARK, -1)

    def draw_hand_label(self, frame: np.ndarray, result) -> None:
        """Muestra etiqueta izquierda/derecha cerca de la muñeca."""
        if not result or not result.handedness:
            return

        h, w = frame.shape[:2]
        for i, handedness in enumerate(result.handedness):
            if result.hand_landmarks and i < len(result.hand_landmarks):
                wrist = result.hand_landmarks[i][0]
                px    = int(wrist.x * w)
                py    = max(int(wrist.y * h) - 15, self.TOP_BAR_H + 15)
                label = handedness[0].category_name
                cv2.putText(frame, label, (px, py), FONT, 0.45, _C_ACCENT, 1)

    # ── Métodos de compatibilidad (usados en scripts de etapas anteriores) ───

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 2)
        cv2.putText(frame, text, (w - tw - 12, th + 12), FONT, 0.55, _C_YELLOW, 2)

    def draw_info_panel(
        self,
        frame: np.ndarray,
        sign: str = "",
        confidence: float = 0.0,
        status: str = "Buscando mano...",
        is_stable: bool = False,
    ) -> None:
        """Compatibilidad con scripts de etapas anteriores."""
        from collections import deque
        from src.recognition.predictor import PredictionState
        state   = PredictionState(None, confidence, is_stable) if not sign else \
                  PredictionState(sign.lower(), confidence, is_stable)
        history = deque()
        self._draw_bottom_panel(frame, state, history)

    def draw_no_hand_warning(self, frame: np.ndarray) -> None:
        """Compatibilidad con scripts de etapas anteriores."""
        pass   # La barra superior ya indica el estado
