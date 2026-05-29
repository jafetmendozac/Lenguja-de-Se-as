"""
Módulo de síntesis de voz (Text-to-Speech).
Implementación completa en ETAPA 7.

Stub funcional para ETAPA 6: registra en consola sin reproducir audio.
"""


class Speaker:
    """Stub de TTS. Se reemplaza en ETAPA 7 con pyttsx3."""

    def __init__(self) -> None:
        self._last_spoken: str = ""

    def speak(self, text: str) -> None:
        """Habla el texto dado. Por ahora solo imprime en consola."""
        if text != self._last_spoken:
            print(f"[TTS-stub] → {text}")
            self._last_spoken = text

    def stop(self) -> None:
        pass
