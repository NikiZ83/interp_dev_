import os
import pandas as pd
import logging
from pydub import AudioSegment
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def get_wpm(audio_path: str, text: str) -> float:
    """
    Рассчитывает скорость речи (WPM) для аудиофайла.
    WPM - word per minute

    :param audio_path: Путь к аудиофайлу (.flac)
    :param text: Транскрипция аудио
    :return: Скорость речи в словах в минуту (WPM)
    """
    try:
        audio = AudioSegment.from_file(audio_path, format="flac")
        duration_minutes = len(audio) / 1000 / 60  # в миллисекундах
        word_count = len([word for word in text.strip().split() if word])
        return word_count / duration_minutes if duration_minutes > 0 else 0
    except Exception as err:
        logger.error(f"Ошибка при обработке {audio_path}: {err}")
        return 0