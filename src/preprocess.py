import argparse
import logging
from pathlib import Path

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_cps(audio_path: Path, text: str) -> float:
    """
    Рассчитывает скорость речи (CPS) для аудиофайла.
    CPS - characters per second (буквы в секунду, учитывая только буквы и цифры).

    :param audio_path: Путь к аудиофайлу (.flac)
    :param text: Транскрипция аудио
    :return: Скорость речи в буквах в секунду (CPS)
    """
    try:
        audio = AudioSegment.from_file(audio_path, format="flac")
        duration_seconds = len(audio) / 1000  # длительность в секундах
        # Учитываем только буквы и цифры в тексте
        char_count = len([char for char in text.strip() if char.isalnum()])
        return char_count / duration_seconds if duration_seconds > 0 else 0
    except Exception as err:
        logger.error(f"Ошибка при обработке {audio_path}: {err}")
        return 0

def process_subset(base_path: Path, output_csv: Path):
    """
    Обрабатывает один subset LibriSpeech (train/dev/test),
    создает CSV с CPS (буквы в секунду).

    :param base_path: Путь к папке subset
    :param output_csv: Путь для сохранения CSV
    :return:
    """
    data = []

    if not base_path.exists():
        logger.error(f"Папка {base_path} не найдена!")
        return

    speaker_paths = [p for p in base_path.iterdir() if p.is_dir()]
    for speaker_path in tqdm(speaker_paths, desc=f"Обработка спикеров в {base_path.name}"):
        for chapter_path in speaker_path.iterdir():
            if not chapter_path.is_dir():
                continue

            trans_file = chapter_path / f"{chapter_path.parent.name}-{chapter_path.name}.trans.txt"
            if not trans_file.exists():
                logger.warning(f"Транскрипция не найдена: {trans_file}")
                continue

            try:
                with trans_file.open('r', encoding="utf-8") as file:
                    lines = file.readlines()
                    for line in tqdm(lines, desc=f"Обработка транскрипций в {chapter_path.name}", leave=False):
                        try:
                            audio_id, text = line.strip().split(' ', 1)
                            audio_file = chapter_path / f"{audio_id}.flac"
                            if audio_file.exists():
                                cps = get_cps(audio_file, text)
                                data.append({
                                    'audio_path': str(audio_file),
                                    'text': text,
                                    'cps': cps,  # Изменено с 'wpm' на 'cps'
                                    'speaker_id': speaker_path.name,
                                    'chapter_id': chapter_path.name,
                                    'audio_id': audio_id
                                })
                            else:
                                logger.warning(f"Аудиофайл не найден: {audio_file}")
                        except ValueError:
                            logger.error(f"Ошибка формата в строке: {line.strip()}")
            except Exception as err:
                logger.error(f"Ошибка при чтении {trans_file}: {err}")

    if data:
        df = pd.DataFrame(data)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Сохранено {len(df)} записей в {output_csv}")
    else:
        logger.warning(f"Нет данных для сохранения в {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess LibriSpeech for speech rate (CPS).")
    parser.add_argument('--data_dir', type=Path, default=Path('data/raw'), help='Path to raw LibriSpeech data')
    parser.add_argument('--output_dir', type=Path, default=Path('data/processed'),
                        help='Path to save processed CSV files')
    args = parser.parse_args()

    subsets = [
        ('dev-clean', 'dev_cps.csv'),  # Изменено с 'dev_wpm.csv' на 'dev_cps.csv'
        ('test-clean', 'test_cps.csv'),  # Изменено с 'test_wpm.csv' на 'test_cps.csv'
        ('train-clean-100', 'train_cps.csv'),  # Изменено с 'train_wpm.csv' на 'train_cps.csv'
    ]

    for subset, output_file in subsets:
        base_path = args.data_dir / subset / "LibriSpeech" / subset
        output_csv = args.output_dir / output_file
        logger.info(f"Обработка {subset}...")
        process_subset(base_path, output_csv)

if __name__ == "__main__":
    main()