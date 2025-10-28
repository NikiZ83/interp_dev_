### Краткий отчёт по `extract_embeddings.py`

#### Роль скрипта в проекте
Скрипт **извлекает спикерские эмбеддинги** из аудиофайлов LibriSpeech с помощью предобученной модели **WeSpeaker (SimAMResNet34 vb)**.  
Он — **ключевой этап Задания 2**:  
- Берёт обработанные CSV-файлы (`train_cps.csv`, `test_cps.csv`, `dev_cps.csv`) из `data/processed`.  
- Извлекает векторные представления (эмбеддинги) для каждого аудио.  
- Присоединяет **лейблы** (в оригинале — `speaker_id`, но будет модифицирован под **скорость речи**).  
- Сохраняет эмбеддинги в формате `.npy` или ChromaDB.

---

#### Формат входных данных
- **CSV-файлы** в `data/processed/`:
  - Столбцы: `audio_path`, `speaker_id`, (позже добавим `speech_speed`)
  - Пример строки:  
    ```
    /data/raw/train-clean-100/ speaker1/utt1.wav, 1234, 12.5
    ```
- **Аудиофайлы**: WAV из LibriSpeech (`train-clean-100`, `dev-clean`, `test-clean`).

---

#### Формат выходных данных
1. **Список словарей** (в памяти):
   ```python
   [
       {
           'file_path': 'data/raw/.../utt1.wav',
           'embedding': np.array([0.12, -0.45, ...]),  # shape: (256,) или (192,) в зависимости от модели
           'label': 12.5  # будет — скорость речи (букв/сек)
       },
       ...
   ]
   ```
2. **Сохранение**:
   - `npy`: `numpy_embs.npy` — массив объектов (train + test + dev).
   - `chromadb`: коллекция `gender_embeddings` (будет переименована под `speech_speed`).

---

#### Что нужно модифицировать под **скорость речи**
| Функция | Что изменить |
|--------|-------------|
| `assign_labels_by_speaker_id` | Переименовать → `assign_speed_labels`<br>Читать `speech_speed` вместо `speaker_id` |
| `main()` | Убрать привязку к `speaker_id`, использовать `speech_speed` как `label` |
| Сохранение | Сохранять **только эмбеддинги + скорость** (без спикера, если не нужно) |

---

#### Вывод
`extract_embeddings.py` — **центральный пайплайн извлечения признаков**.  
После модификации он:
1. Берёт аудио + скорость из `data/processed/*.csv`
2. Выдаёт `embeddings + speech_speed`
3. Готовит данные для **MLP-регрессии** в `train_model.py`

**Следующий шаг**: модифицировать `assign_labels_by_speaker_id` под `speech_speed` и запустить на `train_cps.csv`.