### Краткий отчет по `cca_dev.py`

**Роль скрипта в проекте**  
`cca_dev.py` — модуль для **послойного анализа линейной зависимости** между активациями слоёв модели WeSpeaker и целевым признаком **скорость речи (cps — characters per second)** с помощью **Canonical Correlation Analysis (CCA)**.  
Он **дополняет Задание 3** (послойный probing):  
- Оценивает, насколько активации каждого слоя **коррелируют** с cps (альтернатива MLP-regression).  
- Сохраняет CCA-коэффициенты корреляции по слоям → CSV + график (bar plot).  
- Используется для **сравнения моделей** (SimAMResNet34 vb и др.) на `dev-clean` (5000 сэмплов).  
Не обязателен для базового PR, но усиливает анализ (можно добавить в статью как "линейный probing").

---

**Формат входных данных**

| Источник | Путь | Формат |
|--------|------|--------|
| **CSV с cps** | `data/processed/dev_cps.csv` | Столбцы: `audio_path` (str), `cps` (float) |
| **Модель** | `models/voxblink2_samresnet34_ft/...` | WeSpeaker (SimAMResNet34 vb) |
| **Аудио** | По `audio_path` из CSV | `.flac` → Fbank (80 мел-фильтров, T ≤ 400 фреймов) |

- Ограничение: `n_samples=5000`, `max_frames=400` → фиксированный паддинг.

---

**Поток данных**

1. **Чтение CSV** → `audio_paths` (list[str]), `cps` (np.array[float], shape: [N]).
2. **Для каждого слоя** (`layers = get_layers(model)`):
   - `extract_activations_for_layer()` → активации (N, D):
     - Fbank → `extract_features()` → (T, 80) → паддинг до (400, 80).
     - `GetActivations` → активация слоя → mean-pool (если 4D) → flatten → (1, D).
     - Конкатенация → `X` (N, D).
   - `Y = cps[:, None]` (N, 1).
   - `compute_cca(X, Y)` → корреляция (1 компонента).
3. **Сохранение**:
   - `results_1/cca/cca_dev_cps.csv` → `layer | cca_corr`.
   - `cca_dev_cps.png` → bar plot (layers vs corr).

---

**Формат выходных данных**

- **CSV**:  
  ```
  layer,cca_corr
  first relu,0.412
  layer1 relu 1,0.538
  ...
  ```
- **PNG**: бары по слоям, ylabel="CCA correlation".

---

**Ключевые особенности**
- **Mean-pooling** для conv-слоёв → вектор (1, C).
- **Паддинг** до 400 фреймов → фиксированный input.
- **CCA**: 1 компонента, `max_iter=500`, обработка ошибок → 0.0.
- **Ограничения**: только линейная корреляция, `n_samples ≥ 2`.

---

**Использование**
```bash
poetry run python src/cca_dev.py
```
- Менять `model_name`, `n_samples`, `csv_path` для других моделей/сплитов.

**Вывод**:  
Скрипт готов к запуску после `preprocess.py` и загрузки модели.  
Дополняет MLP-probing: показывает **линейную предсказуемость** cps по слоям.  
Для PR: добавить в `speech_speed/cca/`, запустить на всех моделях, вставить график в статью (раздел "Posloynoy analiz").