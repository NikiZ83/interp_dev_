**Отчет по `run_evaluate.py`**

### Роль скрипта в проекте
Скрипт **оценивает обученную регрессионную модель** на тестовом наборе LibriSpeech.  
Он:
- Загружает эмбеддинги и метки скорости речи (cps — букв/сек).
- Прогоняет модель → получает предсказания.
- Считает метрики (MSE, MAE).
- Сохраняет:
  - метрики в `results_1/metrics.txt`
  - предсказания в `predictions.csv`
  - визуализации (scatter, error hist, bar, t-SNE, metrics plot).

Это **финальный этап Задания 2** — проверка качества MLP на эмбеддингах.

---

### Формат входных данных

| Источник | Путь | Формат |
|--------|------|--------|
| **Модель** | `models/regression_model.pt` | `torch.save(model)` |
| **Тестовая разметка** | `data/processed/test_cps.csv` | CSV: `id, cps` (cps = letters / duration) |
| **Эмбеддинги** | `embeddings_npy/` | `.npy` файлы по ID (или один большой) |
| **Dataset** | `RegressionEmbeddingsDataset` | Возвращает `(embedding: torch.float32 [D], cps: torch.float32)` |

- `D` — размерность эмбеддинга (зависит от модели WeSpeaker, напр. 256).
- Батч: `embeddings_batch.shape = [B, D]`, `cps_batch.shape = [B]`.

---

### Поток данных в скрипте

1. **Загрузка датасета** → `DataLoader` (batch_size=32)
2. **Загрузка модели** → `model.to(device).eval()`
3. **evaluate_emb_model()** → прогон по тесту, возврат dict метрик
4. **Сбор предсказаний + эмбеддингов** (в цикле `with torch.no_grad()`):
   - `true_values` ← `cps_batch.numpy()`
   - `predicted_values` ← `model output`
   - `all_embeddings` ← все эмбеддинги в `np.array`
5. **t-SNE** на `all_embeddings`, цвет по `true_values`
6. **Визуализации**:
   - Scatter: true vs pred
   - Error hist: (pred - true)
   - Bar: MSE/MAE
   - Metrics plot

---

### Выходные файлы (`results_1/`)

| Файл | Содержание |
|------|------------|
| `metrics.txt` | MSE, MAE |
| `predictions.csv` | id, true_cps, pred_cps |
| `scatter_plot.png` | true vs pred |
| `error_histogram.png` | распределение ошибок |
| `bar_metrics.png` | столбцы MSE/MAE |
| `metrics_plot.png` | (если несколько моделей) |
| `tsne_plot.png` | 2D проекция эмбеддингов |

---

### Ключевые зависимости
- `RegressionEmbeddingsDataset` — читает `.npy` + CSV
- `evaluate_emb_model` — считает MSE/MAE, сохраняет предсказания
- `plot_tsne` — использует `sklearn.manifold.TSNE`

---

### Вывод
`run_evaluate.py` — **универсальный evaluator** для регрессии скорости речи.  
Работает с любыми `.npy` эмбеддингами и `test_cps.csv`.  
Готов к использованию в **Задании 2 и 3** (послойный анализ — просто поменять `source_path` на папку активаций).