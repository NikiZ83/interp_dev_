### Краткий отчет по `metrics.py`

**Роль скрипта в проекте**:  
`metrics.py` — центральный модуль для **оценки качества регрессии** по скорости речи (CPS). Он вычисляет метрики (MSE, MAE, R2), сохраняет предсказания, строит визуализации (t-SNE, графики метрик, scatter, гистограммы ошибок). Используется на всех этапах:  
- Задание 2: оценка MLP на эмбеддингах.  
- Задание 3: послойный probing (по активациям слоёв).  

---

### Формат входных/выходных данных

| Функция | Вход | Выход | Назначение |
|--------|------|-------|-----------|
| `evaluate_emb_model` | `model`, `test_loader`, `device`, `dataset`, `layer`, `save_path` | `dict` с MSE/MAE/R2 + CSV с `filename`, `true_cps`, `predicted_{layer}` | Оценка финальной модели на эмбеддингах |
| `evaluate_probing` | `layer`, `y_pred`, `y_true` | `(layer, dict)` с метриками | Оценка одного слоя при probing |
| `save_to_csv` | `chunk_rows` (список словарей) | CSV-файл (дополняется) | Сохранение предсказаний по слоям в один файл |
| `save_metrics` | `metrics_list` | `.txt` файл (по слоям) | Сохранение метрик для последующего чтения |
| `read_metrics` | `.txt` файл | `List[(layer, dict)]` | Чтение сохранённых метрик |
| `plot_metrics` | `metrics_list` | `.png` (3 графика: MSE, MAE, R2 по слоям) | Визуализация динамики по слоям |
| `plot_tsne` | `embeddings`, `true_values` | `.png` | t-SNE эмбеддингов с цветом по CPS |
| `plot_scatter` / `plot_error_histogram` | `true`, `pred` | `.png` | Анализ ошибок предсказания |

---

### Ключевые особенности

- **Поддержка послойного анализа**: метрики сохраняются по `layer`, CSV дополняется по `filename`.
- **Гибкое сохранение**:  
  - CSV: `filename | true_cps | predicted_layerX`  
  - TXT: `layer\nmse: ...\nmae: ...\n`
- **Визуализация**:  
  - `plot_metrics` — **основной график для Задания 3** (линейные графики по слоям).  
  - `plot_tsne` — для Задания 2.  
  - `scatter` и `histogram` — для анализа ошибок.

---

### Использование в проекте

```python
# Пример из Задания 2
metrics = evaluate_emb_model(model, test_loader, device, dataset, layer="final", save_path="results/predictions.csv")
plot_tsne(embeddings, cps_values, "results/tsne.png")

# Пример из Задания 3 (probing)
layer_metrics = []
for layer in layers:
    y_pred = probe_model(activations[layer])
    layer, metrics = evaluate_probing(layer, y_pred, y_true)
    layer_metrics.append((layer, metrics))

save_metrics(layer_metrics, "results/layer_metrics.txt")
plot_metrics(layer_metrics, "results/metrics_by_layer.png")
```

---

**Вывод**:  
`metrics.py` — готовый и **полностью подходит** для задач регрессии по скорости речи.  
Нужно лишь:  
1. Передавать `layer` при probing.  
2. Сохранять CSV/TXT/PNG в `results/`.  
3. Использовать `plot_metrics` для графика по слоям (в отчёт и статью).