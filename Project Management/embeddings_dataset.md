**Отчет по `embeddings_dataset.py`**

### Роль файла в проекте
`embeddings_dataset.py` — **основной класс датасета** для задачи регрессии скорости речи (CPS).  
Он:
- Загружает **эмбеддинги** из `.npy` или **ChromaDB**.
- Сопоставляет их с **значениями CPS** из CSV.
- Формирует **тензоры** для подачи в PyTorch-модель (MLP).
- Наследуется от `BaseDataset` → совместим с `DataLoader`.

Используется в:
- `train_model.py` — для обучения MLP.
- `acts_probing.py` — для анализа активаций (если расширить).
- Любых скриптах, где нужен доступ к эмбеддингам + меткам.

---

### Формат входных данных

| Параметр | Описание | Пример |
|--------|--------|--------|
| `source_path` | Папка с эмбеддингами | `data/processed/embeddings/` |
| `split` | Раздел датасета | `'train'`, `'dev'`, `'test'` |
| `source_type` | Формат хранения | `'npy'` или `'chromadb'` |
| `cps_csv` | CSV с CPS | `data/processed/train_cps.csv` |
| `collection_name` | Имя коллекции в ChromaDB | `'speed_embeddings'` |

---

### Формат выходных данных (`__getitem__`)

```python
return embedding_tensor, cps_tensor
```
- `embedding_tensor` — `torch.float32`, форма: `(embedding_dim,)`
- `cps_tensor` — `torch.float32`, скаляр (1 значение CPS)

---

### Внутренняя структура данных

После `prepare_data()` создаются атрибуты:

| Атрибут | Тип | Описание |
|--------|-----|--------|
| `self.audio_data` | `torch.Tensor` | `(N, D)` — N сэмплов, D размер эмбеддинга |
| `self.labels` | `torch.Tensor` | `(N,)` — CPS значения |
| `self.filenames` | `list[str]` | Пути к аудио (для отладки/визуализации) |

---

### Как работает загрузка

#### 1. `.npy` (рекомендуемый)
- Файл: `source_path/numpy_embs.npy`
- Формат: словарь `{split: [ {embedding: [...], file_path: "..."}, ... ]}`
- CPS: сопоставление по `audio_path` из CSV

#### 2. `chromadb`
- Клиент: `PersistentClient(path=source_path)`
- Коллекция: `speed_embeddings`
- Фильтр: `where={"split": "train"}`
- Метаданные: содержат `file_path`

---

### Ключевые особенности
- **Гибкость**: поддержка двух бэкендов.
- **Безопасность**: проверки на существование файлов.
- **Совместимость**: возвращает тензоры → сразу в `DataLoader`.
- **Отладка**: хранит `filenames` → можно сопоставить предсказания.

---

### Рекомендации по использованию
```python
dataset = RegressionEmbeddingsDataset(
    source_path="data/processed/embeddings",
    split="train",
    source_type="npy",
    cps_csv="data/processed/train_cps.csv"
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

**Вывод**:  
Файл — **ядро пайплайна регрессии**.  
Гарантирует, что **эмбеддинги и CPS синхронизированы** и готовы к обучению.  
Формат: `(embedding, cps)` → идеально для `nn.MSELoss`.