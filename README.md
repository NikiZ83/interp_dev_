PP_past/
├── data/
│   ├── raw/
│   │   ├── train-clean-100/
│   │   │   ├── speaker_id/
│   │   │   │   ├── chapter_id/
│   │   │   │   │   ├── audio_file.flac
│   │   │   │   │   ├── chapter_id.trans.txt
│   │   ├── dev-clean/
│   │   │   ├── speaker_id/
│   │   │   │   ├── chapter_id/
│   │   │   │   │   ├── audio_file.flac
│   │   │   │   │   ├── chapter_id.trans.txt
│   │   ├── test-clean/
│   │   │   ├── speaker_id/
│   │   │   │   ├── chapter_id/
│   │   │   │   │   ├── audio_file.flac
│   │   │   │   │   ├── chapter_id.trans.txt
│   ├── processed/
│   │   ├── train_wpm.csv
│   │   ├── dev_wpm.csv
│   │   ├── test_wpm.csv
│   │   ├── train_embeddings.pkl
│   │   ├── dev_embeddings.pkl
│   │   ├── test_embeddings.pkl
├── src/
│   ├── preprocess.py
│   ├── extract_embeddings.py
│   ├── train_mlp.py
│   ├── utils.py
├── models/
│   ├── mlp_model.h5
│   ├── mlp_best_weights.h5
├── results/
│   ├── training_history.csv
│   ├── test_predictions.csv
│   ├── plots/
│   │   ├── loss_plot.png (опционально)
├── report/
│   ├── report.pdf (или report.docx)
│   ├── slides.pptx (если требуется)
├── README.md