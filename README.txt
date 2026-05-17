NEXT-WORD PREDICTION WITH LSTM AND ATTENTION
============================================

Production-quality implementation of a word-level next-word predictor trained on "The Adventures of Sherlock Holmes" by Arthur Conan Doyle.

The project uses TensorFlow/Keras, a Bidirectional LSTM, and Bahdanau-style additive attention to predict P(next_word | context) from fixed-length token windows.

FEATURES
--------
- Automatic Project Gutenberg corpus download and validation
- Gutenberg boilerplate removal, text cleaning, tokenization, and sequence generation
- Sparse-label training pipeline for memory-efficient modeling
- Embedding -> Bidirectional LSTM -> Attention -> Dense -> Dropout -> Softmax model
- Train/validation/test split with loss, top-1 accuracy, top-5 accuracy, and perplexity
- Epoch-wise JSON/CSV metrics and training plots
- Text generation with temperature, top-k, nucleus, greedy, and multinomial decoding
- Top-5 next-word predictions with probabilities
- Attention heatmap visualization
- Clean notebook and CLI entry points

PROJECT STRUCTURE
-----------------
.
├── ML Engineer Hiring Assignment.pdf
├── pyproject.toml
├── uv.lock
├── README.txt
├── REPORT.txt
├── TRAINING_HISTORY.txt
├── HOW_TO_RUN.txt
├── data/
│   ├── sherlock_holmes.txt
│   └── tokenizer.pkl
├── notebooks/
│   └── next_word_prediction.ipynb
├── outputs/
│   ├── best_model.keras
│   ├── final_model.keras
│   └── training_history.png
└── src/
    ├── config.py
    ├── data_loader.py
    ├── preprocessor.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── generate.py
    └── visualization.py

SETUP
-----
See HOW_TO_RUN.txt for detailed instructions on how to set up and run the project on Mac or any laptop.

TRAIN
-----
uv run python -m src.train

EVALUATE
--------
uv run python -m src.evaluate

GENERATE TEXT
-------------
uv run python -m src.generate

MODEL ARCHITECTURE
------------------
Input context tokens (sequence_length=15)
-> Embedding(vocab_size, 128, mask_zero=True)
-> Bidirectional LSTM(256, return_sequences=True)
-> Bahdanau-style additive attention
-> Dense(256, relu)
-> Dropout(0.3)
-> Dense(vocab_size, softmax)

NOTEBOOK
--------
uv run jupyter notebook notebooks/next_word_prediction.ipynb
