# LLM Text Generation Model

A deep-learning text generation project built with TensorFlow/Keras. The model uses a stacked LSTM (Long Short-Term Memory) network trained on a fake/real news corpus to predict and generate coherent text sequences.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Training](#training)
- [Text Generation](#text-generation)
- [Results & Visualization](#results--visualization)
- [Saved Artifacts](#saved-artifacts)

---

## Overview

This project implements a **word-level language model** using a two-layer LSTM network. Given a seed phrase, the model generates a continuation by iteratively predicting the next most likely word. Training is performed on a large news-text corpus, keeping the 5,000 most frequent words as the vocabulary.

---

## Project Structure

```
LLM_model/
├── main.ipynb            # End-to-end Jupyter notebook (data prep → training → generation)
├── learn_to_speak.py     # Standalone Python script version
├── fake_or_real_news.csv # Training dataset (news articles)
├── diabetes.csv          # Additional dataset (not used in text generation)
├── joined_text.txt       # Pre-joined corpus text
├── best_model.keras      # Best model checkpoint saved during training
├── best_model.h5         # Best model checkpoint (HDF5 format)
├── text_gen_model.keras  # Final saved model (Keras format)
├── text_gen_model.h5     # Final saved model (HDF5 format)
└── history.p             # Pickled training history
```

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- NLTK
- SciPy
- Matplotlib

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Kunalrpawar/LLM_model.git
cd LLM_model

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow numpy pandas nltk scipy matplotlib
```

---

## Dataset

The model is trained on `fake_or_real_news.csv`, a dataset of news article bodies labelled as *fake* or *real*. Only the raw article text is used for training; labels are ignored.

- The first **1,000,000 characters** of the joined corpus are used to keep memory usage manageable.
- Text is tokenized with NLTK's `RegexpTokenizer` (word characters only, lower-cased).
- The **top 5,000** most frequent tokens form the vocabulary.

---

## How It Works

| Step | Description |
|------|-------------|
| 1. Load & join | All article texts are concatenated into one large string. |
| 2. Tokenize | Text is split into word tokens (punctuation removed). |
| 3. Vocabulary | The 5,000 most common words are kept; each is assigned an index. |
| 4. Sequences | Sliding windows of `n_words = 10` tokens create (input, next-word) pairs. |
| 5. One-hot encode | Each input window is encoded into a 3-D binary tensor `(samples, n_words, vocab_size)`. |
| 6. Train | A stacked LSTM model is trained with categorical cross-entropy loss. |
| 7. Generate | The trained model samples from the top-k predictions at each step. |

---

## Training

### Model Architecture

```
LSTM(128, return_sequences=True)  →  input shape: (n_words, vocab_size)
LSTM(128)
Dense(vocab_size)
Softmax activation
```

### Compile & Callbacks

```python
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True),
]
```

### Run Training (Notebook)

Open `main.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook main.ipynb
```

Or run the standalone script:

```bash
python learn_to_speak.py
```

---

## Text Generation

After training, use the `generate_text` helper to produce text from any seed phrase:

```python
# Generate 10 new words with creativity=5 (top-5 sampling)
output = generate_text("I will have to look into this thing because I", n_words=10, creativity=5)
print(output)
```

The `creativity` parameter controls how many top predictions are sampled from at each step — higher values produce more varied (but potentially less coherent) text.

---

## Results & Visualization

Training loss and accuracy curves are plotted automatically at the end of `main.ipynb`:

```python
plot_learning_curve(history)
```

The chart shows **training vs. validation loss** and **accuracy** over epochs so you can detect over/under-fitting at a glance.

---

## Saved Artifacts

| File | Description |
|------|-------------|
| `best_model.keras` / `best_model.h5` | Lowest-loss checkpoint saved by `ModelCheckpoint` |
| `text_gen_model.keras` / `text_gen_model.h5` | Final model after all training epochs |
| `history.p` | Pickled `history` dict for offline analysis |

To load a saved model and generate text without retraining:

```python
from tensorflow.keras.models import load_model
import pickle

model = load_model("best_model.keras")
with open("history.p", "rb") as f:
    history = pickle.load(f)
```
