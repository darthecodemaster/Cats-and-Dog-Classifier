# 🐱🐶 Cat vs Dog Image Classifier — ML Project Journal

> **freeCodeCamp Machine Learning with Python Certification**
> Convolutional Neural Network built with TensorFlow 2.0 & Keras

---

## 📌 Project Overview

This project builds a **Convolutional Neural Network (CNN)** that classifies images as either a **cat** or a **dog** with at least **63% accuracy** (bonus goal: 70%+).

| Item | Detail |
|------|--------|
| **Framework** | TensorFlow 2.0 + Keras |
| **Environment** | Google Colaboratory |
| **Task** | Binary Image Classification |
| **Dataset** | 2,000 train / 1,000 validation / 50 test images |
| **Target Accuracy** | ≥ 63% (70%+ for extra credit) |

---

## 📂 Repository Structure

```
cats-dogs-classifier/
│
├── cats_and_dogs_classifier.py    # Full commented source code
├── cats_and_dogs_classifier.ipynb # Google Colab notebook (submit this)
├── README.md                      # This journal file
└── journal/
    └── JOURNAL.md                 # Detailed learning journal & notes
```

---

## 🚀 How to Run

### Option A — Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `cats_and_dogs_classifier.ipynb` **or** paste the code cells
3. Run Cell 2 — it will auto-download the dataset
4. Run all cells in order (Cell 1 → Cell 11)
5. Enable link sharing → Submit your Colab link

### Option B — Local
```bash
pip install tensorflow matplotlib numpy
python cats_and_dogs_classifier.py
```

---

## 🧠 Model Architecture

```
Input (150x150x3)
    │
    ▼
Conv2D(32, 3x3) → ReLU → MaxPool(2x2)
    │
Conv2D(64, 3x3) → ReLU → MaxPool(2x2)
    │
Conv2D(128, 3x3) → ReLU → MaxPool(2x2)
    │
Conv2D(128, 3x3) → ReLU → MaxPool(2x2)
    │
Flatten → Dense(512, ReLU) → Dropout(0.5)
    │
Dense(1, Sigmoid) → Output (0=cat, 1=dog)
```

**Optimizer:** Adam | **Loss:** Binary Crossentropy

---

## 🔄 Data Augmentation (Cell 5)

To prevent overfitting on the small dataset, the following random transformations were applied:

| Augmentation | Value |
|---|---|
| Rotation | ±40° |
| Width Shift | 20% |
| Height Shift | 20% |
| Shear | 20% |
| Zoom | 20% |
| Horizontal Flip | Yes |
| Fill Mode | Nearest |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~85%+ |
| Validation Accuracy | ~70%+ |
| Pass Threshold | 63% ✅ |
| Bonus Threshold | 70% 🌟 |

*(Update this table after your training run)*

---

## 💡 Key Learnings

- `ImageDataGenerator` rescales pixel values from [0–255] → [0–1] for better gradient flow
- `shuffle=False` in `test_data_gen` ensures predictions match expected image order
- Data augmentation creates synthetic variety, reducing overfitting
- Dropout (0.5) randomly disables neurons during training to improve generalization
- `binary_crossentropy` is the correct loss for two-class problems with sigmoid output

---

## 🔗 Links

- [Project Notebook (Colab)]() ← *paste your link here*
- [freeCodeCamp ML Certification](https://www.freecodecamp.org/learn/machine-learning-with-python/)
- [TensorFlow Keras Docs](https://www.tensorflow.org/api_docs/python/tf/keras)
