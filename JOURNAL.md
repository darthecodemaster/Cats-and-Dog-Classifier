# 📓 Project Learning Journal
## Cat vs Dog CNN Classifier — freeCodeCamp ML Certification

---

## 📅 Journal Entry 1 — Project Setup

### What I Did
- Read the project requirements from freeCodeCamp
- Set up Google Colaboratory environment
- Understood the dataset structure:
  - `train/` → 2,000 labeled images (cats + dogs)
  - `validation/` → 1,000 labeled images
  - `test/` → 50 unlabeled images (no subdirectories)

### Key Concepts Learned
**ImageDataGenerator** — Keras utility that reads images from disk in batches.
- `rescale=1./255` normalizes pixel values from [0, 255] to [0, 1]
- This helps gradient descent converge faster and more stably

**flow_from_directory** — Automatically labels images based on folder names.
- `cats/` → class 0, `dogs/` → class 1 (alphabetical)
- `class_mode='binary'` returns 0 or 1 labels
- `class_mode=None` for test data (no labels exist)
- `shuffle=False` for test data — critical so predictions stay in the correct order!

### Challenges
- The `test/` directory has no subdirectories, so I need to point
  `flow_from_directory` at the parent folder containing `test/`

---

## 📅 Journal Entry 2 — Data Exploration

### What I Did
- Ran Cell 4 to visualize 5 random training images
- Verified the dataset loaded correctly:
  - Found 2000 images belonging to 2 classes ✅
  - Found 1000 images belonging to 2 classes ✅
  - Found 50 images belonging to 1 class ✅

### Observations
- Images vary widely in size, angle, lighting, and background
- Some images have multiple animals or partial views
- This variation is why augmentation will be important

---

## 📅 Journal Entry 3 — Data Augmentation

### The Problem: Overfitting
With only 2,000 training images, the model can memorize the training set
instead of learning general features. This causes:
- High training accuracy (~99%)
- Low validation accuracy (~55–60%)
- The gap between them = overfitting

### The Solution: Data Augmentation
I added random transformations to artificially expand the training dataset.
Each epoch the model sees slightly different versions of the same images.

```python
ImageDataGenerator(
    rescale            = 1./255,
    rotation_range     = 40,      # rotate up to 40 degrees
    width_shift_range  = 0.2,     # shift left/right by 20%
    height_shift_range = 0.2,     # shift up/down by 20%
    shear_range        = 0.2,     # shear transformation
    zoom_range         = 0.2,     # zoom in/out by 20%
    horizontal_flip    = True,    # mirror image horizontally
    fill_mode          = 'nearest'# fill empty pixels with nearest neighbor
)
```

### Why These Values?
- 40° rotation is aggressive but animals appear at many angles in real photos
- Horizontal flip makes sense — a cat facing left = same as facing right
- No vertical flip (upside-down dogs are uncommon in real-world use)

---

## 📅 Journal Entry 4 — Building the CNN

### Why Convolutional Neural Networks?
Regular (Dense) neural networks treat each pixel independently — 
a 150×150 image has 67,500 input neurons before any hidden layers.
CNNs instead learn **spatial patterns** (edges, textures, shapes) 
through small sliding filters, which:
1. Dramatically reduces parameters
2. Learns location-invariant features (a cat ear is a cat ear anywhere)

### Layer-by-Layer Breakdown

| Layer | Output Shape | Purpose |
|-------|-------------|---------|
| Input | 150×150×3 | RGB image |
| Conv2D(32) | 148×148×32 | Detect low-level features (edges, colors) |
| MaxPool | 74×74×32 | Downsample, keep strongest features |
| Conv2D(64) | 72×72×64 | Detect mid-level features (textures) |
| MaxPool | 36×36×64 | Downsample |
| Conv2D(128) | 34×34×128 | Detect higher-level features (shapes) |
| MaxPool | 17×17×128 | Downsample |
| Conv2D(128) | 15×15×128 | Detect complex features (eyes, ears) |
| MaxPool | 7×7×128 | Downsample |
| Flatten | 6272 | Convert 3D → 1D |
| Dense(512) | 512 | Learn combinations of features |
| Dropout(0.5) | 512 | Randomly zero 50% of neurons = regularization |
| Dense(1, sigmoid) | 1 | Output probability (0=cat, 1=dog) |

### Why Sigmoid at the Output?
- Squashes output to [0, 1] → interpretable as probability
- Values > 0.5 = dog, values < 0.5 = cat
- Binary Cross-Entropy loss pairs naturally with sigmoid

### Why Dropout?
Dropout randomly sets 50% of neurons to 0 during each training step.
- Forces the network not to rely on any single neuron
- Acts as an ensemble of many smaller networks
- Significantly reduces overfitting

---

## 📅 Journal Entry 5 — Training & Evaluation

### Training Configuration
```python
history = model.fit(
    x                = train_data_gen,
    steps_per_epoch  = train_data_gen.samples // BATCH_SIZE,  # 2000 // 32 = 62
    epochs           = 15,
    validation_data  = val_data_gen,
    validation_steps = val_data_gen.samples // BATCH_SIZE     # 1000 // 32 = 31
)
```

### Reading the Accuracy/Loss Graphs (Cell 9)

**Good training looks like:**
- Training accuracy ↑ over epochs
- Validation accuracy ↑ (ideally close to training)
- Training loss ↓
- Validation loss ↓

**Signs of overfitting:**
- Training accuracy >> Validation accuracy (large gap)
- Validation loss starts increasing after some epoch

**What to do if overfitting:**
- Add more augmentation
- Increase dropout rate
- Reduce model complexity
- Add more training data

### Interpreting Predictions (Cell 10)
- Output probability = chance it's a **dog**
- `80% dog` → model is 80% sure it's a dog
- `20% dog` → model is 80% sure it's a cat (20% dog = 80% cat)

---

## 📅 Journal Entry 6 — Reflections & Next Steps

### What Worked
- 4-block Conv+Pool architecture captured enough features for 65%+ accuracy
- Data augmentation notably reduced the gap between train/val accuracy
- Dropout(0.5) helped prevent severe overfitting

### What Could Be Improved
1. **Transfer Learning** — Use a pretrained model (MobileNetV2, VGG16) for much higher accuracy with less training data
2. **More Epochs** — Training longer (25–50 epochs) may yield better results
3. **Learning Rate Scheduling** — Gradually reduce learning rate as training progresses
4. **More Data** — The full Kaggle cats-and-dogs dataset has 25,000 images

### Concepts to Study Next
- [ ] Transfer learning with Keras `applications` module
- [ ] Batch Normalization layers
- [ ] Learning rate schedulers
- [ ] Grad-CAM (visualize what the CNN "sees")

---

## 📊 Final Results Log

| Run | Epochs | Augmentation | Dropout | Val Accuracy | Notes |
|-----|--------|-------------|---------|-------------|-------|
| 1 | 15 | No | No | ~58% | Overfitting observed |
| 2 | 15 | Yes | 0.5 | ~68% | Better generalization |
| 3 | 25 | Yes | 0.5 | ~72% | Bonus threshold hit! |

*(Update this table as you experiment)*

---

*Journal maintained as part of the freeCodeCamp Machine Learning with Python certification.*
