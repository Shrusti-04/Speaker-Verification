# File Index - REU2 Speaker Verification Project

Complete guide to all files and their purposes in the project.

---

## 📁 **Core Scripts (Root Directory)**

### `train.py`

**Purpose:** Main training pipeline for ECAPA-TDNN model  
**What it does:**

- Loads balanced or imbalanced data based on config
- Implements two-stage fine-tuning (frozen encoder → full training)
- Handles data augmentation during training
- Saves best model checkpoint based on validation accuracy
- Generates training history plots
- Supports AAM-Softmax loss with Adam optimizer
- Monitors validation EER and accuracy after each epoch

**Usage:** `python train.py --config config/ecapa_balanced_config.yaml`

---

### `evaluate.py`

**Purpose:** Complete evaluation script for trained models  
**What it does:**

- Loads trained model from checkpoint
- Extracts 192-D embeddings from test set
- Computes Equal Error Rate (EER) from 10,000 verification trials
- Generates genuine and impostor score distributions
- Calculates ROC curve and AUC
- Creates t-SNE visualization of embedding space
- Produces score distribution histograms
- Saves all results and plots to `results/` directory
- Reports accuracy, EER, and other metrics

**Usage:** `python evaluate.py --config config/ecapa_balanced_config.yaml`

---

### `demo.py`

**Purpose:** Interactive demo for speaker verification testing  
**What it does:**

- Enrolls a speaker using 2-5 audio samples
- Tests against single or batch of test audio files
- Computes cosine similarity between embeddings
- Makes accept/reject decisions at multiple thresholds
- Displays verification results with similarity scores
- Supports both single and batch verification modes
- Useful for quick testing and demonstrating the system

**Usage:** `python demo.py batch --model checkpoints/ecapa_balanced/best_model.pt --enroll-dir data/Train/1034 --test-dir data/Test/1034`

---

### `README.md`

**Purpose:** Project documentation and quick start guide  
**What it does:**

- Provides project overview and objectives
- Lists installation instructions
- Explains dataset structure
- Shows training and evaluation commands
- Documents key results (7.88% EER, 88.7% accuracy)
- Includes troubleshooting tips

---

### `requirements.txt`

**Purpose:** Python package dependencies  
**What it does:**

- Lists all required Python libraries with versions
- Includes PyTorch, SpeechBrain, torchaudio, scikit-learn, etc.
- Ensures reproducible environment setup

**Usage:** `pip install -r requirements.txt`

---

## ⚙️ **Configuration Files (config/)**

### `config/ecapa_balanced_config.yaml`

**Purpose:** Configuration for balanced 80/20 data split training  
**What it does:**

- Defines model architecture parameters (192-D embeddings, 351 speakers)
- Sets training hyperparameters (lr=0.0001, batch_size=32, 15 epochs)
- Enables combined data loading with 80/20 per-speaker split
- Specifies augmentation settings (speed, noise, reverb)
- Defines checkpoint and logging directories
- Configures AAM-Softmax loss parameters (margin=0.2, scale=30)
- Sets freeze encoder epochs (5 epochs)

**This config achieved 7.88% EER and 88.7% accuracy**

---

### `config/ecapa_config.yaml`

**Purpose:** Configuration for imbalanced baseline training  
**What it does:**

- Same model architecture as balanced config
- Uses original Train/Test folder split (imbalanced)
- Results in only 3 training files per speaker
- Used for baseline comparison

**This config achieved 24.90% EER and 62.5% accuracy (poor performance)**

---

## 🔧 **Source Code (src/)**

### `src/dataset.py`

**Purpose:** Dataset loading and preprocessing pipeline  
**What it does:**

- Implements `SpeakerVerificationDataset` class
- Loads audio files from Train/Test directories
- Handles balanced per-speaker 80/20 splitting via `_load_combined_dataset()`
- Resamples audio to 8 kHz
- Adjusts audio duration (2-10 seconds range)
- Converts stereo to mono
- Implements custom `collate_fn` for variable-length audio batching
- Supports data augmentation via transforms
- Creates DataLoaders with proper batching

**Key Functions:**

- `_load_combined_dataset()` - Implements balanced splitting
- `_adjust_duration()` - Handles audio length normalization
- `collate_fn()` - Pads audio to same length in batch

---

### `src/augmentation.py`

**Purpose:** Audio data augmentation for training robustness  
**What it does:**

- Implements `AudioAugmentation` class with multiple techniques
- **Speed Perturbation:** 0.95x, 1.0x, 1.05x factors
- **Noise Addition:** White noise with SNR 0-15 dB
- **Reverberation:** Simulates room acoustics with exponential decay
- **SpecAugment:** Time and frequency masking (optional)
- Applies augmentations randomly during training
- Helps model generalize to different recording conditions

**Key Classes:**

- `AudioAugmentation` - Main augmentation pipeline
- `SpecAugment` - Frequency/time masking
- `BackgroundNoise` - Real noise file addition

---

### `src/evaluation.py`

**Purpose:** Evaluation metrics and verification computations  
**What it does:**

- Computes Equal Error Rate (EER) from genuine/impostor scores
- Calculates False Acceptance Rate (FAR) and False Rejection Rate (FRR)
- Generates ROC curves and computes AUC
- Implements `VerificationMetrics` class for comprehensive evaluation
- Computes cosine similarity between embeddings
- Creates verification trials (genuine + impostor pairs)
- Supports threshold-based decision making

**Key Functions:**

- `compute_eer()` - Calculates EER from scores
- `compute_cosine_similarity()` - Embedding similarity
- `generate_trials()` - Creates verification pairs

---

### `src/verification.py`

**Purpose:** Speaker verification scoring and decision logic  
**What it does:**

- Implements `CosineScorer` for cosine similarity verification
- Implements `PLDAScorer` for PLDA-based verification (advanced)
- Handles enrollment (storing reference embeddings)
- Performs verification (comparing test vs enrolled embeddings)
- Makes accept/reject decisions based on thresholds
- Used by demo.py and evaluate.py for verification

**Key Classes:**

- `CosineScorer` - Simple cosine similarity verification
- `PLDAScorer` - Probabilistic Linear Discriminant Analysis
- `SpeakerVerifier` - High-level verification interface

---

### `src/visualization.py`

**Purpose:** Plotting and visualization utilities  
**What it does:**

- Generates training history plots (loss, accuracy over epochs)
- Creates ROC curves with AUC display
- Produces score distribution histograms (genuine vs impostor)
- Generates t-SNE embedding space visualizations
- Creates confusion matrices
- Saves publication-quality figures
- Supports customizable plot styles and colors

**Key Functions:**

- `plot_training_history()` - Training curves
- `plot_roc_curve()` - ROC with AUC
- `plot_score_distribution()` - Histogram of scores
- `plot_tsne()` - Embedding space visualization

---

### `src/models/ecapa_tdnn.py`

**Purpose:** ECAPA-TDNN model wrapper and interface  
**What it does:**

- Wraps SpeechBrain's ECAPA-TDNN pretrained model
- Implements `ECAPA_TDNN_Wrapper` class
- Loads pretrained weights from VoxCeleb2
- Replaces classifier for custom number of speakers (351)
- Handles encoder freezing/unfreezing during training
- Extracts 192-dimensional speaker embeddings
- Implements AAM-Softmax loss function
- Provides forward pass for training and inference

**Key Classes:**

- `ECAPA_TDNN_Wrapper` - Main model wrapper
- `AAMSoftmax` - Additive Angular Margin Softmax loss

**Key Methods:**

- `load_pretrained()` - Loads VoxCeleb2 weights
- `extract_embedding()` - Gets 192-D embeddings
- `freeze_encoder()` / `unfreeze_encoder()` - Two-stage training

---

### `src/__init__.py`

**Purpose:** Python package initialization  
**What it does:**

- Makes `src/` a proper Python package
- Allows imports like `from src.dataset import ...`
- Currently empty but enables package structure

---

## 📊 **Data Directories**

### `data/`

**Purpose:** Audio dataset storage  
**Structure:**

```
data/
├── Train/
│   ├── 1034/  (speaker folders)
│   │   ├── 1034_trn_vp_a_1.wav
│   │   ├── 1034_trn_vp_a_2.wav
│   │   └── ...
│   ├── 1037/
│   └── ... (351 speakers)
└── Test/
    ├── 1034/
    ├── 1037/
    └── ... (351 speakers)
```

**What it contains:**

- 17,330 total audio files
- 351 speakers (Hindi and Kannada)
- ~49 files per speaker on average
- WAV format, variable length
- Organized by speaker ID folders

**Note:** With balanced splitting, these folders are combined and re-split 80/20 per speaker

---

## 💾 **Output Directories**

### `checkpoints/`

**Purpose:** Trained model weights storage  
**Structure:**

```
checkpoints/
├── ecapa/
│   └── best_model.pt  (imbalanced: 24.90% EER)
└── ecapa_balanced/
    └── best_model.pt  (balanced: 7.88% EER)
```

**What it contains:**

- PyTorch model state dictionaries (.pt files)
- Best models saved based on validation accuracy
- Used for evaluation and deployment

---

### `logs/`

**Purpose:** Training logs and history  
**Structure:**

```
logs/
├── ecapa/
│   └── training_history.png
└── ecapa_balanced/
    └── training_history.png
```

**What it contains:**

- Training/validation loss curves
- Accuracy progression plots
- Timestamped training runs
- Helpful for debugging and monitoring

---

### `results/`

**Purpose:** Evaluation results and visualizations  
**Structure:**

```
results/
├── ecapa_results.txt            (metrics text file)
├── ecapa_roc_curve.png          (ROC curve plot)
├── ecapa_score_distribution.png (histogram)
├── ecapa_tsne.png               (embedding visualization)
└── ... (similar for balanced)
```

**What it contains:**

- EER and accuracy metrics (text files)
- ROC curves with AUC
- Score distribution histograms
- t-SNE embedding space plots
- Confusion matrices

---

### `pretrained_models/`

**Purpose:** Downloaded pretrained model files  
**What it contains:**

- SpeechBrain ECAPA-TDNN pretrained on VoxCeleb2
- Hyperparameters and configuration files
- Embedding model checkpoints
- Label encoders

**Note:** Created automatically when loading pretrained models

---

## 📄 **Documentation (paper/)**

Contains project documentation, experimental setup, training logs, and paper drafts.

---

## 🎯 **Key Files Summary**

**For Training:**

- `train.py` + `config/ecapa_balanced_config.yaml`

**For Evaluation:**

- `evaluate.py` + `checkpoints/ecapa_balanced/best_model.pt`

**For Testing:**

- `demo.py` + trained checkpoint

**Core Implementation:**

- `src/dataset.py` - Data loading with balanced split
- `src/models/ecapa_tdnn.py` - ECAPA-TDNN model
- `src/augmentation.py` - Training augmentations
- `src/evaluation.py` - Metrics computation

**Results:**

- `results/` - All evaluation outputs
- `checkpoints/ecapa_balanced/` - Best model (7.88% EER)

---

## 📈 **File Usage Workflow**

1. **Setup:** Install dependencies from `requirements.txt`
2. **Prepare Data:** Organize audio in `data/Train/` and `data/Test/`
3. **Configure:** Edit `config/ecapa_balanced_config.yaml`
4. **Train:** Run `train.py` → saves to `checkpoints/`
5. **Evaluate:** Run `evaluate.py` → saves to `results/`
6. **Test:** Run `demo.py` for interactive verification
7. **Document:** Update `paper/` files with findings

---

**Total Project Structure:**

- **Core Scripts:** 3 files (train, evaluate, demo)
- **Configuration:** 2 YAML files
- **Source Code:** 6 Python modules + 1 model wrapper
- **Data:** 17,330 audio files across 351 speakers
- **Outputs:** Checkpoints, logs, results
- **Documentation:** Multiple markdown files

**This is a complete, clean, and working speaker verification system achieving 88.7% accuracy and 7.88% EER on Hindi/Kannada regional languages!**
