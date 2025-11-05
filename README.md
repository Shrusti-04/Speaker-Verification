# Speaker Verification System for Regional Languages (Hindi & Kannada)

A deep learning-based speaker verification system implementing **ECAPA-TDNN** architecture for regional language (Hindi and Kannada) speaker recognition, achieving **7.88% EER** with balanced data distribution.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Achievements](#key-achievements)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Demo](#demo)
- [References](#references)

## 🎯 Overview

This project implements a state-of-the-art speaker verification system designed for regional Indian languages, specifically Hindi and Kannada. The system:

- **ECAPA-TDNN Architecture**: Fine-tuned from VoxCeleb2 pretrained model
- **Balanced Data Distribution**: Per-speaker 80/20 train/test split
- **Comprehensive Evaluation**: EER, accuracy, ROC curves, t-SNE visualizations
- **Robust Training**: Two-stage fine-tuning with data augmentation
- **Production-Ready**: Interactive demo with batch verification support

## 🏆 Key Achievements

- ✅ **7.88% Test EER** with balanced data distribution
- ✅ **88.7% Accuracy** on Hindi/Kannada speaker verification
- ✅ **68.4% Relative Improvement** over imbalanced baseline (24.90% → 7.88%)
- ✅ **100% Demo Accuracy** on genuine verification (10/10) and impostor rejection (4/4)
- ✅ **351 Speakers** with 17,330 audio files
- ✅ **Two-Stage Fine-Tuning** with encoder freezing strategy

## ✨ Features

### Data Processing

- ✅ Balanced per-speaker 80/20 split (13,725 train / 3,605 test files)
- ✅ Automatic audio preprocessing (8kHz mono)
- ✅ Variable-length audio handling (2-10 seconds)
- ✅ Speaker-based data organization (351 speakers)

### Data Augmentation

- ✅ Speed perturbation (0.95x, 1.0x, 1.05x)
- ✅ Additive white noise (SNR 0-15 dB)
- ✅ Reverberation simulation

### Model Training

- ✅ Pretrained ECAPA-TDNN from VoxCeleb2
- ✅ Two-stage fine-tuning (frozen encoder → full training)
- ✅ AAM-Softmax loss with margin=0.2, scale=30
- ✅ Adam optimizer with lr=0.0001
- ✅ Automatic checkpoint saving (best validation accuracy)
- ✅ Training history visualization

### Evaluation

- ✅ Equal Error Rate (EER) computation
- ✅ Accuracy metrics on test set
- ✅ ROC curves with visualization
- ✅ Score distribution plots (genuine vs impostor)
- ✅ t-SNE embedding space visualization
- ✅ Batch verification for multiple samples

## 📊 Dataset

### Dataset Characteristics

- **Languages**: Hindi and Kannada
- **Total Speakers**: 351
- **Total Audio Files**: 17,330
- **Files per Speaker**: ~49 (average)
- **Data Split**:
  - **Balanced**: 80% train (13,725 files) / 20% test (3,605 files) per speaker
  - **Imbalanced Baseline**: 3 train files per speaker from Train/ folder only
- **Audio Format**:
  - Sample rate: 8 kHz (telephone quality)
  - Duration: Variable (~2-10 seconds)
  - Channels: Mono
  - Bit depth: 16-bit
  - Format: WAV

### Dataset Structure

```
data/
├── Train/
│   ├── 1034/
│   │   ├── 1034_trn_vp_a_1.wav
│   │   ├── 1034_trn_vp_a_2.wav
│   │   └── ... (~49 files)
│   ├── 1037/
│   └── ... (351 speakers)
└── Test/
    ├── 1034/
    │   ├── 1034_tst_vp_a_001.wav
    │   ├── 1034_tst_vp_a_002.wav
    │   └── ... (~49 files)
    ├── 1037/
    └── ... (351 speakers)
```

**Note**: With balanced splitting enabled in config, both Train/ and Test/ folders are combined and re-split 80/20 per speaker for better performance.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support) or Google Colab with Tesla T4
- 8GB+ RAM recommended
- 5GB+ free disk space (excluding dataset)

### Step 1: Clone Repository

```bash
git clone https://github.com/Shrusti-04/Speaker-Verification.git
cd Speaker-Verification
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n speaker_verification python=3.9
conda activate speaker_verification

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Key Dependencies:**

- `torch>=2.0.0` - PyTorch deep learning framework
- `torchaudio>=2.0.0` - Audio processing
- `speechbrain>=0.5.0` - ECAPA-TDNN pretrained models
- `scikit-learn>=1.3.0` - Evaluation metrics
- `matplotlib>=3.7.0` - Visualization
- `pyyaml>=6.0` - Configuration management

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import speechbrain; print('SpeechBrain OK')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
```

## 📁 Project Structure

```
Speaker-Verification/
├── config/                          # Configuration files
│   ├── ecapa_balanced_config.yaml  # Balanced split (7.88% EER)
│   └── ecapa_config.yaml           # Imbalanced baseline (24.90% EER)
├── src/                            # Source code
│   ├── dataset.py                  # Dataset with balanced splitting
│   ├── augmentation.py             # Audio augmentation
│   ├── evaluation.py               # EER and metrics computation
│   ├── verification.py             # Cosine similarity verification
│   ├── visualization.py            # Plotting utilities
│   └── models/
│       ├── ecapa_tdnn.py          # ECAPA-TDNN wrapper + AAMSoftmax
│       └── __init__.py
├── data/                           # Dataset (not pushed to GitHub)
│   ├── Train/                     # 351 speakers with audio files
│   └── Test/                      # 351 speakers with audio files
├── checkpoints/                    # Trained models (not pushed)
│   ├── ecapa/                     # Imbalanced baseline
│   └── ecapa_balanced/            # Best model (7.88% EER)
├── logs/                           # Training logs (not pushed)
├── results/                        # Evaluation outputs
│   ├── ecapa_results.txt          # Metrics
│   ├── ecapa_roc_curve.png        # ROC curve
│   ├── ecapa_score_distribution.png
│   └── ecapa_tsne.png
├── paper/                          # Documentation
│   ├── EXPERIMENTAL_SETUP.md
│   ├── TRAINING_LOG.md
│   ├── PROJECT_SUMMARY.md
│   └── figures/
├── train.py                        # Main training script
├── evaluate.py                     # Evaluation script
├── demo.py                         # Interactive verification demo
├── requirements.txt                # Dependencies
├── FILE_INDEX.md                   # Complete file documentation
└── README.md                       # This file
```

**Note**: Large files (data, checkpoints, logs) are excluded via `.gitignore`.

## 🎓 Usage

### 1. Prepare Dataset

Organize your audio files in the `data/` directory:

```
data/
├── Train/
│   ├── speaker_id_1/
│   │   └── *.wav files
│   └── speaker_id_2/
│       └── *.wav files
└── Test/
    ├── speaker_id_1/
    └── speaker_id_2/
```

### 2. Configure Training

Edit `config/ecapa_balanced_config.yaml` to customize:

- `use_combined_dataset: true` - Enable balanced 80/20 split
- `train_split: 0.8` - Train/test ratio per speaker
- `batch_size: 32` - Batch size (adjust for GPU memory)
- `learning_rate: 0.0001` - Learning rate
- `max_epochs: 15` - Number of training epochs
- `freeze_encoder_epochs: 5` - Epochs to freeze encoder

### 3. Train Model

```bash
# Train with balanced data (recommended)
python train.py --config config/ecapa_balanced_config.yaml

# Train with imbalanced data (baseline)
python train.py --config config/ecapa_config.yaml
```

**Training Progress:**

- Epoch 1-5: Encoder frozen, only classifier trains
- Epoch 6-15: Full model training with unfrozen encoder
- Best model saved based on validation accuracy
- Training history plots saved to `logs/`

**Expected Training Time:**

- Google Colab Tesla T4: ~4-5 hours for 15 epochs
- Local GPU (RTX 3080): ~2-3 hours

### 4. Evaluate Model

```bash
# Evaluate best model
python evaluate.py --config config/ecapa_balanced_config.yaml

# Evaluate specific checkpoint
python evaluate.py \
    --config config/ecapa_balanced_config.yaml \
    --checkpoint checkpoints/ecapa_balanced/best_model.pt
```

**Evaluation Outputs:**

- `results/ecapa_results.txt` - EER, accuracy metrics
- `results/ecapa_roc_curve.png` - ROC curve visualization
- `results/ecapa_score_distribution.png` - Score histograms
- `results/ecapa_tsne.png` - Embedding space visualization

### 5. Interactive Demo

```bash
# Single verification
python demo.py single \
    --model checkpoints/ecapa_balanced/best_model.pt \
    --enroll data/Train/1034/1034_trn_vp_a_1.wav \
    --test data/Test/1034/1034_tst_vp_a_001.wav

# Batch verification (enroll multiple + test multiple)
python demo.py batch \
    --model checkpoints/ecapa_balanced/best_model.pt \
    --enroll-dir data/Train/1034 \
    --test-dir data/Test/1034
```

**Demo Output Example:**

```
Enrollment: Processing 3 audio files...
Enrollment embedding created successfully

Testing against 10 audio files...
[✓] 1034_tst_vp_a_001.wav - MATCH (similarity: 0.7101)
[✓] 1034_tst_vp_a_002.wav - MATCH (similarity: 0.6845)
...

Results at threshold 0.50:
  Genuine: 10/10 correct (100.0%)
  Impostor: 4/4 correct (100.0%)
```

## 🏗️ Model Architecture

### ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)

- **Source**: [SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Pretrained on**: VoxCeleb2 (English speakers)
- **Embedding Dimension**: 192-D
- **Input**: Raw waveform at 8 kHz
- **Output**: 192-dimensional speaker embedding

**Architecture Components:**

- **SE-Res2Block**: Squeeze-and-Excitation with channel attention
- **Multi-layer Feature Aggregation**: Combines features from multiple layers
- **Attentive Statistical Pooling**: Weighted pooling across time
- **AAM-Softmax Loss**: Additive Angular Margin for better discrimination
  - Margin: 0.2
  - Scale: 30

**Model Size**: ~6.19M parameters

## 🔬 Training Strategy

### Two-Stage Fine-Tuning

**Stage 1: Frozen Encoder (Epochs 1-5)**

- Encoder weights frozen (pretrained from VoxCeleb2)
- Only classifier layer trains
- Adapts to 351 speakers
- Faster convergence

**Stage 2: Full Training (Epochs 6-15)**

- All layers unfrozen
- End-to-end fine-tuning
- Adapts to Hindi/Kannada audio characteristics
- Achieves final performance

### Data Distribution Strategy

**Balanced Split (Recommended):**

- Combines Train/ and Test/ folders
- Splits 80/20 per speaker
- Result: 13,725 train / 3,605 test files
- **Achieved 7.88% EER**

**Imbalanced Baseline:**

- Uses Train/ folder only (3 files/speaker)
- Result: 1,053 train / 8,775 test files
- Achieved 24.90% EER (poor performance)

### Data Augmentation

Applied randomly during training:

- **Speed Perturbation**: 0.95x, 1.0x, 1.05x (33% each)
- **Additive Noise**: White noise, SNR 0-15 dB
- **Reverberation**: Simulates room acoustics

### Hyperparameters

- **Optimizer**: Adam
- **Learning Rate**: 0.0001 (constant)
- **Batch Size**: 32
- **Max Epochs**: 15
- **Loss Function**: AAM-Softmax (margin=0.2, scale=30)
- **Hardware**: Google Colab Tesla T4 GPU
- **Training Time**: ~4-5 hours

## 📈 Evaluation Metrics

### Equal Error Rate (EER)

The threshold where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).

- **Lower is better**
- Our Result: **7.88% EER**

### Accuracy

Percentage of correct verification decisions (genuine acceptance + impostor rejection).

- Our Result: **88.7% accuracy**

### Cosine Similarity Scoring

Measures similarity between speaker embeddings:

- Range: [-1, 1]
- Higher values indicate same speaker
- Threshold: 0.50 (optimized for EER)

### Visualization Outputs

- **ROC Curves**: True Positive Rate vs False Positive Rate
- **Score Distributions**: Genuine vs impostor score histograms
- **t-SNE Plots**: 2D visualization of 192-D speaker embeddings
- **Training History**: Loss and accuracy curves over epochs

## 📊 Results

### Performance Comparison

| Configuration              | Data Split         | Train Files | Test Files | Test EER  | Accuracy  | Validation EER |
| -------------------------- | ------------------ | ----------- | ---------- | --------- | --------- | -------------- |
| **Balanced** (Recommended) | 80/20 per speaker  | 13,725      | 3,605      | **7.88%** | **88.7%** | 4.41%          |
| Imbalanced Baseline        | Train/ folder only | 1,053       | 8,775      | 24.90%    | 62.5%     | 8.96%          |

**Improvement**: **68.4% relative improvement** in EER (24.90% → 7.88%)

### Demo Testing Results

**Genuine Verification (Speaker 1034 vs 1034):**

- 10 test samples
- 10/10 correct acceptances (100% accuracy)
- Similarity scores: 0.5071 to 0.7101
- All above threshold (0.50)

**Impostor Detection (Speaker 1034 vs 1037):**

- 4 test samples
- 4/4 correct rejections (100% accuracy)
- Similarity scores: -0.0598 to 0.0589
- All below threshold (0.50)

### Output Files

Results saved in `results/` directory:

```
results/
├── ecapa_results.txt               # Detailed metrics
├── ecapa_roc_curve.png             # ROC curve (AUC visualization)
├── ecapa_score_distribution.png    # Genuine vs impostor histograms
└── ecapa_tsne.png                  # Embedding space visualization
```

### Key Findings

1. ✅ **Balanced data distribution is critical**: 68.4% improvement over imbalanced baseline
2. ✅ **Two-stage fine-tuning works well**: Frozen encoder prevents catastrophic forgetting
3. ✅ **ECAPA-TDNN generalizes to regional languages**: Despite training on English (VoxCeleb2)
4. ✅ **8 kHz sampling sufficient**: Telephone quality audio works for speaker verification
5. ✅ **Augmentation helps**: Speed/noise/reverb improves robustness

## 🎤 Demo

### Interactive Verification

The `demo.py` script provides easy-to-use speaker verification:

**Single Verification:**

```bash
python demo.py single \
    --model checkpoints/ecapa_balanced/best_model.pt \
    --enroll data/Train/1034/1034_trn_vp_a_1.wav \
    --test data/Test/1034/1034_tst_vp_a_001.wav
```

**Batch Verification:**

```bash
python demo.py batch \
    --model checkpoints/ecapa_balanced/best_model.pt \
    --enroll-dir data/Train/1034 \
    --test-dir data/Test/1034
```

**Custom Threshold:**

```bash
python demo.py batch \
    --model checkpoints/ecapa_balanced/best_model.pt \
    --enroll-dir data/Train/1034 \
    --test-dir data/Test/1034 \
    --thresholds 0.3 0.5 0.7
```

### Demo Features

- ✅ Enrollment from multiple audio samples (average embedding)
- ✅ Batch testing against multiple files
- ✅ Multiple threshold evaluation
- ✅ Visual feedback (✓/✗) for each verification
- ✅ Accuracy reporting for genuine and impostor trials

## 🔧 Troubleshooting

### Out of Memory Errors

```yaml
# In config file, reduce batch size
batch_size: 16 # or 8
```

### Slow Training

```yaml
# Increase DataLoader workers
num_workers: 4
pin_memory: true
```

### Poor Performance

- ✅ Ensure `use_combined_dataset: true` for balanced splitting
- ✅ Check `train_split: 0.8` for proper 80/20 ratio
- ✅ Verify audio files are 8 kHz mono WAV format
- ✅ Increase epochs if underfitting
- ✅ Enable data augmentation

### CUDA Out of Memory

```python
# Reduce batch size or use CPU
device = 'cpu'  # in config
```

## 📚 References

### ECAPA-TDNN

```bibtex
@inproceedings{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={INTERSPEECH},
  year={2020}
}
```

### SpeechBrain

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Ravanelli, Mirco and Parcollet, Titouan and others},
  howpublished={\url{https://speechbrain.github.io/}},
  year={2021}
}
```

### AAM-Softmax Loss

```bibtex
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}
```

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{speaker_verification_regional,
  title={Speaker Verification System for Regional Languages (Hindi and Kannada)},
  author={Shrusti-04},
  year={2025},
  howpublished={\url{https://github.com/Shrusti-04/Speaker-Verification}}
}
```

## 📄 License

This project is for research and educational purposes.

## 👥 Contributors

- **Shrusti** - Implementation and Research

## 🙏 Acknowledgments

- SpeechBrain team for ECAPA-TDNN pretrained models
- VoxCeleb dataset creators for pretraining data
- Regional language dataset contributors (Hindi/Kannada)
- Google Colab for free GPU access (Tesla T4)

## 📧 Contact

For questions or issues:

- Open a [GitHub Issue](https://github.com/Shrusti-04/Speaker-Verification/issues)
- Repository: [https://github.com/Shrusti-04/Speaker-Verification](https://github.com/Shrusti-04/Speaker-Verification)

---

## 🌟 Highlights

- ✅ **7.88% EER** on Hindi/Kannada speaker verification
- ✅ **68.4% improvement** over imbalanced baseline
- ✅ **100% demo accuracy** on test samples
- ✅ **Production-ready** with interactive demo
- ✅ **Well-documented** with comprehensive guides
- ✅ **Balanced data strategy** for optimal performance

**This project demonstrates the critical importance of proper data distribution in speaker verification systems!**
