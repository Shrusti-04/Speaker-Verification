# Speaker Verification System for Regional Languages (Hindi & Kannada)

A comprehensive speaker verification system implementing and comparing **ECAPA-TDNN** and **TiTANet** architectures for regional language (Hindi and Kannada) speaker recognition.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [References](#references)

## 🎯 Overview

This project implements a state-of-the-art speaker verification system designed for regional Indian languages, specifically Hindi and Kannada. It provides:

- **Two Model Architectures**: ECAPA-TDNN (SpeechBrain) and TiTANet (NVIDIA NeMo)
- **Fine-tuning Pipeline**: Adaptation from VoxCeleb pretrained models
- **Comprehensive Evaluation**: EER, minDCF, ROC curves, t-SNE visualizations
- **Robust Training**: Data augmentation (noise, reverb, speed perturbation)
- **Multiple Scoring Methods**: Cosine similarity and PLDA

## ✨ Features

### Data Processing

- ✅ Automatic audio preprocessing (8kHz mono)
- ✅ Variable-length audio handling
- ✅ Train/test split management
- ✅ Speaker-based data organization

### Feature Extraction

- ✅ 80-dimensional log-Mel filterbanks (FBANK)
- ✅ Cepstral Mean and Variance Normalization (CMVN)
- ✅ Optional delta and delta-delta features
- ✅ Online feature extraction for streaming

### Data Augmentation

- ✅ Additive noise (configurable SNR)
- ✅ Reverberation simulation
- ✅ Speed perturbation (0.95x - 1.05x)
- ✅ SpecAugment (time and frequency masking)

### Model Training

- ✅ Pretrained model loading (VoxCeleb2)
- ✅ Fine-tuning with AAM-Softmax loss
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ Automatic mixed precision (AMP)
- ✅ Checkpoint management

### Evaluation

- ✅ Equal Error Rate (EER)
- ✅ Minimum Detection Cost Function (minDCF)
- ✅ ROC and DET curves
- ✅ Score distribution plots
- ✅ t-SNE visualization
- ✅ Confusion matrices

## 📊 Dataset

### Dataset Characteristics

- **Languages**: Hindi and Kannada
- **Total Speakers**: 351
- **Training Files**: 3 per speaker (1,053 total)
- **Test Files**: 25 per speaker (8,775 total)
- **Audio Format**:
  - Sample rate: 8 kHz (telephone quality)
  - Duration: ~6 seconds per file
  - Channels: Mono
  - Bit depth: 16-bit
  - Format: WAV

### Dataset Structure

```
data/
├── Train/
│   ├── {speaker_id}/
│   │   ├── {id}_trn_vp_a_1.wav
│   │   ├── {id}_trn_vp_a_2.wav
│   │   └── {id}_trn_vp_a_3.wav
└── Test/
    ├── {speaker_id}/
    │   ├── {id}_tst_vp_a_001.wav
    │   ├── {id}_tst_vp_a_002.wav
    │   └── ... (25 files total)
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM recommended
- 10GB+ free disk space

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd REU2
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
# Install PyTorch (with CUDA support)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install SpeechBrain
pip install speechbrain

# Optional: Install NVIDIA NeMo (for TiTANet)
# Note: This requires additional setup, see NeMo documentation
pip install nemo_toolkit[asr]
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "import speechbrain; print('SpeechBrain installed successfully')"
```

## 📁 Project Structure

```
REU2/
├── config/                      # Configuration files
│   ├── ecapa_config.yaml       # ECAPA-TDNN configuration
│   └── titanet_config.yaml     # TiTANet configuration
├── src/                        # Source code
│   ├── __init__.py
│   ├── dataset.py             # Dataset loading and preprocessing
│   ├── features.py            # Feature extraction (FBANK)
│   ├── augmentation.py        # Data augmentation
│   ├── evaluation.py          # Evaluation metrics (EER, minDCF)
│   ├── verification.py        # Verification module (Cosine, PLDA)
│   ├── visualization.py       # Plotting and visualization
│   └── models/                # Model implementations
│       ├── __init__.py
│       ├── ecapa_tdnn.py      # ECAPA-TDNN wrapper
│       └── titanet.py         # TiTANet wrapper
├── data/                       # Dataset directory
│   ├── Train/                 # Training data
│   └── Test/                  # Test data
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎓 Usage

### 1. Configure Training

Edit `config/ecapa_config.yaml` or `config/titanet_config.yaml` to customize:

- Model parameters (embedding dimension, architecture)
- Training hyperparameters (learning rate, batch size, epochs)
- Data augmentation settings
- Hardware settings (GPU/CPU, workers)

### 2. Train ECAPA-TDNN Model

```bash
python train.py --config config/ecapa_config.yaml --model ecapa
```

### 3. Train TiTANet Model

```bash
python train.py --config config/titanet_config.yaml --model titanet
```

### 4. Evaluate Model

```bash
# Evaluate ECAPA-TDNN with cosine similarity
python evaluate.py \
    --config config/ecapa_config.yaml \
    --checkpoint checkpoints/ecapa/best_model.pt \
    --model ecapa \
    --scorer cosine

# Evaluate with PLDA scoring
python evaluate.py \
    --config config/ecapa_config.yaml \
    --checkpoint checkpoints/ecapa/best_model.pt \
    --model ecapa \
    --scorer plda \
    --num-pairs 10000
```

### 5. Compare Models

```bash
# Evaluate both models
python evaluate.py --config config/ecapa_config.yaml --checkpoint checkpoints/ecapa/best_model.pt --model ecapa
python evaluate.py --config config/titanet_config.yaml --checkpoint checkpoints/titanet/best_model.pt --model titanet
```

### 6. Custom Verification

```python
from src.models.ecapa_tdnn import ECAPA_TDNN_Wrapper
from src.verification import SpeakerVerifier

# Load model
model = ECAPA_TDNN_Wrapper(embedding_dim=192, num_speakers=351)
model.load_checkpoint("checkpoints/ecapa/best_model.pt")

# Create verifier
verifier = SpeakerVerifier(model, scorer_type="cosine")

# Enroll speaker
enrollment_files = ["data/Train/1034/1034_trn_vp_a_1.wav",
                    "data/Train/1034/1034_trn_vp_a_2.wav",
                    "data/Train/1034/1034_trn_vp_a_3.wav"]
enrollment_embedding = verifier.enroll_speaker(enrollment_files)

# Verify test audio
is_same, score = verifier.verify(
    enrollment_embedding,
    "data/Test/1034/1034_tst_vp_a_001.wav",
    threshold=0.5
)

print(f"Same speaker: {is_same}, Score: {score:.4f}")
```

## 🏗️ Model Architectures

### ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)

- **Source**: [SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Pretrained on**: VoxCeleb2
- **Embedding Dimension**: 192
- **Key Features**:
  - SE-Res2Block with channel attention
  - Multi-layer feature aggregation
  - Attentive statistical pooling
  - AAM-Softmax loss for training

### TiTANet (TItanet Transformer Attentive Network)

- **Source**: [NVIDIA NeMo](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
- **Pretrained on**: VoxCeleb
- **Embedding Dimension**: 192
- **Key Features**:
  - Transformer-based architecture
  - Multi-head self-attention
  - Squeeze-and-excitation blocks
  - Superior performance on long utterances

## 📈 Evaluation Metrics

### Equal Error Rate (EER)

The point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR). Lower is better.

### Minimum Detection Cost Function (minDCF)

```
DCF = C_miss × P_miss × P_target + C_fa × P_fa × (1 - P_target)
```

Where:

- C_miss = Cost of missing a target (default: 1)
- C_fa = Cost of false alarm (default: 1)
- P_target = Prior probability of target speaker (default: 0.01)

### Visualization Outputs

- **ROC Curves**: True Positive Rate vs False Positive Rate
- **DET Curves**: Detection Error Tradeoff curves
- **Score Distributions**: Genuine vs impostor score histograms
- **t-SNE Plots**: 2D visualization of speaker embeddings
- **Confusion Matrices**: Speaker similarity matrices

## 📊 Results

Results will be saved in the `results/` directory:

```
results/
├── ecapa_results.txt          # EER, minDCF metrics
├── ecapa_roc_curve.png        # ROC curve
├── ecapa_score_distribution.png  # Score histogram
├── ecapa_tsne.png             # t-SNE visualization
├── titanet_results.txt
├── titanet_roc_curve.png
├── titanet_score_distribution.png
└── titanet_tsne.png
```

### Expected Performance (on similar datasets)

| Model      | EER (%) | minDCF    |
| ---------- | ------- | --------- |
| ECAPA-TDNN | 2-4%    | 0.15-0.25 |
| TiTANet    | 1.5-3%  | 0.12-0.20 |

_Note: Actual results depend on dataset quality and training_

## 🔧 Troubleshooting

### Out of Memory Errors

- Reduce batch size in config file
- Enable gradient accumulation
- Use mixed precision training

### Slow Training

- Increase number of workers
- Enable pin_memory
- Use DataLoader prefetch
- Check GPU utilization

### Poor Performance

- Increase training epochs
- Adjust learning rate
- Enable more augmentation
- Check data quality

## 📚 References

### ECAPA-TDNN

```
@inproceedings{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={INTERSPEECH},
  year={2020}
}
```

### TiTANet

```
@article{koluguri2022titanet,
  title={Titanet: Neural model for speaker representation with 1d depth-wise separable convolutions and global context},
  author={Koluguri, Nithin and Park, Taejin and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2110.04410},
  year={2022}
}
```

### SpeechBrain

```
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Ravanelli, Mirco and others},
  howpublished={\url{https://speechbrain.github.io/}},
  year={2021}
}
```

## 📝 License

This project is for research and educational purposes. Please cite appropriately if used in publications.

## 👥 Contributors

- Research Team: Speaker Verification for Regional Languages

## 🙏 Acknowledgments

- SpeechBrain team for ECAPA-TDNN implementation
- NVIDIA NeMo team for TiTANet model
- VoxCeleb dataset creators
- Regional language dataset contributors

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the research team.

---

**Note**: This is a research project. Performance may vary based on dataset quality, hardware, and hyperparameters. Always validate on your specific use case.
