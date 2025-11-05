# Speaker Verification System - File Index

## 📂 Complete File Listing

### 📋 Documentation Files

- **README.md** - Complete project documentation with installation, usage, and references
- **QUICKSTART.md** - Fast 5-minute setup and quick command reference
- **PROJECT_SUMMARY.md** - Detailed implementation summary and validation checklist
- **requirements.txt** - Python dependencies list

### ⚙️ Configuration Files

- **config/ecapa_config.yaml** - ECAPA-TDNN model configuration (hyperparameters, training settings)
- **config/titanet_config.yaml** - TiTANet model configuration (hyperparameters, training settings)

### 🎯 Main Scripts

- **train.py** - Main training script for both models
- **evaluate.py** - Comprehensive evaluation script with metrics and visualizations
- **compare_models.py** - Side-by-side comparison of ECAPA-TDNN vs TiTANet
- **demo.py** - Interactive demo for speaker verification (single and batch modes)
- **utils.py** - Utility script for environment checks, dataset validation, model inspection

### 🧩 Source Modules (src/)

#### Core Functionality

- **src/**init**.py** - Package initialization
- **src/dataset.py** - Dataset loading and preprocessing classes

  - `SpeakerVerificationDataset`
  - `PairwiseVerificationDataset`
  - `collate_fn`, `get_dataloader`

- **src/features.py** - Feature extraction implementations

  - `FbankFeatureExtractor` - 80-dim log-Mel filterbanks
  - `OnlineFbankExtractor` - Streaming feature extraction
  - `extract_features_from_file`

- **src/augmentation.py** - Data augmentation techniques

  - `AudioAugmentation` - Composite augmentation
  - `SpecAugment` - Time and frequency masking
  - `BackgroundNoise` - Real noise addition

- **src/evaluation.py** - Evaluation metrics

  - `compute_eer`, `compute_minDCF`
  - `VerificationMetrics` class
  - `compute_cosine_similarity`
  - `print_metrics`, `evaluate_verification_pairs`

- **src/verification.py** - Verification systems

  - `CosineScorer` - Cosine similarity-based scoring
  - `PLDAScorer` - PLDA-based scoring
  - `SpeakerVerifier` - High-level verification interface

- **src/visualization.py** - Visualization tools
  - `plot_tsne` - t-SNE embedding visualization
  - `plot_roc_curve` - ROC curves with EER
  - `plot_det_curve` - DET curves
  - `plot_score_distribution` - Score histograms
  - `plot_training_history` - Training curves
  - `plot_confusion_matrix` - Similarity matrices

#### Model Implementations (src/models/)

- **src/models/**init**.py** - Models package initialization
- **src/models/ecapa_tdnn.py** - ECAPA-TDNN implementation

  - `ECAPA_TDNN_Wrapper` - Model wrapper class
  - `AAMSoftmax` - AAM-Softmax loss implementation

- **src/models/titanet.py** - TiTANet implementation
  - `TiTANet_Wrapper` - Model wrapper class
  - Pretrained model loading and fine-tuning

### 🛠️ Setup Scripts

- **setup.bat** - Windows setup script (automated environment setup)
- **setup.sh** - Linux/Mac setup script (automated environment setup)

### 📊 Data Directory Structure

```
data/
├── Train/                    # Training data (created by user)
│   ├── 1034/                # Speaker ID folder
│   │   ├── 1034_trn_vp_a_1.wav
│   │   ├── 1034_trn_vp_a_2.wav
│   │   └── 1034_trn_vp_a_3.wav
│   └── ... (351 speakers total)
└── Test/                     # Test data (created by user)
    ├── 1034/
    │   ├── 1034_tst_vp_a_001.wav
    │   └── ... (25 files per speaker)
    └── ... (351 speakers total)
```

### 📁 Generated Directories (Created During Use)

#### Checkpoints

```
checkpoints/
├── ecapa/
│   ├── best_model.pt        # Best ECAPA model (based on EER)
│   └── checkpoint_epoch_*.pt # Periodic checkpoints
└── titanet/
    ├── best_model.pt        # Best TiTANet model
    └── checkpoint_epoch_*.pt
```

#### Logs

```
logs/
├── ecapa/
│   └── training_history.png
└── titanet/
    └── training_history.png
```

#### Results

```
results/
├── ecapa_results.txt        # Detailed metrics
├── ecapa_roc_curve.png      # ROC curve
├── ecapa_score_distribution.png
├── ecapa_tsne.png           # t-SNE visualization
├── titanet_results.txt
├── titanet_roc_curve.png
├── titanet_score_distribution.png
├── titanet_tsne.png
└── comparison/
    ├── model_comparison.png
    └── comparison_results.txt
```

#### Pretrained Models (Downloaded Automatically)

```
pretrained_models/
├── ecapa/                   # SpeechBrain ECAPA-TDNN
└── titanet/                 # NVIDIA NeMo TiTANet (if using NeMo)
```

## 🔧 Quick Reference by Task

### Setup & Validation

- Environment check: `python utils.py check-env`
- Dataset check: `python utils.py check-dataset`
- Model check: `python utils.py check-model <checkpoint>`
- List checkpoints: `python utils.py list-checkpoints`
- Test audio: `python utils.py test-audio <audio_file>`

### Training

- Train ECAPA: `python train.py --config config/ecapa_config.yaml --model ecapa`
- Train TiTANet: `python train.py --config config/titanet_config.yaml --model titanet`

### Evaluation

- Evaluate model: `python evaluate.py --config <config> --checkpoint <checkpoint> --model <type>`
- Compare models: `python compare_models.py --ecapa-checkpoint <path> --titanet-checkpoint <path>`

### Demo & Testing

- Single verify: `python demo.py verify --model <checkpoint> --enroll <files> --test <file>`
- Batch verify: `python demo.py batch --model <checkpoint> --enroll-dir <dir> --test-dir <dir>`

## 📊 File Dependencies

### Training Pipeline

```
train.py
├── config/*.yaml
├── src/dataset.py
├── src/augmentation.py
├── src/models/ecapa_tdnn.py or titanet.py
├── src/evaluation.py (for validation)
└── src/visualization.py (for plots)
```

### Evaluation Pipeline

```
evaluate.py
├── config/*.yaml
├── src/dataset.py
├── src/models/ecapa_tdnn.py or titanet.py
├── src/evaluation.py
├── src/verification.py
└── src/visualization.py
```

### Demo Pipeline

```
demo.py
├── src/models/ecapa_tdnn.py or titanet.py
└── src/verification.py
```

## 📈 Code Statistics

### Total Lines of Code (Approximate)

- **Core Modules**: ~3,500 lines

  - dataset.py: ~400 lines
  - features.py: ~450 lines
  - augmentation.py: ~500 lines
  - evaluation.py: ~400 lines
  - verification.py: ~450 lines
  - visualization.py: ~550 lines
  - models/: ~750 lines

- **Scripts**: ~1,500 lines

  - train.py: ~450 lines
  - evaluate.py: ~350 lines
  - compare_models.py: ~250 lines
  - demo.py: ~250 lines
  - utils.py: ~200 lines

- **Total**: ~5,000 lines of Python code
- **Documentation**: ~2,000 lines
- **Configuration**: ~300 lines

## 🎯 Key Components

### Must-Read Files (Start Here)

1. **QUICKSTART.md** - Get started quickly
2. **README.md** - Understand the system
3. **config/ecapa_config.yaml** - See all configurable parameters
4. **src/dataset.py** - Understand data flow
5. **train.py** - See training process

### Most Important Classes

1. `SpeakerVerificationDataset` (src/dataset.py)
2. `ECAPA_TDNN_Wrapper` (src/models/ecapa_tdnn.py)
3. `TiTANet_Wrapper` (src/models/titanet.py)
4. `Trainer` (train.py)
5. `Evaluator` (evaluate.py)
6. `SpeakerVerifier` (src/verification.py)

### Most Used Functions

1. `get_dataloader()` - Create data loaders
2. `compute_eer()` - Calculate EER
3. `plot_tsne()` - Visualize embeddings
4. `extract_embedding()` - Get speaker embeddings
5. `verify_speaker()` - Perform verification

## 🔄 Typical Workflow

```
1. Setup
   ├── Run setup.bat/setup.sh
   └── python utils.py check-env

2. Prepare Data
   ├── Place data in data/Train and data/Test
   └── python utils.py check-dataset

3. Configure
   └── Edit config/ecapa_config.yaml

4. Train
   ├── python train.py --config config/ecapa_config.yaml --model ecapa
   └── Monitor logs/ directory

5. Evaluate
   ├── python evaluate.py --config config/ecapa_config.yaml --checkpoint checkpoints/ecapa/best_model.pt --model ecapa
   └── Check results/ directory

6. Compare (Optional)
   └── python compare_models.py

7. Deploy/Test
   └── python demo.py verify --model checkpoints/ecapa/best_model.pt --enroll ... --test ...
```

## 📚 Additional Resources

### External Dependencies Documentation

- PyTorch: https://pytorch.org/docs/
- TorchAudio: https://pytorch.org/audio/
- SpeechBrain: https://speechbrain.github.io/
- NVIDIA NeMo: https://docs.nvidia.com/deeplearning/nemo/

### Research Papers

- ECAPA-TDNN: Desplanques et al., INTERSPEECH 2020
- TiTANet: Koluguri et al., arXiv 2022
- AAM-Softmax: Deng et al., CVPR 2019

---

**Last Updated**: November 2025
**Total Files**: 30+ (including generated)
**Project Status**: ✅ Complete
