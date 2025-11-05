# Experimental Setup

## Dataset Characteristics

### Overview

- **Languages**: Hindi and Kannada (Indian regional languages)
- **Number of Speakers**: 351
- **Audio Quality**: 8 kHz telephone quality
- **Total Files**: 17,330 audio files

### Data Split

- **Training Set**:
  - 1,053 files (3 files per speaker)
  - 351 speakers
  - Used for fine-tuning pretrained model
- **Test Set**:
  - 16,277 files (~46 files per speaker)
  - Same 351 speakers
  - Used for verification evaluation

### Dataset Limitations

- **Limited Training Data**: Only 3 utterances per speaker for training
- **Challenge**: Low-resource scenario typical of regional language applications
- **Imbalanced Split**: Heavy bias toward test data (94% test, 6% train)

## Model Architecture

### ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Networks)

**Base Model**: SpeechBrain pretrained ECAPA-TDNN

- **Pretraining Dataset**: VoxCeleb2 (English speakers)
- **Source**: HuggingFace (`speechbrain/spkrec-ecapa-voxceleb`)
- **Embedding Dimension**: 192

**Architecture Components**:

1. **Feature Extraction**:

   - 80-dimensional log-Mel filterbank features
   - Window: 25ms, Hop: 10ms
   - Frequency range: 20-4000 Hz (8 kHz Nyquist)
   - Cepstral Mean and Variance Normalization (CMVN)

2. **Encoder** (ECAPA-TDNN):

   - Time Delay Neural Network layers
   - Squeeze-Excitation (SE) blocks for channel attention
   - Multi-layer feature aggregation
   - Statistics pooling (mean + std)
   - Output: 192-dimensional speaker embeddings

3. **Classifier**:
   - Additive Angular Margin Softmax (AAM-Softmax)
   - Margin: 0.2 (angular penalty between classes)
   - Scale: 30 (scaling factor for logits)
   - Output: 351 classes (one per speaker)

**Fine-tuning Strategy**:

- Transfer learning from English (VoxCeleb2) to Hindi/Kannada
- Both encoder and classifier adapted to regional speakers
- No layer freezing (encoder trained end-to-end)

## Training Configuration

### Hyperparameters

```yaml
# Data
batch_size: 32
num_workers: 0  # Windows compatibility

# Optimization
num_epochs: 30
learning_rate: 0.00005
optimizer: Adam
weight_decay: 0.0001

# Scheduler
scheduler: ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 3
  min_lr: 1e-6

# Loss Function
loss: AAM-Softmax
  margin: 0.2
  scale: 30

# Training Strategy
freeze_encoder_epochs: 0  # Encoder trained from epoch 1
```

### Data Augmentation

Limited augmentation due to small training set:

- **Time Masking**: max_time_mask=0.01 (1% of audio)
- **Frequency Masking**: max_freq_mask=5 (5 mel bands)
- **Speed Perturbation**: Disabled
- **Additive Noise**: Disabled

### Hardware and Environment

- **Platform**: Windows 10
- **Compute**: CPU-only (no CUDA available)
- **Python**: 3.10.11
- **PyTorch**: 2.7.1+cu118
- **SpeechBrain**: Latest version
- **Training Time**: ~8 hours total (30 epochs)
- **Time per Epoch**: ~15-30 minutes

### Windows-Specific Adjustments

- Disabled symlinks for HuggingFace model download (`local_dir_use_symlinks=False`)
- Set `num_workers=0` and `pin_memory=False` in DataLoader
- Used direct file copying instead of symlink strategies

## Evaluation Protocol

### Verification Trials

- **Total Pairs**: 10,000
  - **Positive Pairs (Target)**: 5,000 (same speaker)
  - **Negative Pairs (Non-target)**: 5,000 (different speakers)
- **Balanced**: Equal number of genuine and impostor trials

### Scoring Method

- **Similarity Metric**: Cosine similarity
- **Scoring**: `score = cosine_similarity(embedding1, embedding2)`
- **Range**: [-1, 1] where 1 = identical, -1 = opposite

### Performance Metrics

1. **Equal Error Rate (EER)**:

   - Point where False Acceptance Rate = False Rejection Rate
   - Lower is better
   - Primary metric for speaker verification

2. **Accuracy**:

   - Classification accuracy at optimal threshold
   - Percentage of correctly classified pairs

3. **Minimum Detection Cost Function (minDCF)**:

   - NIST metric: `minDCF = min[C_miss × P_miss × P_target + C_fa × P_fa × (1 - P_target)]`
   - C_miss = 1, C_fa = 1, P_target = 0.01
   - Lower is better

4. **Threshold**:
   - Cosine similarity threshold at EER point
   - Used to decide accept/reject

### Visualization

- **ROC Curve**: True Positive Rate vs False Positive Rate
- **Score Distribution**: Histogram of genuine vs impostor scores
- **Training History**: Loss and EER over epochs
- **t-SNE**: 2D projection of speaker embeddings (planned)

## Training History

### First Training Run (Failed)

- **Configuration**: 15 epochs, encoder frozen for first 10 epochs
- **Best Validation EER**: 15.47% (epoch 15)
- **Test EER**: 49.94% (random performance)
- **Issue**: Encoder not adapted to regional speakers (only 5 unfrozen epochs insufficient)

### Extended Training Run (Successful)

- **Configuration**: 30 epochs, encoder unfrozen throughout
- **Best Validation EER**: 6.54% (epoch 18)
- **Final Test EER**: 24.90%
- **Overfitting Gap**: 18.36 percentage points (validation to test)

### Key Findings

- Pretrained encoder (VoxCeleb2, English) requires significant adaptation for Hindi/Kannada
- Freezing encoder prevented discriminative embeddings for regional speakers
- Small training set (3 files/speaker) causes substantial overfitting
- Model memorizes training speakers but generalizes moderately to test trials
