# Transfer Learning for Low-Resource Speaker Verification in Regional Indian Languages

## Abstract

**Background**: Speaker verification systems typically require substantial amounts of training data per speaker, which is often unavailable for low-resource regional languages like Hindi and Kannada spoken in telephone communications.

**Objective**: This study investigates the effectiveness of transfer learning from English speaker embeddings to Hindi/Kannada speakers in an extreme low-resource scenario with only 3 training utterances per speaker.

**Methods**: We fine-tuned a pretrained ECAPA-TDNN model (originally trained on VoxCeleb2 English speakers) for 351 Hindi/Kannada speakers using 8 kHz telephone-quality audio. The model was trained for 30 epochs with AAM-Softmax loss and evaluated on 10,000 balanced verification trials.

**Results**: The fine-tuned model achieved 24.90% Equal Error Rate (EER) with 75.10% accuracy on held-out test data. Validation performance showed 6.54% EER at epoch 18, indicating an overfitting gap of 18.36 percentage points attributed to limited training data (3 files per speaker).

**Conclusions**: Transfer learning from English to regional Indian languages is feasible but requires careful encoder adaptation. The 24.90% test EER is reasonable given the extreme data constraint, significantly outperforming random guessing (50% EER) but falling short of well-resourced systems (5-10% EER). Increasing training data to 10-20 utterances per speaker is recommended for production deployment.

**Keywords**: Speaker verification, ECAPA-TDNN, transfer learning, low-resource languages, Hindi, Kannada, telephone speech, fine-tuning

---

## 1. Introduction

### 1.1 Motivation

Speaker verification is the task of determining whether two speech samples originate from the same person. While modern deep learning systems achieve excellent performance on high-resource languages like English, deploying these systems for regional Indian languages presents unique challenges:

1. **Data Scarcity**: Limited availability of labeled speaker data
2. **Quality Constraints**: Telephone-quality audio (8 kHz) reduces acoustic information
3. **Linguistic Diversity**: Phonetic differences from pretrained model languages
4. **Real-world Applications**: Call center authentication, forensic analysis, secure access

### 1.2 Research Questions

This work addresses three key questions:

1. **Can pretrained English speaker models transfer to Hindi/Kannada?**

   - How much encoder adaptation is required?
   - Does freezing encoder layers help or hurt?

2. **What performance is achievable with only 3 training files per speaker?**

   - How does this compare to standard benchmarks?
   - What is the overfitting behavior?

3. **What are the practical limitations and improvement opportunities?**
   - Data augmentation strategies?
   - Backend scoring alternatives?

### 1.3 Contributions

- Demonstrate successful transfer learning from English to Hindi/Kannada speakers
- Quantify performance-data tradeoff (3 files/speaker → 24.90% EER)
- Provide detailed analysis of overfitting in low-resource scenarios
- Document Windows-compatible implementation for accessibility

### 1.4 Paper Organization

- **Section 2**: Related work on speaker verification and transfer learning
- **Section 3**: Methodology (model architecture, dataset, training)
- **Section 4**: Experimental results and analysis
- **Section 5**: Discussion of findings and limitations
- **Section 6**: Conclusions and future work

---

## 2. Related Work

### 2.1 Speaker Verification Systems

**Traditional Approaches**:

- **i-vectors** (Dehak et al., 2011): Factor analysis in speaker space
- **PLDA** (Prince & Elder, 2007): Probabilistic backend scoring
- **GMM-UBM** (Reynolds et al., 2000): Gaussian mixture models

**Deep Learning Era**:

- **x-vectors** (Snyder et al., 2018): Time-delay neural networks
- **ECAPA-TDNN** (Desplanques et al., 2020): Channel attention mechanisms
- **TiTANet** (Koluguri et al., 2022): Transformer-based embeddings
- **WavLM** (Chen et al., 2022): Self-supervised learning from speech

### 2.2 Transfer Learning for Speaker Recognition

**Cross-language Transfer**:

- VoxCeleb (Nagrani et al., 2017): English pretraining standard
- VoxLingua107 (Valk & Alumäe, 2021): Multilingual speaker recognition
- Language-invariant features (Song et al., 2019)

**Low-resource Adaptation**:

- Few-shot speaker verification (Zhang et al., 2020)
- Meta-learning for speaker recognition (Ding et al., 2020)
- Data augmentation strategies (Ko et al., 2015)

### 2.3 Indian Language Speaker Recognition

- IITG-MV (Sarma et al., 2018): Multilingual Indian dataset
- Code-mixing challenges (Sitaram et al., 2019)
- Telephone speech characteristics (Ganapathy et al., 2014)

**Research Gap**: Limited work on extreme low-resource scenarios (3 files/speaker) for Hindi/Kannada telephone speech.

---

## 3. Methodology

_[Refer to EXPERIMENTAL_SETUP.md for complete details]_

### 3.1 Dataset

- **Speakers**: 351 (Hindi and Kannada)
- **Training**: 1,053 files (3 per speaker)
- **Test**: 16,277 files (~46 per speaker)
- **Quality**: 8 kHz telephone audio

### 3.2 Model Architecture

**ECAPA-TDNN Components**:

1. Feature extraction (80-dim log-Mel filterbanks)
2. Time-delay neural network with squeeze-excitation blocks
3. Statistics pooling (mean + std)
4. 192-dimensional speaker embeddings
5. AAM-Softmax classifier (margin=0.2, scale=30)

**Pretrained Base**: SpeechBrain `speechbrain/spkrec-ecapa-voxceleb` (VoxCeleb2)

### 3.3 Training Strategy

- **Optimizer**: Adam (lr=0.00005, weight_decay=0.0001)
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Batch size**: 32
- **Epochs**: 30
- **Freezing**: No encoder freezing (end-to-end fine-tuning)

### 3.4 Evaluation Protocol

- **Trials**: 10,000 (5,000 positive, 5,000 negative)
- **Scoring**: Cosine similarity
- **Metrics**: EER, accuracy, minDCF, ROC curve

---

## 4. Results

_[Refer to tables/RESULTS_SUMMARY.md for detailed metrics]_

### 4.1 Main Findings

| Metric                | Value  |
| --------------------- | ------ |
| Test EER              | 24.90% |
| Validation EER (best) | 6.54%  |
| Accuracy              | 75.10% |
| Optimal Threshold     | 0.1112 |

### 4.2 Training Dynamics

- **Best epoch**: 18/30
- **Overfitting gap**: 18.36 percentage points
- **Training time**: ~8 hours (CPU)

### 4.3 Score Distribution

- Genuine scores: μ=0.25, σ=0.15
- Impostor scores: μ=-0.05, σ=0.12
- Clear separation but moderate overlap

---

## 5. Discussion

### 5.1 Transfer Learning Success

✓ Pretrained English model successfully adapted to Hindi/Kannada
✓ Encoder required full unfreezing (freezing caused 49.94% EER failure)
✓ Final 24.90% EER significantly better than random (50%)

### 5.2 Low-Resource Challenge

✗ 3 files/speaker insufficient for robust generalization
✗ 18.36 pp overfitting gap indicates memorization
✓ Performance within expected range for data constraint

### 5.3 Practical Implications

**For Deployment**:

- Collect 10-20 files/speaker minimum
- Implement PLDA backend
- Add data augmentation

**For Research**:

- Explore few-shot learning
- Test meta-learning approaches
- Compare with TiTANet architecture

---

## 6. Conclusions

This study demonstrates that transfer learning from English speaker models to regional Indian languages (Hindi/Kannada) is viable but requires careful consideration of data limitations. With only 3 training utterances per speaker, we achieved 24.90% EER—reasonable for the constraint but insufficient for production systems. Key findings:

1. **Encoder adaptation is essential**: Freezing caused complete failure
2. **Data quantity matters**: 18.36 pp overfitting gap with 3 files/speaker
3. **Transfer learning works**: 75.10% accuracy vs 50% random baseline

**Future Work**:

- Increase training data to 10-20 files/speaker
- Implement PLDA scoring backend
- Train TiTANet for architecture comparison
- Explore data augmentation strategies
- Test on additional Indian languages

---

## References

[To be completed with proper citations]

1. Desplanques et al. (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.
2. Nagrani et al. (2017). VoxCeleb: A large-scale speaker identification dataset.
3. Snyder et al. (2018). X-vectors: Robust DNN embeddings for speaker recognition.
4. Ko et al. (2015). Audio augmentation for speech recognition.

---

## Appendix

### A. Reproducibility

**Code Repository**: [GitHub link if applicable]

**Dependencies**:

```
Python 3.10.11
PyTorch 2.7.1+cu118
SpeechBrain (latest)
librosa, scikit-learn, matplotlib
```

**Training Command**:

```bash
python train.py --config config/ecapa_config.yaml --model ecapa
```

**Evaluation Command**:

```bash
python evaluate.py --config config/ecapa_config.yaml --checkpoint checkpoints/ecapa/best_model.pt --model ecapa
```

### B. Computational Resources

- **Platform**: Windows 10 (CPU-only)
- **Training Time**: 8 hours (30 epochs)
- **Inference**: ~100 embeddings/second
- **Memory**: ~4 GB RAM peak

### C. Ethical Considerations

- Speaker data anonymized with numeric IDs
- No personally identifiable information
- Intended for authentication, not surveillance
- Potential bias in telephone quality variations
