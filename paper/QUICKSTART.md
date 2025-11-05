# Quick Start Guide

## Setup (5 minutes)

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Verify your data structure:**

```bash
python -c "from pathlib import Path; print('Train speakers:', len(list(Path('data/Train').iterdir()))); print('Test speakers:', len(list(Path('data/Test').iterdir())))"
```

## Training (2-4 hours depending on hardware)

### Option 1: Train ECAPA-TDNN

```bash
python train.py --config config/ecapa_config.yaml --model ecapa
```

### Option 2: Train TiTANet

```bash
python train.py --config config/titanet_config.yaml --model titanet
```

### Quick Test (1 epoch)

Edit config file and set `num_epochs: 1` for testing.

## Evaluation (15-30 minutes)

### Evaluate Single Model

```bash
python evaluate.py \
    --config config/ecapa_config.yaml \
    --checkpoint checkpoints/ecapa/best_model.pt \
    --model ecapa \
    --scorer cosine
```

### Compare Both Models

```bash
python compare_models.py \
    --ecapa-checkpoint checkpoints/ecapa/best_model.pt \
    --titanet-checkpoint checkpoints/titanet/best_model.pt
```

## Results

Check the `results/` directory for:

- `*_results.txt` - Detailed metrics
- `*_roc_curve.png` - ROC curves
- `*_tsne.png` - Speaker embedding visualizations
- `comparison/` - Side-by-side comparison

## Troubleshooting

### GPU Memory Issues

Reduce batch size in config:

```yaml
training:
  batch_size: 16 # or 8
```

### Slow Loading

Increase workers:

```yaml
hardware:
  num_workers: 8 # adjust based on CPU cores
```

### Import Errors

```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%\src  # Windows
```

## Expected Timeline

- **Setup**: 5-10 minutes
- **Training ECAPA-TDNN**: 2-3 hours (50 epochs, GPU)
- **Training TiTANet**: 3-4 hours (50 epochs, GPU)
- **Evaluation**: 15-30 minutes per model
- **Total**: ~6-8 hours for complete pipeline

## Quick Commands Reference

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Count parameters
python -c "from src.models.ecapa_tdnn import ECAPA_TDNN_Wrapper; m=ECAPA_TDNN_Wrapper(); print(sum(p.numel() for p in m.parameters()))"

# Test data loading
python src/dataset.py

# Test augmentation
python src/augmentation.py

# Test feature extraction
python src/features.py
```

## Next Steps

1. ✅ Train both models
2. ✅ Evaluate on test set
3. ✅ Compare performance
4. ✅ Generate visualizations
5. ✅ Fine-tune hyperparameters if needed
6. ✅ Test on custom audio samples

## Support

For issues:

1. Check troubleshooting section
2. Review error messages carefully
3. Verify data paths in config files
4. Check GPU memory with `nvidia-smi`

Happy training! 🚀
