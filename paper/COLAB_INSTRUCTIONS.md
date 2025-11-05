# Google Colab Notebook for ECAPA-TDNN Training

# Copy this entire content to a new Colab notebook

## Cell 1: Setup and Installation

```python
# Install packages (this may take 2-3 minutes)
!pip install -q speechbrain==1.0.0
!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q pyyaml scikit-learn matplotlib seaborn tqdm

# Verify installation
import speechbrain
print(f"SpeechBrain version: {speechbrain.__version__}")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch version: {torch.__version__}")
```

## Cell 2: Upload Project Files

```python
# Option A: Upload ZIP file (recommended for first time)
# 1. On your laptop, zip the REU2 folder
# 2. Click Files tab on left -> Upload button
# 3. Upload REU2.zip
# 4. Run this:
!unzip -q REU2.zip
%cd REU2

# OR Option B: Use Google Drive (if already uploaded there)
# !cp -r '/content/drive/MyDrive/REU2' /content/
# %cd REU2

# Verify files
!ls -la
```

## Cell 3: Update Config for GPU

```python
# Modify config to use GPU and adjust epochs
import yaml

config_path = 'config/ecapa_balanced_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update for GPU
config['hardware']['device'] = 'cuda'
config['hardware']['num_workers'] = 2  # Colab supports workers
config['hardware']['pin_memory'] = True
config['hardware']['mixed_precision'] = True  # Faster on GPU

# Change epochs to 15 for Colab's 12-hour limit
config['training']['num_epochs'] = 15

with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("Config updated for GPU training")
print(f"Device: {config['hardware']['device']}")
print(f"Epochs: {config['training']['num_epochs']}")
```

## Cell 4: Start Training

```python
# Train with balanced data
!python train.py --config config/ecapa_balanced_config.yaml --model ecapa

# Training will take ~6-12 hours on GPU
# Colab free tier: 12-hour session limit (should be enough for 15-20 epochs)
```

## Cell 5: Save Results to Google Drive

```python
# Copy checkpoints and logs to Drive (so they're not lost if session ends)
!mkdir -p '/content/drive/MyDrive/REU2_Results'
!cp -r checkpoints/ecapa_balanced '/content/drive/MyDrive/REU2_Results/'
!cp -r logs/ecapa_balanced '/content/drive/MyDrive/REU2_Results/'
!cp -r results/* '/content/drive/MyDrive/REU2_Results/'

print("Results saved to Google Drive!")
```

## Cell 6: Evaluate Model

```python
# Run evaluation
!python evaluate.py --config config/ecapa_balanced_config.yaml \
                    --checkpoint checkpoints/ecapa_balanced/best_model.pt \
                    --model ecapa

# View results
!cat results/ecapa_results.txt
```

## Cell 7: Download Results

```python
# Download specific files to your laptop
from google.colab import files

# Download results
files.download('results/ecapa_results.txt')
files.download('results/ecapa_roc_curve.png')
files.download('results/ecapa_score_distribution.png')
files.download('results/ecapa_tsne.png')
files.download('checkpoints/ecapa_balanced/best_model.pt')
files.download('logs/ecapa_balanced/training_history.png')
```

---

## Tips for Colab:

1. **GPU Selection**: Runtime -> Change runtime type -> GPU (T4 is free)
2. **Session Limit**: Free tier = 12 hours max (enough for 15-20 epochs)
3. **Keep Session Alive**:
   - Click in notebook occasionally
   - Or use:
     ```python
     from google.colab import output
     output.enable_custom_widget_manager()
     ```
4. **Save Frequently**: Copy checkpoints to Drive every few epochs
5. **Resume Training**: If disconnected, can resume from last checkpoint

## Advantages over Local:

- ✅ Free GPU (T4: ~10-15x faster than your CPU)
- ✅ 15-20 epochs in 6-12 hours (vs 5-6 days locally)
- ✅ No need for college server access
- ✅ Can monitor from anywhere
