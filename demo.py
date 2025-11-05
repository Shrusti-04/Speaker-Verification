"""
Demo Script for Speaker Verification
Quick demonstration of the speaker verification system
"""

import torch
import torchaudio
import argparse
from pathlib import Path
from src.models.ecapa_tdnn import ECAPA_TDNN_Wrapper
from src.models.titanet import TiTANet_Wrapper
from src.verification import CosineScorer


def load_audio(audio_path: str, target_sr: int = 8000) -> torch.Tensor:
    """Load and preprocess audio file"""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform


def verify_speaker(
    model_path: str,
    enrollment_audios: list,
    test_audio: str,
    model_type: str = "ecapa",
    device: str = "cuda"
):
    """
    Verify if test audio matches enrolled speaker
    
    Args:
        model_path: Path to trained model checkpoint
        enrollment_audios: List of paths to enrollment audio files
        test_audio: Path to test audio file
        model_type: Type of model ("ecapa" or "titanet")
        device: Device to use
    """
    print("="*70)
    print("SPEAKER VERIFICATION DEMO")
    print("="*70)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"Loading {model_type.upper()} model...")
    if model_type == "ecapa":
        model = ECAPA_TDNN_Wrapper(embedding_dim=192, num_speakers=351)
    else:
        model = TiTANet_Wrapper(embedding_dim=192, num_speakers=351)
    
    # CRITICAL: Load pretrained base model FIRST
    model.load_pretrained(device=str(device))
    
    # THEN load fine-tuned checkpoint
    model.load_checkpoint(model_path, device=str(device))
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load enrollment audios
    print(f"\nEnrolling speaker with {len(enrollment_audios)} audio(s)...")
    enrollment_embeddings = []
    
    for i, audio_path in enumerate(enrollment_audios, 1):
        print(f"  [{i}/{len(enrollment_audios)}] Loading: {Path(audio_path).name}")
        waveform = load_audio(audio_path)
        waveform = waveform.to(device)
        
        with torch.no_grad():
            embedding = model.extract_embedding(waveform)
            enrollment_embeddings.append(embedding)
    
    # Average enrollment embeddings
    enrollment_embedding = torch.stack(enrollment_embeddings).mean(dim=0)
    print("✓ Speaker enrolled")
    
    # Load test audio
    print(f"\nTesting against: {Path(test_audio).name}")
    test_waveform = load_audio(test_audio)
    test_waveform = test_waveform.to(device)
    
    with torch.no_grad():
        test_embedding = model.extract_embedding(test_waveform)
    
    # Compute similarity
    scorer = CosineScorer(normalize=True)
    similarity = scorer.score(enrollment_embedding, test_embedding).item()
    
    # Display results
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    print(f"\nSimilarity Score: {similarity:.4f}")
    print(f"Score Range: [-1.0 to 1.0], Higher means more similar")
    
    # Decision with different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\nDecision at different thresholds:")
    for threshold in thresholds:
        decision = "✓ SAME SPEAKER" if similarity >= threshold else "✗ DIFFERENT SPEAKER"
        print(f"  Threshold {threshold:.1f}: {decision}")
    
    # Recommendation
    print(f"\n" + "-"*70)
    if similarity >= 0.6:
        print("🟢 VERDICT: Strong match - Likely the same speaker")
    elif similarity >= 0.4:
        print("🟡 VERDICT: Moderate match - Possibly the same speaker")
    else:
        print("🔴 VERDICT: Weak match - Likely different speakers")
    
    print("="*70)
    
    return similarity


def batch_verify(
    model_path: str,
    enrollment_dir: str,
    test_dir: str,
    model_type: str = "ecapa",
    device: str = "cuda",
    max_pairs: int = 10
):
    """
    Batch verification of multiple test files
    
    Args:
        model_path: Path to trained model checkpoint
        enrollment_dir: Directory with enrollment audio files
        test_dir: Directory with test audio files
        model_type: Type of model
        device: Device to use
        max_pairs: Maximum number of test files to process
    """
    print("="*70)
    print("BATCH SPEAKER VERIFICATION")
    print("="*70)
    
    # Get enrollment files
    enrollment_files = sorted(Path(enrollment_dir).glob("*.wav"))
    if len(enrollment_files) == 0:
        print(f"❌ No WAV files found in {enrollment_dir}")
        return
    
    print(f"\nEnrollment files: {len(enrollment_files)}")
    for f in enrollment_files:
        print(f"  - {f.name}")
    
    # Get test files
    test_files = sorted(Path(test_dir).glob("*.wav"))[:max_pairs]
    if len(test_files) == 0:
        print(f"❌ No WAV files found in {test_dir}")
        return
    
    print(f"\nTest files: {len(test_files)}")
    
    # Run verification for each test file
    results = []
    for test_file in test_files:
        print(f"\n{'='*70}")
        print(f"Testing: {test_file.name}")
        print(f"{'='*70}")
        
        similarity = verify_speaker(
            model_path,
            [str(f) for f in enrollment_files],
            str(test_file),
            model_type,
            device
        )
        
        results.append((test_file.name, similarity))
    
    # Summary
    print("\n" + "="*70)
    print("BATCH VERIFICATION SUMMARY")
    print("="*70)
    print(f"\n{'Test File':<40} {'Similarity':<15} {'Decision (0.5)'}")
    print("-"*70)
    
    for filename, score in results:
        decision = "MATCH ✓" if score >= 0.5 else "NO MATCH ✗"
        print(f"{filename:<40} {score:.4f}          {decision}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Speaker Verification Demo')
    
    subparsers = parser.add_subparsers(dest='mode', help='Demo mode')
    
    # Single verification
    single_parser = subparsers.add_parser('verify', help='Verify single audio')
    single_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    single_parser.add_argument('--enroll', nargs='+', required=True, help='Enrollment audio files')
    single_parser.add_argument('--test', required=True, help='Test audio file')
    single_parser.add_argument('--model-type', default='ecapa', choices=['ecapa', 'titanet'])
    single_parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    
    # Batch verification
    batch_parser = subparsers.add_parser('batch', help='Batch verify multiple files')
    batch_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    batch_parser.add_argument('--enroll-dir', required=True, help='Enrollment directory')
    batch_parser.add_argument('--test-dir', required=True, help='Test directory')
    batch_parser.add_argument('--model-type', default='ecapa', choices=['ecapa', 'titanet'])
    batch_parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    batch_parser.add_argument('--max-pairs', type=int, default=10, help='Max test files')
    
    args = parser.parse_args()
    
    if args.mode == 'verify':
        verify_speaker(
            args.model,
            args.enroll,
            args.test,
            args.model_type,
            args.device
        )
    elif args.mode == 'batch':
        batch_verify(
            args.model,
            args.enroll_dir,
            args.test_dir,
            args.model_type,
            args.device,
            args.max_pairs
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
