import os
import traceback
import argparse
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.compliance.kaldi as kaldi
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Global variables for Wav2Vec2 model (loaded once)
processor = None
model = None
device = None

def initialize_wav2vec2_model():
    """Initialize Wav2Vec2 model on GPU - called once"""
    global processor, model, device
    if model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", cache_dir=".cache")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", cache_dir=".cache", output_hidden_states=True)
        model = model.to(device)
        model.eval()
        print("Wav2Vec2 model loaded successfully")


def load_and_resample_audio(wav_path, sr_target=16000):
    """Load audio and resample to target sample rate. Returns 1D torch.Tensor."""
    waveform, sr = torchaudio.load(wav_path)

    # Convert to mono if stereo (expect (channels, time))
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != sr_target:
        waveform = torchaudio.functional.resample(waveform, sr, sr_target)

    # Ensure 1D tensor (time,)
    return waveform.squeeze(0).contiguous(), sr_target


def extract_mfb(audio, sr=16000, target_frames=750):
    """Extract MFB features using Kaldi - target shape (300, 40)"""
    # Convert to tensor for Kaldi
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
    else:
        audio_tensor = audio

    # Ensure 1D (time,) tensor for fbank input
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.view(-1)
    
    # Kaldi fbank parameters tuned for 300 frames from 15s audio
    # 15s * 20Hz = 300 frames, so frame_shift = 15s / 300 = 0.05s = 50ms
    frame_shift_ms = (15.0 / target_frames) * 1000  # Convert to milliseconds
    
    # kaldi.fbank expects waveform with shape (channel, time)
    mfb = kaldi.fbank(
        waveform=audio_tensor.unsqueeze(0),
        num_mel_bins=40,
        sample_frequency=sr,
        frame_length=25.0,
        frame_shift=frame_shift_ms,
        dither=0.0,
        energy_floor=0.0,
        use_energy=False
    )
    
    # Align to exactly target_frames
    mfb = align_frames(mfb, target_frames)
    
    return mfb.numpy()


def extract_mfcc(audio, sr=16000, target_frames=750):
    # Convert to tensor if needed
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
    else:
        audio_tensor = audio

    # Ensure 1D (time,) tensor for transform input
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.view(-1)
        
    hop_length = int(sr * 15.0 / target_frames)
    
    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=40,
        melkwargs={
            "n_fft": 512,
            "win_length": 400,  # 25ms @ 16kHz
            "hop_length": hop_length,
            "n_mels": 128,
            "f_max": 8000,
            "window_fn": torch.hann_window
        }
    )
    
    mfcc = mfcc_transform(audio_tensor.unsqueeze(0)).squeeze(0).T  # shape: (T, 40)
    
    # Align to exactly target_frames
    mfcc = align_frames(mfcc, target_frames)
    
    return mfcc.numpy()


def extract_wav2vec2(audio, sr=16000, target_frames=750, selected_layers=(4, 6, 12)):
    """Extract Wav2Vec2 embeddings - target shape (749, 768) for each layer"""
    initialize_wav2vec2_model()
    
    # Convert to tensor if needed
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
    else:
        audio_tensor = audio

    # Ensure 1D (time,) before feeding to processor
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.view(-1)
    
    # Process with Wav2Vec2
    inputs = processor(
        audio_tensor, 
        sampling_rate=sr, 
        return_tensors="pt", 
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Extract selected layers
        layer_features = {}
        for layer_idx in selected_layers:
            # Get features for this layer
            features = outputs.hidden_states[layer_idx][0].cpu()  # shape: (T, 768)

            # Align to target frames (keep as torch, convert to numpy for saving)
            features = align_frames(features, target_frames)

            layer_features[f'layer{layer_idx}'] = features.numpy()
    
    return layer_features


def align_frames(feature, target_frames):
    """Align feature to target number of frames. Accepts torch.Tensor or np.ndarray, returns torch.Tensor."""
    # Convert to torch for consistent ops
    is_numpy = isinstance(feature, np.ndarray)
    feature_tensor = torch.from_numpy(feature) if is_numpy else feature

    current_frames = feature_tensor.shape[0]

    if current_frames == target_frames:
        return feature_tensor
    if current_frames < target_frames:
        pad_length = target_frames - current_frames
        if feature_tensor.dim() == 1:
            last_frame = feature_tensor[-1:].repeat(pad_length)
            return torch.cat([feature_tensor, last_frame], dim=0)
        else:
            last_frame = feature_tensor[-1:].repeat(pad_length, 1)
            return torch.cat([feature_tensor, last_frame], dim=0)
    # current_frames > target_frames
    return feature_tensor[:target_frames]


def process_single_file(args):
    """Process a single audio file and extract all features"""
    fname, audio_dir, features_dir = args
    
    session_id = os.path.splitext(fname)[0]
    wav_path = os.path.join(audio_dir, fname)
    session_dir = os.path.join(features_dir, session_id)
    wav2vec_dir = os.path.join(session_dir, "wav2vec2")
    
    # Create directories
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(wav2vec_dir, exist_ok=True)
    
    try:
        # Load and resample audio
        audio, sr = load_and_resample_audio(wav_path, sr_target=16000)

        # Verify audio duration (should be ~15s)
        duration = (audio.shape[0] if isinstance(audio, torch.Tensor) else len(audio)) / sr
        if abs(duration - 15.0) > 0.5:  # Allow 0.5s tolerance
            print(f"Warning: {session_id} duration is {duration:.2f}s (expected ~15s)")

        # Extract MFB
        try:
            mfb = extract_mfb(audio, sr, target_frames=750)
            np.save(os.path.join(session_dir, "mfb.npy"), mfb)
        except Exception as sub_e:
            tb = traceback.format_exc(limit=2)
            return f"Failed on {session_id}: MFB error: {sub_e} | audio_shape={tuple(audio.shape) if isinstance(audio, torch.Tensor) else (len(audio),)} | {tb.strip()}"

        # Extract MFCC
        try:
            mfcc = extract_mfcc(audio, sr, target_frames=750)
            np.save(os.path.join(session_dir, "mfcc.npy"), mfcc)
        except Exception as sub_e:
            tb = traceback.format_exc(limit=2)
            return f"Failed on {session_id}: MFCC error: {sub_e} | audio_shape={tuple(audio.shape) if isinstance(audio, torch.Tensor) else (len(audio),)} | {tb.strip()}"

        # Extract Wav2Vec2 embeddings
        try:
            wav2vec_features = extract_wav2vec2(audio, sr, target_frames=750, selected_layers=(4, 6, 12))
            for layer_name, features in wav2vec_features.items():
                np.save(os.path.join(wav2vec_dir, f"wav2vec2_{layer_name}.npy"), features)
        except Exception as sub_e:
            tb = traceback.format_exc(limit=2)
            return f"Failed on {session_id}: Wav2Vec2 error: {sub_e} | audio_shape={tuple(audio.shape) if isinstance(audio, torch.Tensor) else (len(audio),)} | {tb.strip()}"

        return f"Success: {session_id}"

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return f"Failed on {session_id}: {e} | {tb.strip()}"


def extract_features_for_split(split_name, audio_dir, features_dir, only: str | None = None, limit: int | None = None):
    """Extract features for a specific split"""
    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES FOR {split_name.upper()} SET")
    print(f"{'='*60}")
    
    # Get all audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if only:
        audio_files = [f for f in audio_files if os.path.splitext(f)[0] == only]
    if limit is not None:
        audio_files = audio_files[:max(0, int(limit))]
    print(f"Found {len(audio_files)} audio files")
    
    # Process files
    results = []
    for fname in tqdm(audio_files, desc=f"Processing {split_name}"):
        result = process_single_file((fname, audio_dir, features_dir))
        results.append(result)
    
    # Print results summary
    success_count = sum(1 for r in results if r.startswith("Success"))
    failed_count = len(results) - success_count
    
    print(f"\n{split_name.upper()} SET SUMMARY:")
    print(f"  Total files: {len(audio_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")
    
    # Show some failures if any
    failures = [r for r in results if r.startswith("Failed")]
    if failures:
        print(f"  Failures:")
        for failure in failures[:5]:  # Show first 5
            print(f"    {failure}")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more")
    
    return success_count, failed_count


def main():
    """Main function to extract features for all splits"""
    print("FEATURE EXTRACTION FOR NEW SPLIT DATA")
    print("="*60)
    
    parser = argparse.ArgumentParser(description="Extract features for new split data")
    parser.add_argument("--base_dir", type=str, default="/work/users/y/u/yuyuwang/cardio_pause")
    parser.add_argument("--splits", type=str, nargs="*", default=['train', 'val', 'test'], help="Which splits to run")
    parser.add_argument("--only", type=str, default=None, help="Only process a single session id (basename without .wav)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files per split")
    args = parser.parse_args()

    base_dir = args.base_dir
    splits = args.splits
    
    total_success = 0
    total_failed = 0
    
    for split_name in splits:
        # Define paths
        audio_dir = os.path.join(base_dir, f"new_split_{split_name}", "segmented_audio")
        features_dir = os.path.join(base_dir, f"new_split_{split_name}", "features_750")
        
        # Check if audio directory exists
        if not os.path.exists(audio_dir):
            print(f"\nWARNING: Audio directory not found: {audio_dir}")
            continue
        
        # Extract features for this split
        success, failed = extract_features_for_split(split_name, audio_dir, features_dir, only=args.only, limit=args.limit)
        total_success += success
        total_failed += failed
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total successful: {total_success}")
    print(f"Total failed: {total_failed}")
    print(f"Success rate: {(total_success / (total_success + total_failed) * 100):.1f}%")
    
    print(f"\nOutput directories:")
    for split_name in splits:
        print(f"  {base_dir}/new_split_{split_name}/features_750/")


if __name__ == "__main__":
    main()