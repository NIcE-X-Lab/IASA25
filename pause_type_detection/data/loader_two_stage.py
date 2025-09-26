import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from utils.set_seed import seed_worker

def get_session_metadata(label_dir, audio_dir):
    session_metadata = []

    for fname in sorted(os.listdir(label_dir)):
        
        session_id = os.path.splitext(fname)[0]
        participant_id = session_id.split("_")[1]
        speed = session_id.split("_")[2]
        task = session_id.split("_")[5]
        stride = session_id.split("_")[-1]

        wav_path = os.path.join(audio_dir, f"{session_id}.wav")
        if os.path.exists(wav_path):
            duration = torchaudio.info(wav_path).num_frames / torchaudio.info(wav_path).sample_rate
        else:
            duration = 0.0

        session_metadata.append({
            "session": session_id,
            "participant": participant_id,
            "speed": speed,
            "task": task,
            "stride": stride,
            "duration": duration 
        })
    return pd.DataFrame(session_metadata)

class TwoStageDataset(Dataset):
    def __init__(self, metadata_df, feature_dir, label_dir=None, selected_wav2vec2_layers=(12,), expected_label_length=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.selected_wav2vec2_layers = selected_wav2vec2_layers
        self.expected_label_length = expected_label_length    

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        session_id = row["session"]
        session_feature_dir = os.path.join(self.feature_dir, session_id)
        
        # Load all features needed for two-stage model (support both legacy and new lengths)
        # MFB for Stage 1 (break detection)
        mfb_path = os.path.join(session_feature_dir, "mfb.npy")
        if not os.path.exists(mfb_path):
            raise FileNotFoundError(f"MFB file does not exist: {mfb_path}")
        mfb = np.load(mfb_path)  # (T, 40) where T can be 300 or 750
        
        # MFCC for Stage 2
        mfcc_path = os.path.join(session_feature_dir, "mfcc.npy")
        if not os.path.exists(mfcc_path):
            raise FileNotFoundError(f"MFCC file does not exist: {mfcc_path}")
        mfcc = np.load(mfcc_path)  # (T_mfcc, 40) e.g., 750 (new) or 1501 (legacy)
        
        # Wav2Vec2 embeddings for Stage 2
        layer_id = self.selected_wav2vec2_layers[0]
        wav2vec2_path = os.path.join(session_feature_dir, f"wav2vec2/wav2vec2_layer{layer_id}.npy")
        if not os.path.exists(wav2vec2_path):
            raise FileNotFoundError(f"Wav2Vec2 layer file does not exist: {wav2vec2_path}")
        wav2vec2 = np.load(wav2vec2_path)  # (T_embed, 768) where T_embed is 750 in the new setting
        
        # For fairness across experiments, enforce equal time dims (no implicit resampling)
    
        # Load label
        label_path = os.path.join(self.label_dir, f"{session_id}.npy")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file does not exist: {label_path}")
        y = np.load(label_path)  # (T_label,) expected 750

        target_len = int(wav2vec2.shape[0])

        # Strict checks: ensure MFB/MFCC match embedding length (expected 750)
        if mfb.shape[0] != target_len:
            raise ValueError(f"MFB length mismatch for session {session_id}: got {mfb.shape[0]}, expected {target_len}")
        if mfcc.shape[0] != target_len:
            raise ValueError(f"MFCC length mismatch for session {session_id}: got {mfcc.shape[0]}, expected {target_len}")
        # Labels: enforce expected length if specified
        if self.expected_label_length is not None:
            if y.shape[0] != int(self.expected_label_length):
                raise ValueError(f"Label length mismatch for session {session_id}: got {y.shape[0]}, expected {self.expected_label_length}")
        else:
            if y.shape[0] != target_len:
                raise ValueError(f"Label length mismatch for session {session_id}: got {y.shape[0]}, expected {target_len}")

        y = torch.tensor(y, dtype=torch.float32)
        
        # Create break targets (binary: 0=no break, 1=any break)
        break_targets = (y > 0.5).float()  # Any break (s, b, bs) = 1, no break (o) = 0
        
        return (
            mfb, mfcc, wav2vec2, y, break_targets,
            session_id,
            row["participant"], row["speed"], row["task"], row["duration"]
        )

def collate_two_stage_batch(batch):
    def to_tensor_and_pad(seqs):
        tensors = [x.detach().clone().float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32) for x in seqs]
        lengths = [len(x) for x in tensors]
        padded = pad_sequence(tensors, batch_first=True)
        return padded, torch.tensor(lengths, dtype=torch.long)

    # Extract all elements
    batch_size = len(batch)
    mfb_seqs = [b[0] for b in batch]
    mfcc_seqs = [b[1] for b in batch]
    wav2vec2_seqs = [b[2] for b in batch]
    y_seqs = [b[3] for b in batch]
    break_targets_seqs = [b[4] for b in batch]
    session_ids = [b[5] for b in batch]
    participants = [b[6] for b in batch]
    speeds = [b[7] for b in batch]
    tasks = [b[8] for b in batch]
    durations = [b[9] for b in batch]

    # Pad sequences
    mfb_padded, mfb_lengths = to_tensor_and_pad(mfb_seqs)
    mfcc_padded, mfcc_lengths = to_tensor_and_pad(mfcc_seqs)
    wav2vec2_padded, wav2vec2_lengths = to_tensor_and_pad(wav2vec2_seqs)
    y_padded, y_lengths = to_tensor_and_pad(y_seqs)
    break_targets_padded, break_targets_lengths = to_tensor_and_pad(break_targets_seqs)

    return {
        "mfb": mfb_padded,
        "mfcc": mfcc_padded,
        "wav2vec2": wav2vec2_padded,
        "labels": y_padded,
        "break_targets": break_targets_padded,
        "session_ids": session_ids,
        "participants": participants,
        "speeds": speeds,
        "tasks": tasks,
        "durations": durations
    }

def get_dataloader_from_sessions(session_ids, meta_df, feature_dir, label_dir,
                                selected_wav2vec2_layers=(12,), expected_label_length=None,
                                batch_size=4, shuffle=False, num_workers=6, pin_memory=True, generator=None):
    # Filter metadata to only include the specified sessions
    filtered_meta = meta_df[meta_df["session"].isin(session_ids)].reset_index(drop=True)
    
    # Create dataset
    dataset = TwoStageDataset(
        metadata_df=filtered_meta,
        feature_dir=feature_dir,
        label_dir=label_dir,
        selected_wav2vec2_layers=selected_wav2vec2_layers,
        expected_label_length=expected_label_length
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_two_stage_batch,
        worker_init_fn=seed_worker,
        generator=generator
    )
    
    return loader
