import os
import argparse
import numpy as np
import pandas as pd

# ======== Config / Mapping ========
LABEL_TO_ID = {"o": 0, "s": 1, "b": 2, "bs": 3}  # priority: bs > b > s > o
DEFAULT_SR_HZ = 50
TAIL_SECONDS = 15.0

# ======== Helpers ========
def normalize_token(tok: str) -> str:
    """Keep only o/s/b/bs; everything else → 'o'."""
    tok = (tok or "").strip().lower()
    return tok if tok in LABEL_TO_ID else "o"

def read_intervals(txt_path: str):
    """
    Read 'start end label' per line (whitespace/CSV tolerant).
    Returns list of (start, end, token) with token normalized.
    """
    intervals = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = [p for p in ln.replace(",", " ").split() if p]
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0]); end = float(parts[1])
            except ValueError:
                continue
            if end <= start:
                continue
            token = normalize_token(parts[2])
            intervals.append((start, end, token))
    return intervals

def crop_intervals(intervals, t0, t1):
    """Intersect intervals with [t0, t1); shift not applied here."""
    out = []
    for (s, e, tok) in intervals:
        if e <= t0 or s >= t1:
            continue
        cs = max(s, t0)
        ce = min(e, t1)
        if ce > cs:
            out.append((cs, ce, tok))
    return out

def intervals_to_frame_labels(intervals, t0, t1, sr_hz=DEFAULT_SR_HZ):
    """
    Convert *cropped* intervals to frame-wise labels at sr_hz over [t0, t1).
    Uses [start,end) and priority bs(3) > b(2) > s(1) > o(0).
    """
    duration = max(0.0, t1 - t0)
    n_frames = int(round(duration * sr_hz))
    labels = np.zeros(n_frames, dtype=np.int16)  # default 'o'

    if n_frames == 0 or not intervals:
        return labels

    for (s, e, tok) in intervals:
        lab_id = LABEL_TO_ID[tok]
        # localize to segment start t0, and map to indices with floor
        start_idx = int(np.floor((s - t0) * sr_hz))
        end_idx   = int(np.floor((e - t0) * sr_hz))
        # clamp to [0, n_frames]
        start_idx = max(0, min(n_frames, start_idx))
        end_idx   = max(0, min(n_frames, end_idx))
        if end_idx <= start_idx:
            continue
        cur = labels[start_idx:end_idx]
        mask = lab_id > cur
        if mask.any():
            cur[mask] = lab_id
            labels[start_idx:end_idx] = cur
    return labels

def count_unique_txt(src_label_dir: str):
    txts = {os.path.splitext(f)[0] for f in os.listdir(src_label_dir) if f.endswith(".txt")}
    print(f"[Count] Unique .txt labels (ignoring .txt.bak): {len(txts)} in {src_label_dir}")
    return txts

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def get_time_window(row, segtype: str, intervals):
    """
    Decide [t0, t1) using CSV fields (prefer split_time when present).
    Fallbacks:
      - full_file: [0, end_time or max_end]
      - first_part: [0, split_time] else [0, (end_time-15)]
      - last_15s: [split_time, split_time+15] else [max_end-15, max_end]
    """
    # Helpful CSV fields if present
    split_time = row["split_time"] if "split_time" in row and pd.notna(row["split_time"]) else None
    start_time = row["start_time"] if "start_time" in row and pd.notna(row["start_time"]) else 0.0
    end_time   = row["end_time"]   if "end_time"   in row and pd.notna(row["end_time"])   else None
    duration   = row["duration"]   if "duration"   in row and pd.notna(row["duration"])   else None

    max_end = max((e for (_, e, _) in intervals), default=0.0)
    # If file has explicit end_time use it; else infer from intervals
    clip_end = float(end_time) if end_time is not None else max_end

    if segtype == "full_file":
        t0 = float(start_time) if start_time is not None else 0.0
        t1 = clip_end if clip_end > t0 else max_end
        return (t0, t1)

    if segtype == "first_part":
        if split_time is not None:
            t0, t1 = 0.0, float(split_time)
        else:
            # fallback: end_time - 15s
            tail_start = (float(end_time) - TAIL_SECONDS) if end_time is not None else (max_end - TAIL_SECONDS)
            t0, t1 = 0.0, max(0.0, tail_start)
        return (t0, t1)

    if segtype == "last_15s":
        if split_time is not None:
            t0, t1 = float(split_time), float(split_time) + TAIL_SECONDS
        else:
            # fallback: last 15s to end
            t1 = clip_end if end_time is not None else max_end
            t0 = max(0.0, t1 - TAIL_SECONDS)
        return (t0, t1)

    raise ValueError(f"Unknown segment_type: {segtype}")

def save_label_npy(out_dir: str, base: str, segment_type: str, arr: np.ndarray) -> str:
    ensure_dir(out_dir)
    # Keep a transparent name showing which part it is
    out_name = f"{base}.npy" if segment_type == "full_file" else f"{base}_{segment_type}.npy"
    out_path = os.path.join(out_dir, out_name)
    np.save(out_path, arr.astype(np.int16, copy=False))
    return out_path

def process_split(csv_path: str, src_label_dir: str, dest_split_dir: str,
                  sr_hz: int = DEFAULT_SR_HZ):
    """
    For each row in CSV:
      - find <base>.txt in src_label_dir
      - split in time FIRST using CSV window (segment_type + split_time)
      - convert cropped window to frame-wise labels at sr_hz
      - save to <dest_split_dir>/labels/
    """
    print(f"\n[Split] Processing: {csv_path}")
    df = pd.read_csv(csv_path)
    needed = {"filename", "segment_type"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{csv_path} must at least contain columns: {needed}")

    out_labels_dir = os.path.join(dest_split_dir, "labels")
    ensure_dir(out_labels_dir)

    saved = 0
    missing_src = 0
    empty_after_crop = 0

    for _, row in df.iterrows():
        fname = str(row["filename"])
        segtype = str(row["segment_type"])
        base = os.path.splitext(fname)[0]
        src_txt = os.path.join(src_label_dir, base + ".txt")
        if not os.path.exists(src_txt):
            print(f"[Missing] {src_txt}")
            missing_src += 1
            continue

        try:
            intervals = read_intervals(src_txt)
            t0, t1 = get_time_window(row, segtype, intervals)
            if t1 <= t0:
                print(f"[Skip] {base} ({segtype}) -> invalid window [{t0}, {t1}).")
                empty_after_crop += 1
                continue
            cropped = crop_intervals(intervals, t0, t1)
            labels = intervals_to_frame_labels(cropped, t0, t1, sr_hz=sr_hz)
            if labels.size == 0:
                print(f"[Skip] {base} ({segtype}) -> 0 frames after crop.")
                empty_after_crop += 1
                continue
        except Exception as e:
            print(f"[Error] {base} ({segtype}): {e}")
            missing_src += 1
            continue

        out_path = save_label_npy(out_labels_dir, base, segtype, labels)
        print(f"[OK] {base} ({segtype}) [{t0:.3f}, {t1:.3f}) -> {out_path} (len={len(labels)})")
        saved += 1

    print(f"\n[{os.path.basename(csv_path)}] Summary:")
    print(f"  Saved:              {saved}")
    print(f"  Missing source:     {missing_src}")
    print(f"  Empty/invalid crop: {empty_after_crop}")

# ======== CLI / Entrypoint ========
def main():
    ap = argparse.ArgumentParser(
        description="Split sessions by CSV times first, then convert cropped labels to frame-wise NPY (50 Hz, o/s/b/bs)."
    )
    ap.add_argument("--src_label_dir", required=True, help="Folder with original .txt labels (and .txt.bak)")
    ap.add_argument("--train_csv", required=True, help="Full path to train CSV")
    ap.add_argument("--val_csv", required=True, help="Full path to val CSV")
    ap.add_argument("--test_csv", required=True, help="Full path to test CSV")
    ap.add_argument("--train_out", required=True, help="Train split root (we create train_out/labels/)")
    ap.add_argument("--val_out", required=True, help="Val split root (we create val_out/labels/)")
    ap.add_argument("--test_out", required=True, help="Test split root (we create test_out/labels/)")
    ap.add_argument("--sr_hz", type=int, default=DEFAULT_SR_HZ, help="Label sample rate (Hz). Default 50")
    args = ap.parse_args()

    # 1) Count unique .txts
    count_unique_txt(args.src_label_dir)

    # 2) Process splits (split first, then rasterize)
    process_split(args.train_csv, args.src_label_dir, args.train_out, sr_hz=args.sr_hz)
    process_split(args.val_csv,   args.src_label_dir, args.val_out,   sr_hz=args.sr_hz)
    process_split(args.test_csv,  args.src_label_dir, args.test_out,  sr_hz=args.sr_hz)

if __name__ == "__main__":
    main()
    
