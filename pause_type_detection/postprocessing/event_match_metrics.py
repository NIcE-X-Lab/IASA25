import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import glob
import json
from typing import List, Dict, Tuple

def _find_runs(values, target_predicate):
    """
    Find contiguous runs where target_predicate(values[i]) is True.
    Returns list of (start_index, end_index_exclusive).
    """
    runs = []
    n = len(values)
    i = 0
    while i < n:
        if not target_predicate(values[i]):
            i += 1
            continue
        j = i + 1
        while j < n and target_predicate(values[j]):
            j += 1
        runs.append((i, j))
        i = j
    return runs

def _apply_min_event_length(seq, min_len):
    """Convert non-zero runs shorter than min_len to zeros (in-place)."""
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] == 0:
            i += 1
            continue
        # Non-zero run starting at i
        j = i + 1
        while j < n and seq[j] != 0:
            j += 1
        if (j - i) < min_len:
            seq[i:j] = 0
        i = j

def _bridge_zero_gaps(seq, max_gap):
    """
    Fill zero runs of length <= max_gap when surrounded by the same non-zero label.
    Example: [2,2,0,0,2,2] with max_gap>=2 -> [2,2,2,2,2,2].
    """
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] != 0:
            i += 1
            continue
        # Zero run
        j = i + 1
        while j < n and seq[j] == 0:
            j += 1
        left_label = seq[i - 1] if i - 1 >= 0 else 0
        right_label = seq[j] if j < n else 0
        if left_label != 0 and left_label == right_label and (j - i) <= max_gap:
            seq[i:j] = left_label
        i = j

def _majority_collapse(seq):
    """
    For each contiguous non-zero event, set entire event to the dominant class.
    Tie-breaking: choose higher-value label.
    """
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] == 0:
            i += 1
            continue
        j = i + 1
        while j < n and seq[j] != 0:
            j += 1
        # Majority within seq[i:j]
        window = seq[i:j]
        # Count labels 1..3
        counts = {1: 0, 2: 0, 3: 0}
        for v in window:
            if v in counts:
                counts[v] += 1
        max_count = max(counts.values())
        chosen = max([k for k, v in counts.items() if v == max_count])
        seq[i:j] = chosen
        i = j

def clean_sequence(y_pred, min_event_len=3, gap_bridge=5):
    """
    Clean a predicted label sequence with the following steps:
    1) Minimum event length: convert non-zero runs shorter than min_event_len to 0
    2) Bridge zero gaps of length <= gap_bridge between the same label
    3) Majority collapse within each non-zero event with tie -> lower label
    Returns a new numpy array (int).
    """
    seq = np.array(y_pred, dtype=int).copy()
    if len(seq) == 0:
        return seq

    # Step 1: minimum event length
    _apply_min_event_length(seq, min_event_len)

    # Step 2: bridge zero gaps
    _bridge_zero_gaps(seq, gap_bridge)

    # Step 3: majority collapse
    _majority_collapse(seq)

    return seq.astype(int)

def clean_and_save_csv(pred_csv_file, out_csv_file, min_event_len=3, gap_bridge=5, drop_last_n=None, sr_label=None):
    df_pred = pd.read_csv(pred_csv_file)
    cleaned = []
    for _, row in df_pred.iterrows():
        try:
            y_pred = np.array(ast.literal_eval(row['y_pred']), dtype=int)
        except Exception:
            y_pred = np.array([], dtype=int)
        y_pred_clean = clean_sequence(y_pred, min_event_len=min_event_len, gap_bridge=gap_bridge)
        # Simple truncation: drop last 1 second (sr_label frames) if requested
        n_drop = None
        if drop_last_n is not None:
            n_drop = int(drop_last_n)
        elif sr_label is not None:
            n_drop = int(sr_label)
        if n_drop is not None and n_drop > 0 and len(y_pred_clean) > 0:
            keep_len = max(0, len(y_pred_clean) - n_drop)
            y_pred_clean = y_pred_clean[:keep_len]
        cleaned.append(y_pred_clean.tolist())
    df_pred = df_pred.copy()
    df_pred['y_pred_clean'] = [str(x) for x in cleaned]
    # Also expose under the requested name
    df_pred['y_pred_post_processed'] = df_pred['y_pred_clean']
    os.makedirs(os.path.dirname(out_csv_file), exist_ok=True)
    df_pred.to_csv(out_csv_file, index=False)
    return out_csv_file

def update_report_json(exp_dir: str, cleaned_csv_file: str, report_filename: str = "classification_report_test_fold_1.json"):
    """
    Update the experiment's report JSON by adding a key 'y_pred_post_processed'
    mapping session_id -> cleaned prediction list.
    """
    report_path = os.path.join(exp_dir, "metrics", report_filename)
    try:
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
        else:
            report = {}
    except Exception:
        report = {}

    # Load cleaned predictions
    df = pd.read_csv(cleaned_csv_file)
    mapping = {}
    for _, row in df.iterrows():
        session_id = row.get('session_id')
        try:
            cleaned = ast.literal_eval(row.get('y_pred_clean', '[]'))
        except Exception:
            cleaned = []
        mapping[str(session_id)] = cleaned

    report['y_pred_post_processed'] = mapping

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    return report_path

def plot_all_preds_stacked(
    pred_csv_file,
    cleaned_csv_file,
    output_dir,
    sr_label=20
):
    os.makedirs(output_dir, exist_ok=True)

    df_raw = pd.read_csv(pred_csv_file)
    df_cln = pd.read_csv(cleaned_csv_file)

    # Align by session_id just in case
    # Prefer the post-processed column if present, otherwise fallback to y_pred_clean
    use_col = 'y_pred_post_processed' if 'y_pred_post_processed' in df_cln.columns else 'y_pred_clean'
    df = pd.merge(df_raw, df_cln[['session_id', use_col]], on='session_id', how='inner')

    for _, row in df.iterrows():
        session_id = row['session_id']
        try:
            y_pred = np.array(ast.literal_eval(row['y_pred']))
            y_true = np.array(ast.literal_eval(row['y_true']))
            y_pred_clean = np.array(ast.literal_eval(row[use_col]))

            T_raw = len(y_pred)
            time_raw = np.arange(T_raw) / sr_label
            T_clean = len(y_pred_clean)
            time_clean = np.arange(T_clean) / sr_label

            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False, sharey=True)

            # Top: raw
            axes[0].plot(time_raw, y_pred, drawstyle='steps-mid', color='red', label='Predicted (raw)')
            axes[0].plot(time_raw, y_true[:T_raw], drawstyle='steps-mid', color='blue', linestyle='--', alpha=0.6, label='True')
            axes[0].set_ylabel("Class")
            axes[0].set_title(f"Session: {session_id} (raw vs true)")
            axes[0].set_yticks([0, 1, 2, 3])
            axes[0].set_yticklabels(["o", "s", "b", "bs"])
            axes[0].grid(True)
            axes[0].legend()

            # Bottom: cleaned
            axes[1].plot(time_clean, y_pred_clean, drawstyle='steps-mid', color='green', label='Predicted (cleaned)')
            axes[1].plot(time_clean, y_true[:T_clean], drawstyle='steps-mid', color='blue', linestyle='--', alpha=0.6, label='True')
            axes[1].set_ylabel("Class")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_title("Cleaned vs true")
            axes[1].set_yticks([0, 1, 2, 3])
            axes[1].set_yticklabels(["o", "s", "b", "bs"])
            axes[1].grid(True)
            axes[1].legend()

            output_path = os.path.join(output_dir, f"{session_id}_stacked.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"[Saved] {output_path}")

        except Exception as e:
            print(f"[Error] {session_id}: {e}")

def plot_all_preds(
    pred_csv_file,
    output_dir,
    sr_label=20
):
    os.makedirs(output_dir, exist_ok=True)

    # Load prediction CSV
    df_pred = pd.read_csv(pred_csv_file)

    for idx, row in df_pred.iterrows():
        session_id = row['session_id']
        try:
            y_pred = np.array(ast.literal_eval(row['y_pred']))
            y_true = np.array(ast.literal_eval(row['y_true']))

            time = np.arange(len(y_pred)) / sr_label

            # Plot
            plt.figure(figsize=(12, 3))
            plt.plot(time, y_pred, drawstyle='steps-mid', color='red', label='Predicted')
            plt.plot(time, y_true[:len(y_pred)], drawstyle='steps-mid', color='blue', linestyle='--', alpha=0.6, label='True (CSV)')

            plt.ylabel("Class")
            plt.xlabel("Time (s)")
            plt.title(f"Session: {session_id}")
            plt.yticks([0, 1, 2, 3], ["o", "s", "b", "bs"])
            plt.grid(True)
            plt.legend()

            # Save plot
            output_path = os.path.join(output_dir, f"{session_id}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"[Saved] {output_path}")

        except Exception as e:
            print(f"[Error] {session_id}: {e}")


def process_all_experiments(experiments_base_dir, sr_label=20, save_plots=True, update_json=False):
    """
    Process all experiments in the experiments directory
    """
    # Find all experiment directories
    experiment_dirs = glob.glob(os.path.join(experiments_base_dir, "*"))
    
    for exp_dir in experiment_dirs:
        if not os.path.isdir(exp_dir):
            continue
            
        exp_name = os.path.basename(exp_dir)
        print(f"\n=== Processing experiment: {exp_name} ===")
        
        # Look for the CSV file in metrics directory
        csv_file = os.path.join(exp_dir, "metrics", "raw_test_segment_preds_fold_1.csv")
        
        if not os.path.exists(csv_file):
            print(f"[Skip] CSV file not found: {csv_file}")
            continue
            
        # Create output directory for plots
        output_dir = os.path.join(exp_dir, "metrics", "pred_label_plots")
        
        print(f"[Processing] {csv_file}")
        print(f"[Output] {output_dir}")
        
        try:
            # 1) Clean and save CSV (apply drop-last-second using sr_label)
            cleaned_csv = os.path.join(exp_dir, "metrics", "raw_test_segment_preds_fold_1_cleaned.csv")
            clean_and_save_csv(csv_file, cleaned_csv, min_event_len=3, gap_bridge=5, drop_last_n=sr_label)

            # 2) Optionally update report JSON with cleaned predictions
            if update_json:
                update_report_json(exp_dir, cleaned_csv)

            # 3) Optionally plot stacked (raw vs true, cleaned vs true)
            if save_plots:
                plot_all_preds_stacked(csv_file, cleaned_csv, output_dir, sr_label)

            print(f"[Success] Completed {exp_name}")
        except Exception as e:
            print(f"[Error] Failed to process {exp_name}: {e}")


def _extract_events(seq: np.ndarray) -> List[Dict[str, int]]:
    """
    Extract contiguous non-zero events as dicts with keys:
    {'start': int, 'end': int, 'label': int}
    Indexing is inclusive for end.
    """
    events: List[Dict[str, int]] = []
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] == 0:
            i += 1
            continue
        label = int(seq[i])
        j = i + 1
        while j < n and seq[j] != 0:
            j += 1
        events.append({'start': i, 'end': j - 1, 'label': label})
        i = j
    return events

def _overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    return max(0, right - left + 1)

def _match_events_greedy(true_events: List[Dict[str, int]], pred_events: List[Dict[str, int]], seq_len: int, window_frames: int, overlap_min: float) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """
    Returns two lists of pairs:
    - matches: (t_idx, p_idx, score) where labels match and constraints satisfied
    - near_misses: (t_idx, p_idx, score) where labels differ but overlap constraint satisfied
    score = overlap(T,P)/len(T)
    Greedy one-to-one matching.
    """
    candidates_match: List[Tuple[int, int, float, int]] = []  # (t_idx, p_idx, score, tie)
    candidates_near: List[Tuple[int, int, float, int]] = []
    for ti, T in enumerate(true_events):
        t_start, t_end, t_lab = T['start'], T['end'], T['label']
        win_start = max(0, t_start - window_frames)
        win_end = min(seq_len - 1, t_end + window_frames)
        t_len = t_end - t_start + 1
        for pi, P in enumerate(pred_events):
            p_start, p_end, p_lab = P['start'], P['end'], P['label']
            if p_start < win_start or p_end > win_end:
                continue
            ov = _overlap_len(t_start, t_end, p_start, p_end)
            if t_len <= 0:
                continue
            score = ov / float(t_len)
            if score < overlap_min:
                continue
            tie = abs(t_start - p_start) + abs(t_end - p_end)
            if p_lab == t_lab:
                candidates_match.append((ti, pi, score, tie))
            else:
                candidates_near.append((ti, pi, score, tie))

    # Greedy selection by score desc, then smaller tie
    candidates_match.sort(key=lambda x: (-x[2], x[3]))
    used_t = set()
    used_p = set()
    matches: List[Tuple[int, int, float]] = []
    for ti, pi, score, _ in candidates_match:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        matches.append((ti, pi, score))

    # Near misses on remaining
    candidates_near.sort(key=lambda x: (-x[2], x[3]))
    near_misses: List[Tuple[int, int, float]] = []
    for ti, pi, score, _ in candidates_near:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        near_misses.append((ti, pi, score))

    return matches, near_misses

def _compute_per_class_metrics(all_true_events: List[Dict[str, int]], all_pred_events: List[Dict[str, int]], matches: List[Tuple[int, int, float]]) -> pd.DataFrame:
    labels = [1, 2, 3]
    true_counts = {c: 0 for c in labels}
    pred_counts = {c: 0 for c in labels}
    match_counts = {c: 0 for c in labels}

    for ev in all_true_events:
        if ev['label'] in true_counts:
            true_counts[ev['label']] += 1
    for ev in all_pred_events:
        if ev['label'] in pred_counts:
            pred_counts[ev['label']] += 1
    for ti, pi, _ in matches:
        c = all_true_events[ti]['label']
        if c in match_counts and all_pred_events[pi]['label'] == c:
            match_counts[c] += 1

    rows = []
    for c in labels:
        tp = match_counts[c]
        p = pred_counts[c]
        t = true_counts[c]
        precision = tp / p if p > 0 else 0.0
        recall = tp / t if t > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            'class': c,
            'true_events': t,
            'pred_events': p,
            'matched': tp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    return pd.DataFrame(rows)

def run_event_matching_for_experiment(exp_dir: str, sr_label: int, window_frames: int = 10, overlap_min: float = 0.30, save_plots: bool = True):
    cleaned_csv_file = os.path.join(exp_dir, 'metrics', 'raw_test_segment_preds_fold_1_cleaned.csv')
    if not os.path.exists(cleaned_csv_file):
        raise FileNotFoundError(f"Cleaned CSV not found: {cleaned_csv_file}")

    df = pd.read_csv(cleaned_csv_file)
    plots_dir = os.path.join(exp_dir, 'metrics', 'event_match_plots')
    os.makedirs(plots_dir, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []

    # Aggregate events across sessions for overall metrics
    all_true_events_global: List[Dict[str, int]] = []
    all_pred_events_global: List[Dict[str, int]] = []
    offset_true = 0
    offset_pred = 0

    for _, row in df.iterrows():
        session_id = row['session_id']
        try:
            y_true = np.array(ast.literal_eval(row['y_true']), dtype=int)
        except Exception:
            y_true = np.array([], dtype=int)
        # Prefer y_pred_post_processed
        col = 'y_pred_post_processed' if 'y_pred_post_processed' in df.columns else 'y_pred_clean'
        try:
            y_pred = np.array(ast.literal_eval(row[col]), dtype=int)
        except Exception:
            y_pred = np.array([], dtype=int)

        # Align lengths: compare within available cleaned pred length
        T = len(y_pred)
        if T == 0:
            continue
        y_true_clip = y_true[:T] if len(y_true) >= T else y_true

        true_events = _extract_events(y_true_clip)
        pred_events = _extract_events(y_pred)

        # Record with global indexing to compute global metrics later
        idx_map_true = list(range(offset_true, offset_true + len(true_events)))
        idx_map_pred = list(range(offset_pred, offset_pred + len(pred_events)))
        all_true_events_global.extend(true_events)
        all_pred_events_global.extend(pred_events)

        matches_local, near_misses_local = _match_events_greedy(true_events, pred_events, seq_len=T, window_frames=window_frames, overlap_min=overlap_min)

        # Expand to global indices for metrics
        matches_global = [(idx_map_true[ti], idx_map_pred[pi], sc) for (ti, pi, sc) in matches_local]

        # Fill summary rows: true-side
        matched_true = {ti: (pi, sc) for (ti, pi, sc) in matches_local}
        near_true = {ti: (pi, sc) for (ti, pi, sc) in near_misses_local}
        for ti, ev in enumerate(true_events):
            if ti in matched_true:
                pi, sc = matched_true[ti]
                p_ev = pred_events[pi]
                overlap = _overlap_len(ev['start'], ev['end'], p_ev['start'], p_ev['end'])
                summary_rows.append({
                    'session_id': session_id,
                    'side': 'true',
                    'event_idx': ti,
                    'label': ev['label'],
                    'start': ev['start'],
                    'end': ev['end'],
                    'matched_to': pi,
                    'match_type': 'match',
                    'pred_label': p_ev['label'],
                    'overlap_len': overlap,
                    'overlap_frac_true': sc,
                })
            elif ti in near_true:
                pi, sc = near_true[ti]
                p_ev = pred_events[pi]
                overlap = _overlap_len(ev['start'], ev['end'], p_ev['start'], p_ev['end'])
                summary_rows.append({
                    'session_id': session_id,
                    'side': 'true',
                    'event_idx': ti,
                    'label': ev['label'],
                    'start': ev['start'],
                    'end': ev['end'],
                    'matched_to': pi,
                    'match_type': 'near_miss',
                    'pred_label': p_ev['label'],
                    'overlap_len': overlap,
                    'overlap_frac_true': sc,
                })
            else:
                summary_rows.append({
                    'session_id': session_id,
                    'side': 'true',
                    'event_idx': ti,
                    'label': ev['label'],
                    'start': ev['start'],
                    'end': ev['end'],
                    'matched_to': '',
                    'match_type': 'miss',
                    'pred_label': '',
                    'overlap_len': 0,
                    'overlap_frac_true': 0.0,
                })

        # Pred-side false positives
        matched_pred = {pi for (_, pi, _) in matches_local}
        near_pred = {pi for (_, pi, _) in near_misses_local}
        for pi, ev in enumerate(pred_events):
            if pi in matched_pred or pi in near_pred:
                continue
            summary_rows.append({
                'session_id': session_id,
                'side': 'pred',
                'event_idx': pi,
                'label': ev['label'],
                'start': ev['start'],
                'end': ev['end'],
                'matched_to': '',
                'match_type': 'false_positive',
                'pred_label': ev['label'],
                'overlap_len': 0,
                'overlap_frac_true': 0.0,
            })

        # Plot with highlighted matched pairs if requested
        if save_plots and T > 0:
            time = np.arange(T) / sr_label
            fig, ax = plt.subplots(1, 1, figsize=(12, 3))
            ax.plot(time, y_true_clip, drawstyle='steps-mid', color='blue', linestyle='--', alpha=0.7, label='True')
            ax.plot(time, y_pred, drawstyle='steps-mid', color='green', alpha=0.9, label='Pred post-processed')
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(["o", "s", "b", "bs"])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Class')
            ax.set_title(f'Event match: {session_id}')
            # Shade matched true events
            for ti, pi, _ in matches_local:
                ev = true_events[ti]
                ax.axvspan(ev['start']/sr_label, (ev['end']+1)/sr_label, color='green', alpha=0.15)
            # Optionally shade near misses
            for ti, pi, _ in near_misses_local:
                ev = true_events[ti]
                ax.axvspan(ev['start']/sr_label, (ev['end']+1)/sr_label, color='orange', alpha=0.12)
            ax.grid(True)
            ax.legend()
            out_path = os.path.join(plots_dir, f"{session_id}_matched.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()

        offset_true += len(true_events)
        offset_pred += len(pred_events)

    # Save per-experiment summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(exp_dir, 'metrics', 'event_match_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    # Compute overall metrics per class
    # Using global lists and matches only (exclude near-miss)
    # Rebuild matches_global by iterating summary_df
    matches_global: List[Tuple[int, int, float]] = []
    # We didn't store global indices in summary; recompute from local is complex; for class metrics,
    # aggregate from summary_df directly by counting 'match' rows per class and total counts.
    metrics_rows = []
    for c in [1, 2, 3]:
        true_events_c = len(summary_df[(summary_df.side == 'true') & (summary_df.label == c)])
        pred_events_c = len(summary_df[(summary_df.side == 'pred') & (summary_df.label == c)]) + \
                        len(summary_df[(summary_df.side == 'true') & (summary_df.match_type.isin(['match', 'near_miss'])) & (summary_df.pred_label == c)])
        matched_c = len(summary_df[(summary_df.side == 'true') & (summary_df.match_type == 'match') & (summary_df.label == c)])
        precision = matched_c / pred_events_c if pred_events_c > 0 else 0.0
        recall = matched_c / true_events_c if true_events_c > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics_rows.append({
            'class': c,
            'true_events': true_events_c,
            'pred_events': pred_events_c,
            'matched': matched_c,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join(exp_dir, 'metrics', 'event_match_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Cluster-level metrics: cluster 'sb' = {1,2}, cluster 'bs' = {3}
    def _cluster_of(x: int) -> str:
        return 'sb' if x in (1, 2) else ('bs' if x == 3 else 'o')

    clusters = ['sb', 'bs']
    cluster_rows = []
    for cl in clusters:
        # true events per cluster (all true-side rows of that cluster)
        true_c = len(summary_df[(summary_df.side == 'true') & (summary_df.label.apply(_cluster_of) == cl)])
        # predicted events per cluster = false positives in pred-side of cluster
        #   + matched/near_miss predicted labels of that cluster (listed on true-side rows)
        pred_fp_c = len(summary_df[(summary_df.side == 'pred') & (summary_df.label.apply(_cluster_of) == cl)])
        pred_matched_or_near_c = len(summary_df[(summary_df.side == 'true') & (summary_df.match_type.isin(['match', 'near_miss'])) & (summary_df.pred_label.apply(lambda v: _cluster_of(v) == cl))])
        pred_c = pred_fp_c + pred_matched_or_near_c
        # matched per cluster = true-side rows with match/near_miss where both clusters match
        matched_c = len(summary_df[(summary_df.side == 'true') & (summary_df.match_type.isin(['match', 'near_miss'])) & (summary_df.label.apply(_cluster_of) == cl) & (summary_df.pred_label.apply(lambda v: _cluster_of(v) == cl))])
        precision = matched_c / pred_c if pred_c > 0 else 0.0
        recall = matched_c / true_c if true_c > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        cluster_rows.append({
            'cluster': cl,
            'true_events': true_c,
            'pred_events': pred_c,
            'matched': matched_c,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_metrics_csv_path = os.path.join(exp_dir, 'metrics', 'event_match_metrics_cluster.csv')
    cluster_df.to_csv(cluster_metrics_csv_path, index=False)

    print(f"[Saved] {summary_csv_path}")
    print(f"[Saved] {metrics_csv_path}")
    print(f"[Saved] {cluster_metrics_csv_path}")
    print(f"[Saved plots] {plots_dir}")


def aggregate_class_and_cluster_metrics(experiments_base_dir: str):
    """
    Aggregate per-class and cluster metrics from all new_split_cls_* experiments
    into two CSVs placed at the experiments base directory.
    Returns (class_metrics_csv, cluster_metrics_csv).
    """
    exp_dirs = glob.glob(os.path.join(experiments_base_dir, 'new_split_cls_*'))
    rows_class = []
    rows_cluster = []

    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        m_class = os.path.join(exp_dir, 'metrics', 'event_match_metrics.csv')
        m_cluster = os.path.join(exp_dir, 'metrics', 'event_match_metrics_cluster.csv')
        if os.path.exists(m_class):
            try:
                df_c = pd.read_csv(m_class)
                df_c.insert(0, 'experiment', exp_name)
                rows_class.append(df_c)
            except Exception:
                pass
        if os.path.exists(m_cluster):
            try:
                df_cl = pd.read_csv(m_cluster)
                df_cl.insert(0, 'experiment', exp_name)
                rows_cluster.append(df_cl)
            except Exception:
                pass

    out_class = os.path.join(experiments_base_dir, 'event_match_metrics_all_experiments.csv')
    out_cluster = os.path.join(experiments_base_dir, 'event_match_metrics_cluster_all_experiments.csv')

    if rows_class:
        pd.concat(rows_class, ignore_index=True).to_csv(out_class, index=False)
    else:
        pd.DataFrame(columns=['experiment','class','true_events','pred_events','matched','precision','recall','f1']).to_csv(out_class, index=False)

    if rows_cluster:
        pd.concat(rows_cluster, ignore_index=True).to_csv(out_cluster, index=False)
    else:
        pd.DataFrame(columns=['experiment','cluster','true_events','pred_events','matched','precision','recall','f1']).to_csv(out_cluster, index=False)

    print(f"[Saved] {out_class}")
    print(f"[Saved] {out_cluster}")
    return out_class, out_cluster


# =====================
# Folder-level utilities
# =====================

def ensure_folder_postprocess(folder_path: str, sr_label: int = 50, min_event_len: int = 3, gap_bridge: int = 5, drop_last_n: int = None):
    """
    For CSV files in folder_path:
    - If filename contains 'cls', compute y_pred_post_processed using clean_sequence and drop_last_n (default sr_label) and add columns:
      y_pred_post_processed and y_processed (same as y_pred_post_processed)
    - For all files, ensure a column y_processed exists; if missing, set it equal to y_pred
    Saves edits in-place.
    Expected CSV columns: session_id, y_true, y_pred
    """
    import glob
    csv_paths = sorted(glob.glob(os.path.join(folder_path, '*_test_pred.csv')))
    # Only consider classification/regression experiment files
    csv_paths = [p for p in csv_paths if ('cls' in os.path.basename(p)) or ('reg' in os.path.basename(p))]
    if drop_last_n is None:
        drop_last_n = sr_label
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if 'y_pred' not in df.columns:
            continue

        base = os.path.basename(path)
        is_cls = ('cls' in base)
        is_reg = ('reg' in base)
        if is_cls:
            processed = []
            for _, row in df.iterrows():
                try:
                    y_pred = np.array(ast.literal_eval(row['y_pred']), dtype=float).astype(int)
                except Exception:
                    y_pred = np.array([], dtype=int)
                y_pred_clean = clean_sequence(y_pred, min_event_len=min_event_len, gap_bridge=gap_bridge)
                if drop_last_n and drop_last_n > 0 and len(y_pred_clean) > 0:
                    keep_len = max(0, len(y_pred_clean) - int(drop_last_n))
                    y_pred_clean = y_pred_clean[:keep_len]
                processed.append(y_pred_clean.tolist())
            df['y_pred_post_processed'] = [str(x) for x in processed]
        elif is_reg and 'y_pred_post_processed' in df.columns:
            # For regression files: only drop last N frames from existing y_pred_post_processed
            truncated = []
            for _, row in df.iterrows():
                try:
                    seq = np.array(ast.literal_eval(row['y_pred_post_processed']), dtype=float).astype(int)
                except Exception:
                    seq = np.array([], dtype=int)
                if drop_last_n and drop_last_n > 0 and len(seq) > 0:
                    keep_len = max(0, len(seq) - int(drop_last_n))
                    seq = seq[:keep_len]
                truncated.append(seq.tolist())
            df['y_pred_post_processed'] = [str(x) for x in truncated]
        # For non-cls files, do nothing; they are assumed already processed
        df.to_csv(path, index=False)

def match_and_aggregate_folder(folder_path: str, window_frames_list: List[int], sr_label: int = 50, overlap_min: float = 0.30):
    import glob
    csv_paths = sorted(glob.glob(os.path.join(folder_path, '*_test_pred.csv')))

    def map_cluster(x: int) -> str:
        return 'sb' if x in (1, 2) else ('bs' if x == 3 else 'o')

    for w in window_frames_list:
        # Validate required columns exist
        missing = []
        for p in csv_paths:
            try:
                dft = pd.read_csv(p, nrows=1)
            except Exception:
                missing.append((os.path.basename(p), 'unreadable'))
                continue
            if 'y_true' not in dft.columns:
                missing.append((os.path.basename(p), 'missing y_true'))
            if 'y_pred_post_processed' not in dft.columns:
                missing.append((os.path.basename(p), 'missing y_pred_post_processed'))
        if missing:
            names = ', '.join([f"{n}({r})" for n, r in missing])
            raise ValueError(f"Missing required columns in: {names}")

        # Per-experiment rows
        per_exp_class_rows = []
        per_exp_cluster_rows = []

        for path in csv_paths:
            exp_name = os.path.basename(path)
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if 'y_true' not in df.columns or 'y_pred_post_processed' not in df.columns:
                continue

            # Initialize per-experiment accumulators
            true_counts = {1: 0, 2: 0, 3: 0}
            pred_counts = {1: 0, 2: 0, 3: 0}
            matched_counts = {1: 0, 2: 0, 3: 0}
            true_cluster = {'sb': 0, 'bs': 0}
            pred_cluster = {'sb': 0, 'bs': 0}
            matched_cluster = {'sb': 0, 'bs': 0}

            for _, row in df.iterrows():
                try:
                    y_true = np.array(ast.literal_eval(row['y_true']), dtype=float).astype(int)
                except Exception:
                    y_true = np.array([], dtype=int)
                try:
                    y_proc = np.array(ast.literal_eval(row['y_pred_post_processed']), dtype=float).astype(int)
                except Exception:
                    y_proc = np.array([], dtype=int)

                T = len(y_proc)
                if T == 0:
                    continue
                y_true_clip = y_true[:T] if len(y_true) >= T else y_true

                true_events = _extract_events(y_true_clip)
                pred_events = _extract_events(y_proc)

                for ev in true_events:
                    if ev['label'] in true_counts:
                        true_counts[ev['label']] += 1
                        cl = map_cluster(ev['label'])
                        if cl in true_cluster:
                            true_cluster[cl] += 1
                for ev in pred_events:
                    if ev['label'] in pred_counts:
                        pred_counts[ev['label']] += 1
                        cl = map_cluster(ev['label'])
                        if cl in pred_cluster:
                            pred_cluster[cl] += 1

                matches, _near = _match_events_greedy(true_events, pred_events, seq_len=T, window_frames=w, overlap_min=overlap_min)
                for ti, pi, _score in matches:
                    tl = true_events[ti]['label']
                    pl = pred_events[pi]['label']
                    if tl == pl and tl in matched_counts:
                        matched_counts[tl] += 1
                        cl = map_cluster(tl)
                        if cl in matched_cluster:
                            matched_cluster[cl] += 1

            # Build per-experiment class row (one row containing metrics for all classes)
            row_c = {'experiment': exp_name}
            for c in [1, 2, 3]:
                tp = matched_counts[c]
                p = pred_counts[c]
                t = true_counts[c]
                precision = tp / p if p > 0 else 0.0
                recall = tp / t if t > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                row_c.update({
                    f'true_events_{c}': t,
                    f'pred_events_{c}': p,
                    f'matched_{c}': tp,
                    f'precision_{c}': precision,
                    f'recall_{c}': recall,
                    f'f1_{c}': f1,
                })
            per_exp_class_rows.append(row_c)

            # Build per-experiment cluster row
            row_cl = {'experiment': exp_name}
            for cl in ['sb', 'bs']:
                tp = matched_cluster[cl]
                p = pred_cluster[cl]
                t = true_cluster[cl]
                precision = tp / p if p > 0 else 0.0
                recall = tp / t if t > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                row_cl.update({
                    f'true_events_{cl}': t,
                    f'pred_events_{cl}': p,
                    f'matched_{cl}': tp,
                    f'precision_{cl}': precision,
                    f'recall_{cl}': recall,
                    f'f1_{cl}': f1,
                })
            per_exp_cluster_rows.append(row_cl)

        # Add per-class accuracy (= recall) and overall accuracy across classes
        for row in per_exp_class_rows:
            # Per-class accuracy mirrors recall
            for c in [1, 2, 3]:
                row[f'accuracy_{c}'] = row.get(f'recall_{c}', 0.0)
            true_total = sum(row.get(f'true_events_{c}', 0) for c in [1, 2, 3])
            matched_total = sum(row.get(f'matched_{c}', 0) for c in [1, 2, 3])
            row['overall_accuracy'] = (matched_total / true_total) if true_total > 0 else 0.0

        # Add cluster accuracy and overall accuracy across clusters
        for row in per_exp_cluster_rows:
            for cl in ['sb', 'bs']:
                row[f'accuracy_{cl}'] = row.get(f'recall_{cl}', 0.0)
            t_sum = row.get('true_events_sb', 0) + row.get('true_events_bs', 0)
            m_sum = row.get('matched_sb', 0) + row.get('matched_bs', 0)
            row['overall_accuracy_cluster'] = (m_sum / t_sum) if t_sum > 0 else 0.0

        # Merge class and cluster metrics into one CSV per window
        df_class = pd.DataFrame(per_exp_class_rows)
        df_cluster = pd.DataFrame(per_exp_cluster_rows)
        combined = pd.merge(df_class, df_cluster, on='experiment', how='left', suffixes=('', '_cluster'))

        # Reorder columns: overall_accuracy then overall_accuracy_cluster right after experiment;
        # per-class accuracy_* right after matched_* for each class.
        desired_cols = ['experiment', 'overall_accuracy', 'overall_accuracy_cluster']
        for c in [1, 2, 3]:
            desired_cols += [
                f'true_events_{c}', f'pred_events_{c}', f'matched_{c}',
                f'accuracy_{c}', f'precision_{c}', f'recall_{c}', f'f1_{c}'
            ]
        # Cluster metrics (keep overall cluster accuracy at the end)
        desired_cols += [
            'true_events_sb', 'pred_events_sb', 'matched_sb', 'accuracy_sb', 'precision_sb', 'recall_sb', 'f1_sb',
            'true_events_bs', 'pred_events_bs', 'matched_bs', 'accuracy_bs', 'precision_bs', 'recall_bs', 'f1_bs'
        ]
        ordered = [c for c in desired_cols if c in combined.columns]
        # Append any other columns (if present) to avoid dropping unexpected ones
        others = [c for c in combined.columns if c not in ordered]
        combined = combined[ordered + others]

        out_class = os.path.join(folder_path, f'event_match_metrics_all_window_{w}.csv')
        combined.to_csv(out_class, index=False)

# Process all experiments
if __name__ == "__main__":
    experiments_base_dir = "/work/users/y/u/yuyuwang/cardio_pause/experiments_750_750"
    process_all_experiments(experiments_base_dir, sr_label=50)