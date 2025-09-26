import os
import sys
import yaml
import torch
import multiprocessing
import matplotlib.pyplot as plt
from utils.set_seed import set_seed
from datetime import date
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from data.loader_two_stage import get_session_metadata, get_dataloader_from_sessions
from models.models_cls import TwoStageCNNLSTM, TwoStageGRU
from models.losses import TwoStageClassificationLoss

from metrics import (
    group_preds_by_session,
    compute_per_file_accuracy,
    save_per_file_accuracy,
    save_confusion_matrix,
    apply_temporal_smoothing,
    merge_nearby_segments,
    compute_tIoU,
    compute_segment_precision_recall,
)

class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def restore(self):
        sys.stdout = self.terminal


def run_experiment_with_config(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment settings
    model = config["experiment"]["model"]
    selected_wav2vec2_layers = config["experiment"]["selected_wav2vec2_layers"]
    label_length = int(config["experiment"].get("label_length", 750))
    fused_feature = str(config["experiment"].get("fused_feature", "mfb")).lower()

    layer_str = f"layer{'_'.join(map(str, selected_wav2vec2_layers))}"
    date_str = date.today().strftime("%Y-%m-%d")
    experiment_name = f"new_split_cls_two_stage_{model}_{fused_feature}_{layer_str}_{date_str}"
    print(f"Experiment name: {experiment_name}")

    # Paths
    base_dir = config["paths"]["base_dir"]
    train_audio_dir = f"{base_dir}/new_split_train/segmented_audio"
    train_label_dir = f"{base_dir}/new_split_train/segmented_labels"
    train_feature_dir = f"{base_dir}/new_split_train/features"
    val_audio_dir = f"{base_dir}/new_split_val/segmented_audio"
    val_label_dir = f"{base_dir}/new_split_val/segmented_labels"
    val_feature_dir = f"{base_dir}/new_split_val/features"
    test_audio_dir = f"{base_dir}/new_split_test/segmented_audio"
    test_label_dir = f"{base_dir}/new_split_test/segmented_labels"
    test_feature_dir = f"{base_dir}/new_split_test/features"

    cm_dir = f"{base_dir}/experiments_750_750/{experiment_name}/cm"
    loss_plt_dir = f"{base_dir}/experiments_750_750/{experiment_name}/loss"
    per_file_dir = f"{base_dir}/experiments_750_750/{experiment_name}/per_file_acc"
    metrics_dir = f"{base_dir}/experiments_750_750/{experiment_name}/metrics"
    config_dir = f"{base_dir}/experiments_750_750/{experiment_name}/config"
    for d in [cm_dir, loss_plt_dir, per_file_dir, metrics_dir, config_dir]:
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(config_dir, "experiment_name.txt"), "w") as f:
        f.write(experiment_name + "\n")
    with open(os.path.join(config_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    log_terminal_path = os.path.join(config_dir, "terminal_log.txt")
    sys.stdout = sys.stderr = TeeLogger(log_terminal_path)

    # Training config
    seed = config["training"]["seed"]
    set_seed(seed)
    n_epoch = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    early_stopping = config["training"]["early_stopping"]
    lr_scheduler = config["training"]["lr_scheduler"]

    torch.backends.cudnn.benchmark = False
    num_workers = 2

    # Metadata
    print("Loading metadata from all splits...")
    train_meta_df = get_session_metadata(train_label_dir, train_audio_dir)
    val_meta_df = get_session_metadata(val_label_dir, val_audio_dir)
    test_meta_df = get_session_metadata(test_label_dir, test_audio_dir)
    print(f"Train set: {len(train_meta_df)} files")
    print(f"Val set: {len(val_meta_df)} files")
    print(f"Test set: {len(test_meta_df)} files")

    def validate_files(meta_df, feature_dir, label_dir, split_name):
        missing_files = []
        for _, row in meta_df.iterrows():
            session_id = row["session"]
            feature_path = os.path.join(feature_dir, session_id)
            label_path = os.path.join(label_dir, f"{session_id}.npy")
            if not os.path.exists(feature_path):
                missing_files.append(f"Feature dir: {feature_path}")
            if not os.path.exists(label_path):
                missing_files.append(f"Label file: {label_path}")
        if missing_files:
            print(f"{split_name} - Missing files:")
            for f in missing_files[:10]:
                print(f"  {f}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
            return False
        else:
            print(f"{split_name} - All files exist")
            return True

    train_valid = validate_files(train_meta_df, train_feature_dir, train_label_dir, "Train")
    val_valid = validate_files(val_meta_df, val_feature_dir, val_label_dir, "Val")
    test_valid = validate_files(test_meta_df, test_feature_dir, test_label_dir, "Test")
    if not (train_valid and val_valid and test_valid):
        print("Validation failed. Please check missing files.")
        return

    train_sessions = train_meta_df["session"].tolist()
    val_sessions = val_meta_df["session"].tolist()
    test_sessions = test_meta_df["session"].tolist()
    print(f"Using pre-defined splits:")
    print(f"  Train: {len(train_sessions)} sessions")
    print(f"  Val: {len(val_sessions)} sessions")
    print(f"  Test: {len(test_sessions)} sessions")

    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_loader = get_dataloader_from_sessions(
        train_sessions, train_meta_df,
        feature_dir=train_feature_dir, label_dir=train_label_dir,
        selected_wav2vec2_layers=selected_wav2vec2_layers,
        expected_label_length=label_length,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator=generator)
    val_loader = get_dataloader_from_sessions(
        val_sessions, val_meta_df,
        feature_dir=val_feature_dir, label_dir=val_label_dir,
        selected_wav2vec2_layers=selected_wav2vec2_layers,
        expected_label_length=label_length,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = get_dataloader_from_sessions(
        test_sessions, test_meta_df,
        feature_dir=test_feature_dir, label_dir=test_label_dir,
        selected_wav2vec2_layers=selected_wav2vec2_layers,
        expected_label_length=label_length,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    model_key = str(model).lower().replace('-', '_')
    if model_key in ("two_stage_cnnlstm", "cnnlstm"):
        grid = config.get("cnnlstm_param_grid", [{}])[0]
        model = TwoStageCNNLSTM(
            hidden_dim_stage2=grid.get("hidden_dim", 128),
            target_length=label_length,
            num_layers_stage2=grid.get("num_layers", 2),
            bidirectional_stage2=grid.get("bidirectional", True),
            kernel_size_conv=grid.get("kernel_size_conv", 3),
            padding_conv=grid.get("padding_conv", 1),
            stride_conv=grid.get("stride_conv", 1),
            dilation_conv=grid.get("dilation_conv", 1),
            bias_conv=grid.get("bias_conv", True),
            groups_conv=grid.get("groups_conv", 1),
            fused_feature=fused_feature,
            num_classes=4
        )
        model_params = {"lr": grid.get("lr", 1e-4)}
    elif model_key in ("two_stage_gru", "gru"):
        grid = config.get("gru_param_grid", [{}])[0]
        model = TwoStageGRU(
            hidden_dim_stage2=grid.get("hidden_dim", 128),
            target_length=label_length,
            num_layers_stage2=grid.get("num_layers", 1),
            bidirectional_stage2=grid.get("bidirectional", True),
            use_attention_stage2=grid.get("use_attention", False),
            attention_dim_stage2=grid.get("attention_dim", None),
            fused_feature=fused_feature,
            num_classes=4
        )
        model_params = {"lr": grid.get("lr", 1e-4)}
    else:
        raise ValueError(f"Unsupported two-stage model: {model}")

    criterion = TwoStageClassificationLoss(
        break_loss_weight=config["loss"]["break_loss_weight"],
        cls_loss_weight=config["loss"].get("cls_loss_weight", 1.0),
        class_weights=config["loss"].get("class_weights")
    )

    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params.get("lr", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) if lr_scheduler else None

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config["training"]["patience"]

    print("Starting training (classification)...")
    for epoch in range(n_epoch):
        model.train()
        total, num_batches = 0.0, 0
        for batch in train_loader:
            mfb = batch["mfb"].to(device)
            mfcc = batch["mfcc"].to(device)
            wav2vec2 = batch["wav2vec2"].to(device)
            labels = batch["labels"].to(device).long()
            break_targets = (labels > 0).float()  # any non-zero class => break

            optimizer.zero_grad()
            break_probs, logits = model(mfb, wav2vec2, mfcc)
            loss, bce, ce = criterion(break_probs, logits, break_targets, labels)
            loss.backward()
            optimizer.step()

            total += loss.item(); num_batches += 1
        avg_train = total / max(1, num_batches)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        total_v, count_v = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                mfb = batch["mfb"].to(device)
                mfcc = batch["mfcc"].to(device)
                wav2vec2 = batch["wav2vec2"].to(device)
                labels = batch["labels"].to(device).long()
                break_targets = (labels > 0).float()
                break_probs, logits = model(mfb, wav2vec2, mfcc)
                loss, _, _ = criterion(break_probs, logits, break_targets, labels)
                total_v += loss.item(); count_v += 1
        avg_val = total_v / max(1, count_v)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f}")

        if lr_scheduler and scheduler is not None:
            scheduler.step(avg_val)

        if early_stopping:
            if avg_val < best_val_loss:
                best_val_loss = avg_val; patience_counter = 0
                torch.save(model.state_dict(), os.path.join(config_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch + 1 >= 10:
                    print("Early stopping triggered.")
                    break

    if early_stopping and os.path.exists(os.path.join(config_dir, "best_model.pth")):
        model.load_state_dict(torch.load(os.path.join(config_dir, "best_model.pth")))
        print("Loaded best model for evaluation.")

    # Plot training curves
    os.makedirs(loss_plt_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='x')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Two-Stage CLS Train/Val Loss"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(loss_plt_dir, "two_stage_cls_train_val_loss.png")); plt.close()

    def eval_two_stage(dataloader):
        total_loss = 0.0
        preds_all, labels_all, sessions_all = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                mfb = batch["mfb"].to(device)
                mfcc = batch["mfcc"].to(device)
                wav2vec2 = batch["wav2vec2"].to(device)
                labels = batch["labels"].to(device).long()
                break_targets = (labels > 0).float()
                break_probs, logits = model(mfb, wav2vec2, mfcc)
                loss, _, _ = criterion(break_probs, logits, break_targets, labels)
                total_loss += loss.item()
                pred_cls = torch.argmax(logits, dim=-1)
                for i in range(pred_cls.size(0)):
                    preds_all.append(pred_cls[i].cpu())
                    labels_all.append(labels[i].cpu())
                    sessions_all.append(batch["session_ids"][i])
        return total_loss / max(1, len(dataloader)), preds_all, labels_all, sessions_all

    val_loss, val_preds, val_labels, val_sessions = eval_two_stage(val_loader)
    test_loss, test_preds, test_labels, test_sessions = eval_two_stage(test_loader)

    # Prepare raw and smooth predictions
    def to_list(tensors):
        return [t.numpy().tolist() for t in tensors]

    val_preds_raw = to_list([p for p in val_preds])
    val_labels_raw = to_list([t for t in val_labels])
    test_preds_raw = to_list([p for p in test_preds])
    test_labels_raw = to_list([t for t in test_labels])

    def smooth_preds(preds_list):
        smoothed = []
        for pred in preds_list:
            smooth = apply_temporal_smoothing(pred, window=5)
            merged = merge_nearby_segments(smooth, frame_rate=20, max_gap_sec=0.3)
            smoothed.append(torch.tensor(merged))
        return smoothed

    val_preds_smooth = smooth_preds(val_preds_raw)
    test_preds_smooth = smooth_preds(test_preds_raw)

    # Group by session and save per-file accuracy
    val_grouped_raw = group_preds_by_session(val_preds, val_labels, val_sessions)
    val_grouped_smooth = group_preds_by_session(val_preds_smooth, val_labels, val_sessions)
    test_grouped_raw = group_preds_by_session(test_preds, test_labels, test_sessions)
    test_grouped_smooth = group_preds_by_session(test_preds_smooth, test_labels, test_sessions)

    save_per_file_accuracy(compute_per_file_accuracy(val_grouped_raw), os.path.join(per_file_dir, "raw_val_fold_1.csv"))
    save_per_file_accuracy(compute_per_file_accuracy(val_grouped_smooth), os.path.join(per_file_dir, "smooth_val_fold_1.csv"))
    save_per_file_accuracy(compute_per_file_accuracy(test_grouped_raw), os.path.join(per_file_dir, "raw_test_fold_1.csv"))
    save_per_file_accuracy(compute_per_file_accuracy(test_grouped_smooth), os.path.join(per_file_dir, "smooth_test_fold_1.csv"))

    # Confusion matrices and reports
    cls_names = ['o', 's', 'b', 'bs']
    def save_confmats_and_reports(prefix, preds_seq, labels_seq, sessions):
        y_true = torch.cat(labels_seq).cpu().numpy()
        y_pred = torch.cat(preds_seq).cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        try:
            y_true_1hot = torch.nn.functional.one_hot(torch.tensor(y_true), num_classes=4)
            y_pred_1hot = torch.nn.functional.one_hot(torch.tensor(y_pred), num_classes=4)
            auc = roc_auc_score(y_true_1hot, y_pred_1hot, multi_class="ovo", average="macro")
        except:
            auc = float("nan")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        cm_path = os.path.join(cm_dir, f"{prefix}_confusion_matrix_fold_1.png")
        save_confusion_matrix(cm, labels=["o", "b", "s", "bs"], output_path=cm_path, title=f"{prefix.upper()} Confusion Matrix")

        # Segment-level metrics
        labels_list = [t.numpy().tolist() for t in labels_seq]
        preds_list = [p.numpy().tolist() for p in preds_seq]
        report = {}
        for cls_idx in [1, 2, 3]:
            all_tiou, all_prec, all_rec = [], [], []
            for y_t, y_p in zip(labels_list, preds_list):
                tiou = compute_tIoU(y_t, y_p, cls=cls_idx)
                prec, rec = compute_segment_precision_recall(y_t, y_p, cls=cls_idx)
                all_tiou.append(tiou); all_prec.append(prec); all_rec.append(rec)
            report[f"segment_tiou_{cls_names[cls_idx]}"] = float(np.mean(all_tiou))
            report[f"segment_precision_{cls_names[cls_idx]}"] = float(np.mean(all_prec))
            report[f"segment_recall_{cls_names[cls_idx]}"] = float(np.mean(all_rec))

        # Save segment-level predictions
        from metrics import save_segment_level_predictions
        save_segment_level_predictions(sessions, preds_seq, labels_seq, output_path=os.path.join(metrics_dir, f"{prefix}_segment_preds_fold_1.csv"))

        # Save JSON report
        import json
        report.update({"accuracy": float(acc), "auc": float(auc)})
        with open(os.path.join(metrics_dir, f"classification_report_{prefix}_fold_1.json"), "w") as f:
            json.dump(report, f, indent=2)

    save_confmats_and_reports("raw_val", val_preds, val_labels, val_sessions)
    save_confmats_and_reports("smooth_val", val_preds_smooth, val_labels, val_sessions)
    save_confmats_and_reports("raw_test", test_preds, test_labels, test_sessions)
    save_confmats_and_reports("smooth_test", test_preds_smooth, test_labels, test_sessions)

    torch.save(model.state_dict(), os.path.join(config_dir, "final_model.pth"))
    print("Classification training completed! Results saved in:", config_dir)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        with open(config_path) as f:
            config = yaml.safe_load(f)
        run_experiment_with_config(config)
    else:
        raise ValueError("This script expects a config path when run directly.")


