import os
import sys
import yaml
import torch
import multiprocessing
import matplotlib.pyplot as plt

from data.loader_two_stage import get_session_metadata, get_dataloader_from_sessions
from models.models_reg import TwoStageCNNLSTM, TwoStageGRU
from losses_two_stage import TwoStageRegressionLoss
from utils.set_seed import set_seed
from datetime import date

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
    # --- 1. General Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 6

    # --- 2. Experiment Name ---
    model = config["experiment"]["model"]
    param_index = config["experiment"]["param_index"]
    selected_wav2vec2_layers = config["experiment"]["selected_wav2vec2_layers"]
    label_length = int(config["experiment"].get("label_length", 750))
    fused_feature = str(config["experiment"].get("fused_feature", "mfb")).lower()

    layer_str = f"layer{'_'.join(map(str, selected_wav2vec2_layers))}"
    date_str = date.today().strftime("%Y-%m-%d")
    experiment_name = f"new_split_reg_{model}_{fused_feature}_{layer_str}_{date_str}"
    print(f"Experiment name: {experiment_name}")

    # --- 3. Paths ---
    base_dir = config["paths"]["base_dir"]
    
    # Define paths for all splits
    train_audio_dir = f"{base_dir}/new_split_train/segmented_audio"
    train_label_dir = f"{base_dir}/new_split_train/segmented_labels"
    train_feature_dir = f"{base_dir}/new_split_train/features"
    
    val_audio_dir = f"{base_dir}/new_split_val/segmented_audio"
    val_label_dir = f"{base_dir}/new_split_val/segmented_labels"
    val_feature_dir = f"{base_dir}/new_split_val/features"
    
    test_audio_dir = f"{base_dir}/new_split_test/segmented_audio"
    test_label_dir = f"{base_dir}/new_split_test/segmented_labels"
    test_feature_dir = f"{base_dir}/new_split_test/features"

    cm_dir         = f"{base_dir}/experiments_750_750/{experiment_name}/cm"
    loss_plt_dir   = f"{base_dir}/experiments_750_750/{experiment_name}/loss"
    per_file_dir   = f"{base_dir}/experiments_750_750/{experiment_name}/per_file_acc"
    metrics_dir    = f"{base_dir}/experiments_750_750/{experiment_name}/metrics"
    config_dir     = f"{base_dir}/experiments_750_750/{experiment_name}/config"
    
    for d in [cm_dir, loss_plt_dir, per_file_dir, metrics_dir, config_dir]:
        os.makedirs(d, exist_ok=True)
        
    with open(os.path.join(config_dir, "experiment_name.txt"), "w") as f:
        f.write(experiment_name + "\n")
    with open(os.path.join(config_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    log_path = os.path.join(config_dir, "training_log.txt")
    log_file = open(log_path, "a")
    log_file.write(f"\n=== Experiment: {experiment_name} ===\n")

    log_terminal_path = os.path.join(config_dir, "terminal_log.txt")
    sys.stdout = sys.stderr = TeeLogger(log_terminal_path)

    # --- 4. Training Config ---
    seed = config["training"]["seed"]
    set_seed(seed)
    n_epoch     = config["training"]["epochs"]
    batch_size  = config["training"]["batch_size"]
    early_stopping = config["training"]["early_stopping"]
    lr_scheduler = config["training"]["lr_scheduler"]

    # Basic reproducibility (without deterministic algorithms)
    torch.backends.cudnn.benchmark = False
    
    # Reduce workers for reproducibility
    num_workers = 2  # Reduced from 6 to minimize non-determinism
    
    print(f"Using seed: {seed} for reproducibility")
    log_file.write(f"Seed: {seed}\n")

    # --- 5. Training Pipeline ---
    # Load metadata from all splits
    print("Loading metadata from all splits...")
    train_meta_df = get_session_metadata(train_label_dir, train_audio_dir)
    val_meta_df = get_session_metadata(val_label_dir, val_audio_dir)
    test_meta_df = get_session_metadata(test_label_dir, test_audio_dir)
    
    print(f"Train set: {len(train_meta_df)} files")
    print(f"Val set: {len(val_meta_df)} files")
    print(f"Test set: {len(test_meta_df)} files")
    
    # Validate that all files exist
    print("Validating file existence...")
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
            for f in missing_files[:10]:  # Show first 10
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

    # Get session lists directly from metadata
    train_sessions = train_meta_df["session"].tolist()
    val_sessions = val_meta_df["session"].tolist()
    test_sessions = test_meta_df["session"].tolist()
    
    print(f"Using pre-defined splits:")
    print(f"  Train: {len(train_sessions)} sessions")
    print(f"  Val: {len(val_sessions)} sessions")
    print(f"  Test: {len(test_sessions)} sessions")

    # Create data loaders with reproducible shuffling
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

    # Get input dimensions from first batch
    for batch in train_loader:
        mfb_dim = batch["mfb"].shape[-1]  # 40
        mfcc_dim = batch["mfcc"].shape[-1]  # 40
        wav2vec2_dim = batch["wav2vec2"].shape[-1]  # 768
        break

    print(f"Feature dimensions: MFB={mfb_dim}, MFCC={mfcc_dim}, Wav2Vec2={wav2vec2_dim}")
    
    print(f"\n====== Running Two-Stage Experiment ======")
    
    # Create model and loss
    if model == "two_stage_cnnlstm":
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
            fused_feature=fused_feature
        )
        model_params = {"lr": grid.get("lr", 1e-4)}
    elif model == "two_stage_gru":
        grid = config.get("gru_param_grid", [{}])[0]
        model = TwoStageGRU(
            hidden_dim_stage2=grid.get("hidden_dim", 128),
            target_length=label_length,
            num_layers_stage2=grid.get("num_layers", 1),
            bidirectional_stage2=grid.get("bidirectional", True),
            use_attention_stage2=grid.get("use_attention", False),
            attention_dim_stage2=grid.get("attention_dim", None),
            fused_feature=fused_feature
        )
        model_params = {"lr": grid.get("lr", 1e-4)}
    else:
        raise ValueError(f"Unsupported two-stage model: {model}")
    criterion = TwoStageRegressionLoss(
        break_loss_weight=config["loss"]["break_loss_weight"],
        regression_loss_weight=config["loss"]["regression_loss_weight"],
        focal_alpha=config["loss"]["focal_alpha"],
        focal_gamma=config["loss"]["focal_gamma"],
        class_weights={0.0: 1.0, 1.0: 4.0, 2.0: 3.0, 3.0: 2.0},
        delta=0.5
    )
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params.get("lr", 1e-4))
    
    # ReduceLROnPlateau scheduler (validation-loss-based)
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = None
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config["training"]["patience"]
    
    print("Starting training...")
    for epoch in range(n_epoch):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            mfb = batch["mfb"].to(device)
            mfcc = batch["mfcc"].to(device)
            wav2vec2 = batch["wav2vec2"].to(device)
            labels = batch["labels"].to(device)
            break_targets = batch["break_targets"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            break_probs, regression_output = model(mfb, wav2vec2, mfcc)
            
            # Calculate loss
            loss, break_loss, regression_loss = criterion(
                break_probs, regression_output, break_targets, labels
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f} "
                      f"(Break: {break_loss.item():.4f}, Reg: {regression_loss.item():.4f})")
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                mfb = batch["mfb"].to(device)
                mfcc = batch["mfcc"].to(device)
                wav2vec2 = batch["wav2vec2"].to(device)
                labels = batch["labels"].to(device)
                break_targets = batch["break_targets"].to(device)
                
                # Forward pass
                break_probs, regression_output = model(mfb, wav2vec2, mfcc)
                
                # Calculate loss
                loss, _, _ = criterion(break_probs, regression_output, break_targets, labels)
                
                val_loss += loss.item()
                val_num_batches += 1
        
        avg_val_loss = val_loss / val_num_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling (ReduceLROnPlateau)
        if lr_scheduler and scheduler is not None:
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != prev_lr:
                print(f"[LR Scheduler] Epoch {epoch+1}: LR reduced from {prev_lr:.6f} to {new_lr:.6f}")
        
        # Early stopping
        if early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(config_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch + 1 >= 10:
                    print("Early stopping triggered.")
                    break
    
    # Load best model for evaluation
    if early_stopping and os.path.exists(os.path.join(config_dir, "best_model.pth")):
        model.load_state_dict(torch.load(os.path.join(config_dir, "best_model.pth")))
        print("Loaded best model for evaluation.")
    
    # Plot training curves
    def plot_training_curves(train_losses, val_losses, loss_plt_dir, fold_idx=0):
        os.makedirs(loss_plt_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train Loss", marker='o')
        plt.plot(val_losses, label="Val Loss", marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Two-Stage Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        filename = f"two_stage_train_val_loss.png"
        plt.savefig(os.path.join(loss_plt_dir, filename))
        plt.close()
    
    plot_training_curves(train_losses, val_losses, loss_plt_dir)
    
    # Evaluation
    print("\n====== Evaluation ======")
    
    # Validation evaluation
    model.eval()
    val_preds = []
    val_labels = []
    val_sessions = []
    
    with torch.no_grad():
        for batch in val_loader:
            mfb = batch["mfb"].to(device)
            mfcc = batch["mfcc"].to(device)
            wav2vec2 = batch["wav2vec2"].to(device)
            labels = batch["labels"].to(device)
            session_ids = batch["session_ids"]
            
            # Forward pass
            break_probs, regression_output = model(mfb, wav2vec2, mfcc)
            
            # Collect predictions and labels
            for i in range(regression_output.size(0)):
                val_preds.append(regression_output[i].cpu().numpy())
                val_labels.append(labels[i].cpu().numpy())
                val_sessions.append(session_ids[i])
    
    # Test evaluation
    test_preds = []
    test_labels = []
    test_sessions = []
    
    with torch.no_grad():
        for batch in test_loader:
            mfb = batch["mfb"].to(device)
            mfcc = batch["mfcc"].to(device)
            wav2vec2 = batch["wav2vec2"].to(device)
            labels = batch["labels"].to(device)
            session_ids = batch["session_ids"]
            
            # Forward pass
            break_probs, regression_output = model(mfb, wav2vec2, mfcc)
            
            # Collect predictions and labels
            for i in range(regression_output.size(0)):
                test_preds.append(regression_output[i].cpu().numpy())
                test_labels.append(labels[i].cpu().numpy())
                test_sessions.append(session_ids[i])
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config_dir, "final_model.pth"))
    
    print("Training completed! Results saved in:", config_dir)

    # Save final messages
    log_file.write("Experiment done\n")
    log_file.close()

    # Restore sys.stdout/sys.stderr safely
    if isinstance(sys.stdout, TeeLogger):
        sys.stdout.restore()
        
        
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    import yaml
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        with open(config_path) as f:
            config = yaml.safe_load(f)
        run_experiment_with_config(config)
    else:
        raise ValueError("This script expects a config path when run directly. Use from another script for in-memory configs.")
