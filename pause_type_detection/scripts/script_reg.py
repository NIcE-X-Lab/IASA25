import os
import sys
import yaml
import torch
import multiprocessing
from data.loader import get_session_metadata, get_dataloader_from_sessions
from run_experiment.run_experiment_reg import run_experiment
from utils.set_seed import set_seed
from datetime import date

class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()
        
    def restore(self):
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        self.log.close()


def run_experiment_with_config(config: dict):

    # --- 1. General Setup ---
    model = config["experiment"]["model"]
    num_workers = 0
    param_index = config["experiment"]["param_index"]
    input_features = config["experiment"]["input_features"]
    selected_wav2vec2_layers = tuple(config["experiment"].get("selected_wav2vec2_layers", []))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load parameters
    param_grid = config[f"{model}_param_grid"]
    assert param_grid is not None, f"Parameter grid not found for model: {model}"
    params = param_grid[param_index]
    model_param_dict = {
        "mlp_params": None,
        "lstm_params": None,
        "tcnnlstm_params": None,
        "alexnet_params": None,
        "grunet_params": None,
        "vgg16_params": None
    }
    model_param_dict[f"{model}_params"] = params

    # --- 2. Experiment Name ---
    feature_str = "_".join(input_features)
    layer_str = f"_{'_'.join(map(str, selected_wav2vec2_layers))}" if "embed" in input_features else ""
    date_str = date.today().strftime("%Y-%m-%d")
    experiment_name = f"new_split_reg_{model}_{feature_str}{layer_str}_{date_str}"
    print(f"Experiment name: {experiment_name}")

    # --- 2.5. Training Parameters ---
    batch_size = config["training"]["batch_size"]
    n_epoch     = config["training"]["epochs"]

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
    set_seed(config["training"]["seed"])
    n_epoch     = config["training"]["epochs"]
    batch_size  = config["training"]["batch_size"]

    # --- 5. Training Pipline ---
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

    all_metrics = []
    

    # Run experiments
    for feature_type in input_features:
        print(f"\n=== Running with params: {params}, feature_type: {feature_type} ===")
        
        use_mfcc_flag = (feature_type=="mfcc") or (feature_type=="embed" and model=="vgg16")
        use_mfb_flag = (feature_type=="mfb")
        use_embed_flag = (feature_type=="embed")
        
        # Create data loaders using pre-defined splits
        train_loader = get_dataloader_from_sessions(
            train_sessions, train_meta_df, 
            feature_dir=train_feature_dir, label_dir=train_label_dir, 
            use_mfcc=use_mfcc_flag, 
            use_mfb=use_mfb_flag, 
            use_embed=use_embed_flag, 
            selected_wav2vec2_layers = selected_wav2vec2_layers, 
            batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory=True)
        
        val_loader = get_dataloader_from_sessions(
            val_sessions, val_meta_df, 
            feature_dir=val_feature_dir, label_dir=val_label_dir, 
            use_mfcc=use_mfcc_flag, 
            use_mfb=use_mfb_flag, 
            use_embed=use_embed_flag, 
            selected_wav2vec2_layers = selected_wav2vec2_layers, 
            batch_size=batch_size, shuffle=False, num_workers = num_workers,pin_memory=True)
        
        test_loader = get_dataloader_from_sessions(
            test_sessions, test_meta_df, 
            feature_dir=test_feature_dir, label_dir=test_label_dir, 
            use_mfcc=use_mfcc_flag, 
            use_mfb=use_mfb_flag, 
            use_embed=use_embed_flag, 
            selected_wav2vec2_layers = selected_wav2vec2_layers, 
            batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory=True)

        for batch in train_loader:
            # One-time debug of B, T, D for current feature
            x = batch.get(feature_type, None)
            if x is not None:
                B, T, D = tuple(x.shape)
                print(f"[DEBUG] REG input feature '{feature_type}' first batch shape: (B={B}, T={T}, D={D})")
                input_dim = D
            else:
                print(f"[DEBUG] ERROR: Primary feature '{feature_type}' is None!")
                print(f"[DEBUG] Available features: {[k for k, v in batch.items() if v is not None and k not in ['session_ids', 'participants', 'speeds', 'tasks', 'durations']]}")
                available_features = [k for k, v in batch.items() if v is not None and hasattr(v, 'shape')]
                if available_features:
                    x = batch[available_features[0]]
                    B, T, D = tuple(x.shape)
                    input_dim = D
                    print(f"[DEBUG] REG fallback feature: {available_features[0]}, shape: (B={B}, T={T}, D={D})")
                else:
                    raise ValueError("No valid features found in batch!")
            break
        
        print(f"\n====== Running Experiment ======")
        val_metrics, test_metrics = run_experiment(train_loader, val_loader, test_loader, input_dim=input_dim, epochs=n_epoch, loss_plt_dir=loss_plt_dir, per_file_dir=per_file_dir, metrics_dir=metrics_dir, cm_dir=cm_dir, fold_idx=0, model_type=model, feature_type=feature_type, device=device, **model_param_dict)
        all_metrics.append({"val": val_metrics, "test": test_metrics})


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
