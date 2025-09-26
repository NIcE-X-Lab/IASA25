import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.models_reg import get_model
from models.losses import PenaltyHuberLoss

def train_one_epoch(model, dataloader, criterion, optimizer, device, model_type, feature_type="mfcc"):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        x = batch[feature_type].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        lengths = batch["lengths"]

        if model_type == "mlp":
            x_flat = []
            y_flat = []
            for i in range(x.size(0)):
                seq_len = min(x[i].size(0), y[i].size(0), lengths[i])
                x_flat.append(x[i, :seq_len])
                y_flat.append(y[i, :seq_len])
            x_flat = torch.cat(x_flat, dim=0)
            y_flat = torch.cat(y_flat, dim=0).float()  # Changed to float for regression

            assert x_flat.shape[0] == y_flat.shape[0], f"Flattened size mismatch: x={x_flat.shape}, y={y_flat.shape}"
            
            preds = model(x_flat)         # Shape: (N, 1) for regression
            loss = criterion(preds.squeeze(-1), y_flat)

        elif model_type == "lstm":
            preds = model(x, lengths)  # Shape: (B, T) for regression
            y = y.float()  # Convert to float for regression
            
            # Flatten and apply mask
            y_flat = y.view(-1)
            preds_flat = preds.view(-1)
            mask = y_flat != -100
            loss = criterion(preds_flat[mask], y_flat[mask])
            
        elif model_type == "tcnnlstm":
            preds = model(x, lengths)  # Shape: (B, T) for regression
            y = y.float()  # Convert to float for regression
            
            # Flatten and apply mask
            y_flat = y.view(-1)
            preds_flat = preds.view(-1)
            mask = y_flat != -100
            loss = criterion(preds_flat[mask], y_flat[mask])

        elif model_type == "alexnet":
            x_img = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, T)
            preds = model(x_img)  # Shape: (B, T) for regression
            y = y.float()  # Convert to float for regression
            
            # Flatten and apply mask
            y_flat = y.view(-1)
            preds_flat = preds.view(-1)
            mask = y_flat != -100
            loss = criterion(preds_flat[mask], y_flat[mask])

        elif model_type == "grunet":
            preds = model(x, return_repr=False)  # Shape: (B, T) for regression
            y = y.float()  # Convert to float for regression
            
            # Flatten and apply mask
            y_flat = y.view(-1)
            preds_flat = preds.view(-1)
            mask = y_flat != -100
            loss = criterion(preds_flat[mask], y_flat[mask])
            
        elif model_type == "vgg16":
            if feature_type == "embed":
                # For embeddings, convert to image format and use only embedding input
                embed_input = x  # (B, 749, 768) 
                x_img = embed_input.unsqueeze(1).transpose(2, 3)  # (B, 1, 768, 749)
                preds = model(x_img)  # (B, T)
            else:
                # For MFB/MFCC, use the regular image format
                x_img = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, T)
                preds = model(x_img)  # (B, T)
                
            y = y.float()         # (B, T)
            mask = (y != -100)    # (B, T)
            # Compute loss only on valid predictions
            if mask.sum() > 0:
                loss = criterion(preds[mask], y[mask])
            else:
                loss = torch.tensor(0.0, device=preds.device, requires_grad=True)

        else:
            raise ValueError("Unsupported model type")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, model_type, feature_type="mfcc"):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_sessions = []
    if not hasattr(evaluate, "_printed_output_shape"):
        evaluate._printed_output_shape = False

    with torch.no_grad():
        for batch in dataloader:
            x = batch[feature_type].to(device)
            y = batch["labels"].to(device)
            lengths = batch["lengths"]

            if model_type == "mlp":
                x_flat = []
                y_flat = []
                for i in range(x.size(0)):
                    valid_len = min(x[i].size(0), y[i].size(0), lengths[i])
                    x_flat.append(x[i, :valid_len])
                    y_flat.append(y[i, :valid_len])
                x_flat = torch.cat(x_flat, dim=0)
                y_flat = torch.cat(y_flat, dim=0).float()  # Changed to float for regression

                preds = model(x_flat)
                loss = criterion(preds.squeeze(-1), y_flat)

                preds_used = preds.squeeze(-1)
                y_used = y_flat

                # per-frame, so do per-frame session grouping
                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds_used[start:end].detach().cpu())
                    all_labels.append(y_used[start:end].detach().cpu())
                    all_sessions.append(batch["session_ids"][i])
                    start = end

            elif model_type == "lstm":
                preds = model(x, lengths)  # Shape: (B, T) for regression
                y = y.float()  # Convert to float for regression
                
                # Flatten and apply mask
                y_flat = y.view(-1)
                preds_flat = preds.view(-1)
                mask = y_flat != -100
                loss = criterion(preds_flat[mask], y_flat[mask])

                preds_used = preds_flat[mask]
                y_used = y_flat[mask]

                # per-frame session-wise grouping
                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds_flat[start:end].detach().cpu())
                    all_labels.append(y_flat[start:end].detach().cpu())
                    all_sessions.append(batch["session_ids"][i])
                    start = end

            elif model_type == "tcnnlstm":
                preds = model(x, lengths)  # Shape: (B, T) for regression
                y = y.float()  # Convert to float for regression
                
                # Flatten and apply mask
                y_flat = y.view(-1)
                preds_flat = preds.view(-1)
                mask = y_flat != -100
                loss = criterion(preds_flat[mask], y_flat[mask])

                preds_used = preds_flat[mask]
                y_used = y_flat[mask]

                # per-frame session-wise grouping
                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds_flat[start:end].detach().cpu())
                    all_labels.append(y_flat[start:end].detach().cpu())
                    all_sessions.append(batch["session_ids"][i])
                    start = end

            elif model_type == "alexnet":
                x_img = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, T)
                preds = model(x_img)  # Shape: (B, T) for regression
                y = y.float()  # Convert to float for regression
                
                # Flatten and apply mask
                y_flat = y.view(-1)
                preds_flat = preds.view(-1)
                mask = y_flat != -100
                loss = criterion(preds_flat[mask], y_flat[mask])

                preds_used = preds_flat[mask]
                y_used = y_flat[mask]

                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds_flat[start:end].detach().cpu())
                    all_labels.append(y_flat[start:end].detach().cpu())
                    all_sessions.append(batch["session_ids"][i])
                    start = end
                    
            elif model_type == "grunet":
                preds = model(x, return_repr=False)  # Shape: (B, T) for regression
                y = y.float()  # Convert to float for regression
                
                # Flatten and apply mask
                y_flat = y.view(-1)
                preds_flat = preds.view(-1)
                mask = y_flat != -100
                loss = criterion(preds_flat[mask], y_flat[mask])

                preds_used = preds_flat[mask]
                y_used = y_flat[mask]

                # Frame-level grouping by session
                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds_flat[start:end].detach().cpu())
                    all_labels.append(y_flat[start:end].detach().cpu())
                    all_sessions.append(batch["session_ids"][i])
                    start = end
                    
            elif model_type == "vgg16":
                if feature_type == "embed":
                    # For embeddings, convert to image format and use only embedding input
                    embed_input = x  # (B, 749, 768) 
                    x_img = embed_input.unsqueeze(1).transpose(2, 3)  # (B, 1, 768, 749)
                    preds = model(x_img)  # (B, T)
                else:
                    # For MFB/MFCC, use the regular image format
                    x_img = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, T)
                    preds = model(x_img)  # (B, T)
                # print(f"[DEBUG] Raw preds range: [{preds.min():.4f}, {preds.max():.4f}]")
                
                preds = torch.clamp(preds, min=0.0, max=3.0)
                # print(f"[DEBUG] Clamped preds range: [{preds.min():.4f}, {preds.max():.4f}]")
                
                y = y.float()  # (B, T)
                mask = (y != -100)  # (B, T)
                
                if mask.sum() > 0:
                    loss = criterion(preds[mask], y[mask])
                else:
                    loss = torch.tensor(0.0, device=preds.device)
                

                for i, seq_len in enumerate(lengths):
                    valid_len = mask[i, :seq_len].sum().item()
                    all_preds.append(preds[i, :valid_len].detach().cpu())
                    all_labels.append(y[i, :valid_len].detach().cpu())
                    all_sessions.append(batch["session_ids"][i])
                
            else:
                raise ValueError("Unsupported model type")

            total_loss += loss.item()

            # One-time debug of per-sample output sequence shape (should be (750, ) typically)
            if not evaluate._printed_output_shape and lengths is not None and lengths.numel() > 0:
                valid_len0 = int((y[0, :lengths[0]] != -100).sum().item())
                print(f"[DEBUG][Fusion] model output per-sample shape: ({valid_len0},)")
                evaluate._printed_output_shape = True

    return total_loss / len(dataloader), all_preds, all_labels, all_sessions


def train_model(model_type, model_params, feature_type, input_dim, dataloaders, device, num_epochs=20, early_stopping=True, lr_scheduler=True, log_file=None, criterion=None):
    model_hparams = {k: v for k, v in model_params.items() if k != "lr"}  # Clean out learning rate
    model = get_model(model_type, model_hparams, input_dim=input_dim, output_dim=1).to(device)  # output_dim=1 for regression

    optimizer = torch.optim.Adam(model.parameters(), lr=model_params.get("lr", 1e-4))
    criterion = criterion

    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
            model_type=model_type,
            feature_type=feature_type
        )
        val_loss, val_preds, val_labels, _ = evaluate(
        model,
        dataloaders["val"],
        criterion,
        device,
        model_type=model_type,
        feature_type=feature_type
        )
        

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if log_file:
            log_file.write(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if lr_scheduler:
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != prev_lr:
                msg = f"[LR Scheduler] Epoch {epoch+1}: LR changed from {prev_lr:.6f} → {new_lr:.6f}"
                print(msg)
                if log_file:
                    log_file.write(msg + "\n")


        min_epochs = 10
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch + 1 >= min_epochs:
                    print("Early stopping triggered.")
                    break

    return model, train_losses, val_losses


def plot_training_curves(train_losses, val_losses, loss_plt_dir="results/loss_curves", fold_idx=0):
    os.makedirs(loss_plt_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training & Validation Loss (Fold {fold_idx + 1})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"train_val_loss_fold_{fold_idx + 1}.png"
    plt.savefig(os.path.join(loss_plt_dir, filename))
    plt.close()

    
def run_experiment(train_loader, val_loader, test_loader, input_dim, epochs=10, device='cpu', loss_plt_dir=None, per_file_dir=None, metrics_dir=None, cm_dir=None, fold_idx=0, model_type="mlp", feature_type="mfcc", **model_param_dict):
    
    print("Device:", device)
    model_params = model_param_dict.get(f"{model_type}_params", None)
    
    # criterion = SimpleFocalRegressionLoss(alpha=0.25, gamma=2.0, class_weights={0.0: 1.0, 1.0: 4.0, 2.0: 3.0, 3.0: 2.0}, delta=0.5)
    penalty_weights = {0.0: 1.0, 1.0: 5.0, 2.0: 4.0, 3.0: 2.0}
    criterion = PenaltyHuberLoss(delta=0.5, penalty_weights=penalty_weights)
        
    model, train_losses, val_losses = train_model(
    model_type=model_type,
    feature_type = feature_type,
    model_params=model_params,
    dataloaders={"train": train_loader, "val": val_loader},
    input_dim=input_dim,
    device=device,
    num_epochs=epochs,
    early_stopping=True,
    lr_scheduler=True,
    criterion=criterion
    )
    print("Model on:", next(model.parameters()).device)

    val_loss, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, device, model_type, feature_type)

    # Aggregate simple regression metrics (MAE/MSE/RMSE) without external metrics module
    y_val_true = np.concatenate([t.numpy() if hasattr(t, 'numpy') else np.asarray(t) for t in val_labels])
    y_val_pred = np.concatenate([p.numpy() if hasattr(p, 'numpy') else np.asarray(p) for p in val_preds])
    val_mae = float(np.mean(np.abs(y_val_pred - y_val_true))) if y_val_true.size > 0 else float('nan')
    val_mse = float(np.mean((y_val_pred - y_val_true) ** 2)) if y_val_true.size > 0 else float('nan')
    val_rmse = float(np.sqrt(val_mse)) if y_val_true.size > 0 else float('nan')
    val_metrics = {"mae": val_mae, "mse": val_mse, "rmse": val_rmse}

    test_loss, test_preds, test_labels, _ = evaluate(model, test_loader, criterion, device, model_type, feature_type)

    y_test_true = np.concatenate([t.numpy() if hasattr(t, 'numpy') else np.asarray(t) for t in test_labels])
    y_test_pred = np.concatenate([p.numpy() if hasattr(p, 'numpy') else np.asarray(p) for p in test_preds])
    test_mae = float(np.mean(np.abs(y_test_pred - y_test_true))) if y_test_true.size > 0 else float('nan')
    test_mse = float(np.mean((y_test_pred - y_test_true) ** 2)) if y_test_true.size > 0 else float('nan')
    test_rmse = float(np.sqrt(test_mse)) if y_test_true.size > 0 else float('nan')
    test_metrics = {"mae": test_mae, "mse": test_mse, "rmse": test_rmse}
    

    plot_training_curves(train_losses, val_losses, loss_plt_dir, fold_idx)
    
    return val_metrics, test_metrics



