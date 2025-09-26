import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from models.models_cls import get_model
from models.losses import PenaltyCrossEntropyLoss

def train_one_epoch(model, dataloader, criterion, optimizer, device, model_type, feature_type, scaler, accum_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    def compute_micro_loss(x_local, y_local, lengths_local):
        if model_type == "mlp":
            x_flat = []
            y_flat = []
            for i in range(x_local.size(0)):
                seq_len = min(x_local[i].size(0), y_local[i].size(0), lengths_local[i])
                x_flat.append(x_local[i, :seq_len])
                y_flat.append(y_local[i, :seq_len])
            x_flat = torch.cat(x_flat, dim=0)
            y_flat = torch.cat(y_flat, dim=0).long()
            logits = model(x_flat)
            return criterion(logits, y_flat)

        elif model_type == "lstm":
            logits = model(x_local, lengths_local)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y_local.view(-1).long()
            return criterion(logits_flat, y_flat)

        elif model_type == "tcnnlstm":
            logits, _, _, _ = model(x_local, lengths_local)
            Bm, Tm, C = logits.shape
            y_flat = y_local.view(-1).long()
            logits_flat = logits.view(Bm * Tm, C)
            mask = y_flat != -100
            return criterion(logits_flat[mask], y_flat[mask])

        elif model_type == "alexnet":
            x_img = x_local.permute(0, 2, 1).unsqueeze(1)
            logits = model(x_img)
            Bm, Tm, C = logits.shape
            y_flat = y_local.view(-1).long()
            logits_flat = logits.view(Bm * Tm, C)
            mask = y_flat != -100
            return criterion(logits_flat[mask], y_flat[mask])

        elif model_type == "grunet":
            logits = model(x_local, return_repr=False)
            Bm, Tm, C = logits.shape
            y_flat = y_local.view(-1).long()
            logits_flat = logits.view(Bm * Tm, C)
            mask = y_flat != -100
            return criterion(logits_flat[mask], y_flat[mask])

        elif model_type == "vgg16":
            x_img = x_local.permute(0, 2, 1).unsqueeze(1)
            logits = model(x_img)
            Bm, Tm, C = logits.shape
            y_flat = y_local.view(-1).long()
            logits_flat = logits.view(Bm * Tm, C)
            mask = y_flat != -100
            return criterion(logits_flat[mask], y_flat[mask])
        else:
            raise ValueError("Unsupported model type")

    for step_idx, batch in enumerate(dataloader, 1):
        x = batch[feature_type].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        lengths = batch["lengths"]

        # Use micro-batches for memory-heavy combos (VGG16/AlexNet + embed)
        use_micro = (model_type in ("vgg16", "alexnet")) and (feature_type == "embed")
        B = x.size(0)
        micro_bsz = 8 if use_micro else B
        num_micro = (B + micro_bsz - 1) // micro_bsz

        if num_micro > 1:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for start in range(0, B, micro_bsz):
                    end = min(start + micro_bsz, B)
                    loss_micro = compute_micro_loss(x[start:end], y[start:end], lengths[start:end])
                    scaler.scale(loss_micro / num_micro).backward()
                    total_loss += loss_micro.item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = compute_micro_loss(x, y, lengths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, model_type, feature_type):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_sessions = []

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        for batch in dataloader:
            x = batch[feature_type].to(device, non_blocking=True)
            y = batch["labels"].to(device)
            lengths = batch["lengths"]
            # Light micro-batching to avoid eval OOM for VGG16 embed
            use_micro = (model_type in ("vgg16", "alexnet")) and (feature_type == "embed")
            B = x.size(0)
            micro_bsz = 16 if use_micro else B

            if model_type == "mlp":
                x_flat = []
                y_flat = []
                for i in range(x.size(0)):
                    valid_len = min(x[i].size(0), y[i].size(0), lengths[i])
                    x_flat.append(x[i, :valid_len])
                    y_flat.append(y[i, :valid_len])
                x_flat = torch.cat(x_flat, dim=0)
                y_flat = torch.cat(y_flat, dim=0).long()

                logits = model(x_flat)
                loss = criterion(logits, y_flat)

                preds = torch.argmax(logits, dim=1)
                y_used = y_flat

                # per-frame, so do per-frame session grouping
                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds[start:end])
                    all_labels.append(y_used[start:end])
                    all_sessions.append(batch["session_ids"][i])
                    start = end

            elif model_type == "lstm":
                logits = model(x, lengths)
                B, T, C = logits.shape
                logits_flat = logits.view(-1, C)
                y_flat = y.view(-1).long()
                loss = criterion(logits_flat, y_flat)

                mask = y_flat != -100
                logits_masked = logits_flat[mask]
                y_masked = y_flat[mask]

                preds = torch.argmax(logits_masked, dim=1)
                y_used = y_masked

                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds[start:end])
                    all_labels.append(y_used[start:end])
                    all_sessions.append(batch["session_ids"][i])
                    start = end

            elif model_type == "tcnnlstm":
                logits, _, _, _ = model(x, lengths)  # Extract only logits from tuple
                B, T, C = logits.shape

                y_flat = y.view(-1).long()
                logits_flat = logits.view(B * T, C)

                mask = y_flat != -100
                logits_masked = logits_flat[mask]
                y_masked = y_flat[mask]

                loss = criterion(logits_masked, y_masked)
                preds = torch.argmax(logits_masked, dim=1)
                y_used = y_masked

                # per-frame session-wise grouping
                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds[start:end])
                    all_labels.append(y_used[start:end])
                    all_sessions.append(batch["session_ids"][i])
                    start = end

            elif model_type == "alexnet":
                x_img = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, T)
                logits = model(x_img)  # (B, T', C)
                B, T, C = logits.shape

                y_flat = y.view(-1).long()
                logits_flat = logits.view(B * T, C)

                mask = y_flat != -100
                logits_masked = logits_flat[mask]
                y_masked = y_flat[mask]

                loss = criterion(logits_masked, y_masked)
                preds = torch.argmax(logits_masked, dim=1)
                y_used = y_masked

                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds[start:end])
                    all_labels.append(y_used[start:end])
                    all_sessions.append(batch["session_ids"][i])
                    start = end
                    
            elif model_type == "grunet":
                logits = model(x, return_repr=False)  # [B, T, C]
                B, T, C = logits.shape

                y_flat = y.view(-1).long()
                logits_flat = logits.view(B * T, C)

                mask = y_flat != -100
                logits_masked = logits_flat[mask]
                y_masked = y_flat[mask]

                loss = criterion(logits_masked, y_masked)

                preds = torch.argmax(logits_masked, dim=1)
                y_used = y_masked

                # Frame-level grouping by session
                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds[start:end])
                    all_labels.append(y_used[start:end])
                    all_sessions.append(batch["session_ids"][i])
                    start = end
                    
            elif model_type == "vgg16":
                preds_cat = []
                for s in range(0, B, micro_bsz):
                    e = min(s + micro_bsz, B)
                    x_img = x[s:e].permute(0, 2, 1).unsqueeze(1)
                    preds_cat.append(model(x_img))
                logits = torch.cat(preds_cat, dim=0)
                B2, T2, C2 = logits.shape

                y_flat = y.view(-1).long()
                logits_flat = logits.view(B2 * T2, C2)

                mask = y_flat != -100
                logits_masked = logits_flat[mask]
                y_masked = y_flat[mask]

                loss = criterion(logits_masked, y_masked)
                preds = torch.argmax(logits_masked, dim=1)
                y_used = y_masked

                start = 0
                for i, seq_len in enumerate(lengths):
                    valid_len = (y[i, :seq_len] != -100).sum().item()
                    end = start + valid_len
                    all_preds.append(preds[start:end])
                    all_labels.append(y_used[start:end])
                    all_sessions.append(batch["session_ids"][i])
                    start = end

            else:
                raise ValueError("Unsupported model type")

            total_loss += loss.item()

    return total_loss / len(dataloader), all_preds, all_labels, all_sessions


def train_model(model_type, model_params, feature_type, input_dim, dataloaders, device, num_epochs=20, early_stopping=True, lr_scheduler=True, log_file=None, criterion=None):
    model_hparams = {k: v for k, v in model_params.items() if k != "lr"}  # Clean out learning rate
    model = get_model(model_type, model_hparams, input_dim=input_dim, output_dim=4).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_params.get("lr", 1e-4))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    accum_steps = 2 if torch.cuda.is_available() else 1  # effectively halves per-step memory
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
            feature_type=feature_type,
            scaler=scaler,
            accum_steps=accum_steps
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

    
def run_experiment(train_loader, val_loader, test_loader, input_dim, epochs=10, device='cpu', loss_plt_dir=None, per_file_dir=None, metrics_dir=None, cm_dir=None, fold_idx=0, model_type="mlp", feature_type=None, **model_param_dict):
    
    print("Device:", device)
    model_params = model_param_dict.get(f"{model_type}_params", None)
    
    penalty_matrix = [
    [0.0, 3.0, 3.0, 3.0],   # true 'o'
    [2.0, 0.0, 5.0, 2.0],   # true 's' → misclassifying as 'b' is bad (5.0)
    [2.0, 5.0, 0.0, 2.0],   # true 'b' → misclassifying as 's' is bad (5.0)
    [2.0, 2.0, 2.0, 0.0]    # true 'bs'
    ]
    weight_tensor = torch.tensor([0.24, 6.24, 4.65, 0.91], dtype=torch.float32).to(device)

    criterion = PenaltyCrossEntropyLoss(penalty_matrix, weight=weight_tensor)

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

    # Evaluate on validation set to compute final val metrics
    val_loss, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, device, model_type, feature_type)
    y_val_true = torch.cat(val_labels).cpu().numpy()
    y_val_pred = torch.cat(val_preds).cpu().numpy()
    val_acc = accuracy_score(y_val_true, y_val_pred)
    try:
        y_val_true_1hot = torch.nn.functional.one_hot(torch.tensor(y_val_true), num_classes=4)
        y_val_pred_1hot = torch.nn.functional.one_hot(torch.tensor(y_val_pred), num_classes=4)
        val_auc = roc_auc_score(y_val_true_1hot, y_val_pred_1hot, multi_class="ovo", average="macro")
    except Exception:
        val_auc = float("nan")
    print(f"[VAL] Fold {fold_idx+1} | Acc: {val_acc:.3f} | AUC: {val_auc:.3f}")

    test_loss, test_preds, test_labels, _ = evaluate(model, test_loader, criterion, device, model_type, feature_type)
    y_test_true = torch.cat(test_labels).cpu().numpy()
    y_test_pred = torch.cat(test_preds).cpu().numpy()
    accuracy = accuracy_score(y_test_true, y_test_pred)
    try:
        y_test_true_1hot = torch.nn.functional.one_hot(torch.tensor(y_test_true), num_classes=4)
        y_test_pred_1hot = torch.nn.functional.one_hot(torch.tensor(y_test_pred), num_classes=4)
        auc = roc_auc_score(y_test_true_1hot, y_test_pred_1hot, multi_class="ovo", average="macro")
    except Exception:
        auc = float("nan")
    print(f"[TEST] Fold {fold_idx+1} | Acc: {accuracy:.3f} | AUC: {auc:.3f}")

    plot_training_curves(train_losses, val_losses, loss_plt_dir, fold_idx)

    return {
        "accuracy": accuracy,
        "auc": auc,
        "y_true": y_test_true,
        "y_pred": y_test_pred,
        "test_per_file_acc": None
    }
