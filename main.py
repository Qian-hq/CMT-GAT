"""
CMT-GAT: A Contrastive Learning Framework for Predicting Lane Change Manoeuvres
of Surrounding Vehicles in Weaving Areas under Missing Data Conditions

Main training script for the CMT-GAT model.
This script handles data loading, model initialization, training, and validation.
"""

import time
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import csv

# Import model components
import models.Contrastive_trend_at as cmtgat_model
from models.loss import loss_cross_entropy, forcast_value


# =============================================================================
# Hyperparameters (configurable)
# =============================================================================
SEED = 24                           # Random seed for reproducibility
NUM_EPOCHS = 50                     # Total training epochs
BATCH_SIZE = 64                     # Mini-batch size
LEARNING_RATE = 1e-4                # Initial learning rate
STEP_SIZE = 10                      # Learning rate decay step (epochs)
GAMMA = 0.2                         # Learning rate decay factor
CONTRASTIVE_WEIGHT = 0.1            # Weight for contrastive loss term (alpha)
DROPOUT_RATE = 0.1                  # Dropout rate in the network
MASK_LENGTH = '8'                   # Missing data mask length: {'0','2','4','6','8','10'}
                                    #   '0' = no missing, '8' = 80% missing, etc.
TRAIN_RATIO = 0.8                   # Train/val split ratio
DEVICE = None                       # Set automatically: cuda if available, else cpu


def set_seed(seed):
    """Set random seed for reproducibility across numpy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir, mask_length):
    """
    Load training and validation datasets from preprocessed .npy files.

    Args:
        data_dir: Path to the directory containing data_x*.npy files.
        mask_length: String identifier for the missing-data version
                     ('0' = complete, '2' = 20%, ..., '8' = 80% missing).

    Returns:
        X_train, X_val: Masked trajectory features (train / val splits).
        X_all_train, X_all_val: Original (unmasked) trajectory features.
        X_env_train, X_env_val: Environmental context features.
        Y_train, Y_val: Lane-change intention labels (one-hot encoded).

    Data shapes:
        - data_x / data_x_{mask}: (N_samples, T, N_features, D_dim) — trajectory tensors
        - data_env:               (N_samples, env_features)          — context vector
        - data_y_1:               (N_samples, T, num_classes)       — labels
    """
    data_x_orig = np.load(os.path.join(data_dir, 'data_x.npy'), allow_pickle=True)
    data_x_masked = np.load(os.path.join(data_dir, f'data_x_{mask_length}.npy'), allow_pickle=True)
    data_env = np.load(os.path.join(data_dir, 'data_env.npy'), allow_pickle=True)
    data_y = np.load(os.path.join(data_dir, 'data_y_1.npy'), allow_pickle=True)

    # Replace NaN with zeros (missing-value placeholder)
    data_x_orig = np.nan_to_num(data_x_orig, nan=0)
    data_x_masked = np.nan_to_num(data_x_masked, nan=0)

    # Temporal split: first TRAIN_RATIO fraction for training, rest for validation
    split_idx = int(len(data_y) * TRAIN_RATIO)

    return (
        data_x_masked[:split_idx],      data_x_masked[split_idx:],       # X masked
        data_x_orig[:split_idx],        data_x_orig[split_idx:],         # X original
        data_env[:split_idx],           data_env[split_idx:],            # env
        data_y[:split_idx],             data_y[split_idx:]               # labels
    )


def train_epoch(model, train_loader, device, contrastive_weight):
    """
    Execute one training epoch over the entire training set.

    The combined loss is:
        L_total = L_prediction + alpha * L_contrastive

    where L_prediction is cross-entropy loss on lane-change classification,
    and L_contrastive is MSE distance between representations of the masked
    and original trajectories (contrastive regularization).

    Args:
        model: CMT-GAT model instance.
        train_loader: DataLoader yielding (X_masked, X_original, env, Y).
        device: torch.device for computation.
        contrastive_weight: Scalar weight (alpha) for the contrastive term.

    Returns:
        avg_loss: Mean total loss over all batches.
        acc_per_class: Per-class accuracy counts (array of shape [num_classes]).
    """
    model.train()
    total_loss = 0.0
    total_acc = np.zeros(10)

    for batch_X, batch_X_all, batch_X_env, batch_Y in train_loader:
        batch_X = batch_X.to(device)
        batch_X_all = batch_X_all.to(device)
        batch_X_env = batch_X_env.to(device)
        batch_Y = batch_Y.to(device)

        # Forward pass: z_i (masked rep), z_j (original rep), output (predictions)
        z_i, z_j, output = model(batch_X, batch_X_all, batch_X_env)

        # Contrastive loss: MSE between paired representations
        con_loss = nn.MSELoss()(z_i, z_j)

        # Prediction loss: cross-entropy per time step
        pred_loss = loss_cross_entropy(output, batch_Y)
        acc_counts, _ = forcast_value(output, batch_Y)

        # Combined objective
        loss = pred_loss + contrastive_weight * con_loss

        # Backward pass and optimizer step
        model.zero_grad()
        loss.backward()
        model.optimizer.step()

        total_loss += loss.item()
        total_acc += acc_counts

    return total_loss / len(train_loader), total_acc


@torch.no_grad()
def validate(model, val_loader, device):
    """
    Evaluate the model on the validation set without gradient computation.

    Returns:
        avg_loss: Mean prediction loss (cross-entropy only).
        acc_per_class: Per-class accuracy counts.
    """
    model.eval()
    total_loss = 0.0
    total_acc = np.zeros(10)

    for batch_X, batch_X_all, batch_X_env, batch_Y in val_loader:
        batch_X = batch_X.to(device)
        batch_X_all = batch_X_all.to(device)
        batch_X_env = batch_X_env.to(device)
        batch_Y = batch_Y.to(device)

        _, _, output = model(batch_X, batch_X_all, batch_X_env)
        val_loss = loss_cross_entropy(output, batch_Y)
        val_acc_counts, _ = forcast_value(output, batch_Y)

        total_loss += val_loss.item()
        total_acc += val_acc_counts

    return total_loss / len(val_loader), total_acc


def train(model, train_loader, val_loader, Y_train, Y_val,
          num_epochs, contrastive_weight, output_dir, run_tag):
    """
    Full training loop with early stopping based on validation loss.

    Saves training metrics to a CSV file after each epoch and checkpoints
    the best model (lowest validation loss) to disk.

    Args:
        model: Initialized CMT-GAT model (with optimizer and scheduler attached).
        train_loader, val_loader: Training and validation DataLoaders.
        Y_train, Y_val: Label tensors (for computing accuracy rates).
        num_epochs: Maximum number of training epochs.
        contrastive_weight: Alpha weight for contrastive loss.
        output_dir: Directory to save outputs.
        run_tag: Identifier string for file naming (e.g., 'CMT-GAT_m8').

    Returns:
        best_state_dict: Model state dict with lowest validation loss.
    """
    csv_path = os.path.join(output_dir, f'{run_tag}_training_log.csv')
    model_path = os.path.join(output_dir, f'{run_tag}_best_model.pth')

    # Initialize CSV log header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Epoch', 'Train_Loss', 'Val_Loss',
            'Train_Acc', 'Val_Acc', 'Train_Acc_Mean', 'Val_Acc_Mean'
        ])

    best_val_loss = float('inf')
    best_state = None
    t_start = time.time()

    print(f"{'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>9} | "
          f"{'Train Acc':>10} | {'Val Acc':>9}")
    print("-" * 65)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, DEVICE, contrastive_weight)
        val_loss, val_acc = validate(model, val_loader, DEVICE)

        # Compute per-class accuracy rates
        train_acc_rate = train_acc / len(Y_train)
        val_acc_rate = val_acc / len(Y_val)
        train_acc_mean = float(np.mean(train_acc_rate)) if train_acc_rate.size > 0 else 0.0
        val_acc_mean = float(np.mean(val_acc_rate)) if val_acc_rate.size > 0 else 0.0

        # Step learning rate scheduler
        model.scheduler.step()

        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f'{train_loss:.4f}', f'{val_loss:.4f}',
                f'{train_acc_rate[0]:.4f}', f'{val_acc_rate[0]:.4f}',
                f'{train_acc_mean:.4f}', f'{val_acc_mean:.4f}'
            ])

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t_start
        print(f"{epoch + 1:>5} | {train_loss:>11.4f} | {val_loss:>9.4f} | "
              f"{train_acc_mean:>10.2%} | {val_acc_mean:>9.2%} | "
              f"({elapsed // 60:.0f}m{elapsed % 60:.0f}s)")

    # Persist best checkpoint
    torch.save(best_state, model_path)
    print(f"\nBest model saved to {model_path} (Val Loss: {best_val_loss:.4f})")
    print(f"Training log saved to {csv_path}")

    return best_state


def prepare_dataloaders(X_train, X_all_train, env_train, Y_train,
                        X_val, X_all_val, env_val, Y_val, batch_size):
    """
    Convert numpy arrays into PyTorch TensorDatasets and DataLoaders.

    Returns:
        train_loader, val_loader: Configured DataLoaders.
        Y_train_t, Y_val_t: Label tensors (for accuracy computation).
    """
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_all_train_t = torch.tensor(X_all_train, dtype=torch.float32)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    env_train_t = torch.tensor(env_train, dtype=torch.float32)

    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    X_all_val_t = torch.tensor(X_all_val, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    env_val_t = torch.tensor(env_val, dtype=torch.float32)

    train_set = TensorDataset(X_train_t, X_all_train_t, env_train_t, Y_train_t)
    val_set = TensorDataset(X_val_t, X_all_val_t, env_val_t, Y_val_t)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            pin_memory=True, drop_last=False)

    return train_loader, val_loader, Y_train_t, Y_val_t


def main():
    global DEVICE

    # --- Reproducibility ---
    set_seed(SEED)

    # --- Device setup ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[CMT-GAT] Device: {DEVICE}")

    # --- Load data ---
    DATA_DIR = './data'
    print(f"\n[CMT-GAT] Loading data (mask_length={MASK_LENGTH})...")
    X_train, X_val, X_all_train, X_all_val, env_train, env_val, Y_train, Y_val = \
        load_data(DATA_DIR, MASK_LENGTH)

    print(f"  Train samples: {X_train.shape[0]} | Val samples: {X_val.shape[0]}")
    print(f"  Feature shape: T={X_train.shape[1]}, N={X_train.shape[2]}, D={X_train.shape[3]}")

    # --- Create dataloaders ---
    train_loader, val_loader, Y_train_t, Y_val_t = prepare_dataloaders(
        X_train, X_all_train, env_train, Y_train,
        X_val, X_all_val, env_val, Y_val, BATCH_SIZE
    )

    # --- Initialize model ---
    model = cmtgat_model.Network(dropout_rate=DROPOUT_RATE).to(DEVICE)

    # Attach optimizer and scheduler as attributes for convenience
    model.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.scheduler = optim.lr_scheduler.StepLR(model.optimizer,
                                                step_size=STEP_SIZE,
                                                gamma=GAMMA)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[CMT-GAT] Model initialized: {n_params:,} params ({n_trainable:,} trainable)")

    # --- Output directory ---
    OUTPUT_DIR = './output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_tag = f'CMT-GAT_m{MASK_LENGTH}'  # e.g., 'CMT-GAT_m8' for 80% missing

    # --- Train ---
    print(f"\n[CMT-GAT] Starting training: {NUM_EPOCHS} epochs, "
          f"batch={BATCH_SIZE}, lr={LEARNING_RATE}, alpha={CONTRASTIVE_WEIGHT}\n")
    best_state = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        Y_train=Y_train_t,
        Y_val=Y_val_t,
        num_epochs=NUM_EPOCHS,
        contrastive_weight=CONTRASTIVE_WEIGHT,
        output_dir=OUTPUT_DIR,
        run_tag=run_tag
    )

    elapsed = time.time()
    # Note: actual timing handled inside train()
    print("\n[CMT-GAT] Training complete.")


if __name__ == '__main__':
    main()
