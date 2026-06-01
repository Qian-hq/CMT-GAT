"""
Loss functions for CMT-GAT training.

Provides:
- loss_cross_entropy: Per-timestep cross-entropy prediction loss.
- forcast_value:     Per-class accuracy computation for monitoring.
"""

import numpy as np
import torch
import torch.nn.functional as F


def loss_cross_entropy(output, batch_Y):
    """
    Compute cross-entropy loss summed over all time steps.

    Args:
        output:  Model predictions of shape (B, T, N, num_classes).
        batch_Y: Ground-truth labels of shape (B, T, N, num_classes) (one-hot).

    Returns:
        Total scalar loss (sum over time steps).
    """
    output = output.reshape(batch_Y.size())
    output = output.transpose(0, 1)
    batch_Y = batch_Y.transpose(0, 1)
    loss = 0
    for i in range(len(batch_Y)):
        y = batch_Y[i]
        x = output[i]
        loss_value = F.cross_entropy(x, y)
        loss = loss + loss_value
    return loss


def forcast_value(output, batch_Y):
    """
    Compute per-class accuracy counts and collect predictions vs. ground truth.

    Args:
        output:  Model predictions (B, T, N, num_classes).
        batch_Y: Ground-truth labels (B, T, N, num_classes).

    Returns:
        acc_array:   Per-time-step correct-count array, shape [T].
        pred_labels: Stacked arrays of predicted and true labels for logging.
    """
    output = output.reshape(batch_Y.size())
    output = output.transpose(0, 1)
    batch_Y = batch_Y.transpose(0, 1)
    acc = []
    for i in range(len(output)):
        y = batch_Y[i]
        x = output[i]
        action_x = torch.max(x, 1)[1].cpu().numpy()
        action_y = torch.max(y, 1)[1].cpu().numpy()
        xy_acc = action_x - action_y
        xy_acc = xy_acc.size - np.count_nonzero(xy_acc)
        acc.append(xy_acc)
        xy_arr = np.concatenate((action_x, action_y))
        xy_arr = np.expand_dims(xy_arr, axis=0)
        if i == 0:
            forcast_arr = xy_arr
        else:
            forcast_arr = np.concatenate((xy_arr, forcast_arr))
    acc = np.array(acc)
    return acc, forcast_arr
