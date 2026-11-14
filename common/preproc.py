import torch

# This preprocessing function performs no actions
def noPreProc(target):
    return target

# This preprocessing function converts the target input(X, Y, \theta)
# into (x, y, cos(\theta), sin(\theta)
# We also handle batchs 
def convertToCosSin(target):
    # Check if we are already in the correct format
    if target.shape[-1] == 4:
        return target

    # Check if we are using a batch or a single observation
    if len(target.shape) == 3:
        target_th = target[:,:,-1]

        target_cos = torch.unsqueeze(torch.cos(target_th),2)
        target_sin = torch.unsqueeze(torch.sin(target_th),2)
        target_input = torch.concat([target[:,:,:2], target_cos, target_sin], dim=2)
    else:
        target_th = target[:,-1]
        target_cos = torch.unsqueeze(torch.cos(target_th),1)
        target_sin = torch.unsqueeze(torch.sin(target_th),1)
        target_input = torch.concat([target[:,:2], target_cos, target_sin], dim=1)

    return target_input



# Delay padding: lets a model trained with a fixed P-step window (P=100) be queried
# for a shorter delay D <= P
def padDelayWindow(window, delay, total=100, vel_start=4):
    if delay >= total:
        return window[-total:]
    real = window[-delay:]
    pad = real[-1:].repeat(total - delay, 1) if torch.is_tensor(real) else \
          real[-1:].repeat(total - delay, axis=0)
    pad[:, vel_start:] = 0
    if torch.is_tensor(real):
        return torch.cat([real, pad], dim=0)
    import numpy as np
    return np.concatenate([real, pad], axis=0)


# Matching target: keep the first D poses and hold the D-th for the rest of the horizon
def padDelayTarget(target, delay, total=100):
    if delay >= total:
        return target
    hold = target[delay - 1:delay].repeat(total - delay, 1)
    return torch.cat([target[:delay], hold], dim=0)


# Set the number of output steps for models with an autoregressive decoder
def setModelHorizon(model, num_steps):
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'seqLen'):
        model.decoder.seqLen = num_steps
    if hasattr(model, 'seqLen'):
        model.seqLen = num_steps
