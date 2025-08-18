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

