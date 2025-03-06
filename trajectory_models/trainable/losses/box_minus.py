from torch import nn
import torch
import math

# A box-minus loss for calculating the difference between two positions
# on an SE2 manifold
class BoxMinusMatNLLLoss(nn.Module):
    def __init__(self):
        super(BoxMinusMatNLLLoss, self).__init__()

    # Calculate the loss
    def forward(self, q, q_hat, cov, device):
        # Calculate the boxminus operation for the true value and the estimate and send it to the device 
        q_diff = self.BoxMinusMat(q, q_hat).reshape(-1, 1, 4).to(device)
        cov = cov.reshape(-1, 4, 4).to(device)
        # Calculate the NLL of the estimate
        l_traj = 0.5 * torch.bmm(torch.bmm(q_diff, torch.linalg.inv(cov)), q_diff.reshape(-1,4,1))
        l_cov = 0.5 * torch.log(torch.norm(cov, dim=[1,2]))
        return torch.mean(l_traj + l_cov)
  
    def BoxMinusMat(self, q, q_hat):
        # Get the x-difference
        x = q[:,:,0] - q_hat[:,:,0]
        y = q[:,:,1] - q_hat[:,:,1]
        # Avoid NaN outputs by adding epsilon to the estimates
        epsilon = 1e-7
        nudge = (q_hat[:,:,3] == 0) * epsilon
        # Calculate the theta deltas for sing and cos theta
        th1 = vpi(q[:,:,2]- torch.atan2(q_hat[:,:,2],q_hat[:,:,3] + nudge))
        th2 = vpi(q[:,:,2]- torch.atan2(q_hat[:,:,2],q_hat[:,:,3] + nudge))
        return torch.stack([x,y,th1, th2], 2)

# A basic box-minus mean squared error loss. Does not account for the covariance at all
class BoxMinusMSELoss(nn.Module):
    def __init__(self):
        super(BoxMinusMSELoss, self).__init__()

    def forward(self, q, q_hat, cov, device):
        q_diff = self.BoxMinusVector(q, q_hat).reshape(-1, 1, 3).to(device)
        return torch.mean(torch.linalg.vector_norm(q_diff, dim=2))
  
    def BoxMinusVector(self, q, q_hat):
        x = q[:,:,0] - q_hat[:,:,0]
        y = q[:,:,1] - q_hat[:,:,1]
        th =vpi(q[:,:,2]- q_hat[:,:,2])
        return torch.stack([x,y,th], 2)

# Vpi implementation based on the description the Vector Space and planar rotations described
# in Integrating generic sensor fusion algorithms with sound state representations
# through encapsulation of manifolds, sec. 3.6.1, 3.6.2
def vpi(delta):
    return delta - 2*math.pi * torch.floor( (delta + math.pi) / (2*math.pi))
