import numpy as np
import random
import os
import datetime
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import platform
import math

# This method is used to generate timeouts when using the external
# RRT planner, which hangs occasionally
def RRTTimeoutHandler(signum, frame):
    print('Planning timed out')
    raise Exception("RRT Timeout")

# Get the device we are training on 
def get_device():
    if "darwin" in platform.system().lower():
        device = torch.device("mps")
    elif "linux" in platform.platform():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device("cpu")
    return device

# Create and return the name of a directory to store results in
def generate_timestamped_path(dir):
    exp_dir = os.path.join(dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))
    os.makedirs(exp_dir)
    return exp_dir


def save_training_data(dir, data):
    train_data = np.array(data)
    np.save(dir+"train_loss", train_data)

def save_val_data(dir, data):
    val_data = np.array(data)
    np.save(dir+"val_loss", val_data)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for p in named_parameters:
        if(p.requires_grad):# and ("bias" not in n):
            #layers.append(n)
            if p.grad is None:
                ave_grads.append(0)
                max_grads.append(0)
            else:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    #plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.2) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
