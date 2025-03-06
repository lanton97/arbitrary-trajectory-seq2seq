import numpy as np
import common.configs as config
import common.util as util
from common.plotting import *
from datasets.dataset import convoyDataset
from trajectory_models.base import *
import argparse

parser = argparse.ArgumentParser(description='This script calculates the RMS error for trajectory models.')

parser.add_argument('--model', dest='model', metavar='model_name', default='SkipSeq2Seq',
                    help='Name of the model we wish to train. Valid options include ' + str(config.model_list.keys()))

parser.add_argument('--ds', dest='ds', metavar='ds_name', default='datasets/test.csv',
                    help='Path to the dataset we wish to use.')

parser.add_argument('--dev', dest='dev', metavar='dev', default='cpu',
                    help='Device we wish to use. Using argument auto automatically selects a GPU.')

parser.add_argument('--target_preproc', dest='preproc', metavar='preproc', default='CosSin',
                    help='Preprocessing used on the inital target variable for skip connections. Options include ' + str(config.preproc_list.keys()))

parser.add_argument('--load-path', dest='load_path', metavar='load-path', default='models/skipseq2seq/skipGRUDecoder/less_noise/',
                    help='Path to the trained model we wish to use.')

parser.add_argument('--save-type', dest='save', metavar='save', default='val_',
                    help='Prefix for the saved model from val_ (for best validation loss), best_ (for best training loss) and \'\' for the final weights.')

args = parser.parse_args()

# Validate selections for script and load everything
# Set up training device
if args.dev == 'auto':
    dev = util.get_device()
else:
    dev = args.dev

# Checkt trained model
if args.model not in config.model_list.keys():
    print('Invalid model: ' + args.model +'. Select from: ' + str(config.model_list.keys()))

model = config.model_list[args.model]

# Check and extract info for target/skip connection preprocessing in training
if args.preproc not in config.preproc_out_size.keys():
    print('Invalide preprocessing function selected. Choose from ' + str(config.preproc_list.keys()))

preproc_func = config.preproc_list[args.preproc]
skip_size = config.preproc_out_size[args.preproc]


# Load and setup dataset
DS = convoyDataset(file_path=args.ds)

path = args.load_path 
name = args.save

model = model(8, numSteps=100, device=dev)
modelIf = manifoldTrajectoryModel(model=model) 
modelIf.loadModel(path, name=name)


for i in range(100,120):#len(train)): 
    inp, target = DS[i]
    target_input = preproc_func(target)
    q_hat, pred = modelIf.getModelOutput(inp, target_input)

    x_tick = get_min_change_x_tick(target, delta=0.01)

    plot_trajectory(q_hat, pred, target)
    plot_state_viz(q_hat, pred, vert_x_tick=x_tick)
    plot_state_error_viz(q_hat, pred, target, vert_x_tick=x_tick)
    plot_true_and_pred(q_hat, pred, target, vert_x_tick=x_tick)

loss = np.load(path+"train_loss.npy")
val_loss = np.load(path+"val_loss.npy")

plot_rewards(loss, val_loss, show=True, y_ax_txt='Negative Log Likelihood')
