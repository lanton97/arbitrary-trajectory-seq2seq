import common.configs as config
import common.util as util
from datasets.dataset import convoyDataset
from torch.utils.data import DataLoader
import common.plotting as plotting
from trajectory_models.trainable.trainer import modelTrainer
from trajectory_models.base import *
import argparse

parser = argparse.ArgumentParser(description='This script handles training the various trajectory models.')

parser.add_argument('--model', dest='model', metavar='model_name', default='SkipSeq2Seq',
                    help='Name of the model we wish to train. Valid options include ' + str(config.model_list.keys()))

parser.add_argument('--ds', dest='ds', metavar='ds_name', default='datasets/convoyDS.csv',
                    help='Path to the dataset we wish to use.')

parser.add_argument('--dev', dest='dev', metavar='dev', default='cpu',
                    help='Device we wish to use. Using argument auto automatically selects a GPU.')

parser.add_argument('--loss', dest='loss', metavar='loss', default='BoxMinusNLL',
                    help='Loss we wish to use. Options include ' + str(config.loss_list.keys()))

parser.add_argument('--target_preproc', dest='preproc', metavar='preproc', default='CosSin',
                    help='Preprocessing used on the inital target variable for skip connections. Options include ' + str(config.preproc_list.keys()))

parser.add_argument('--epochs', dest='epochs', metavar='epochs', default='1000',
                    help='Integer for number of epochs to train for.')

parser.add_argument('--iekf_trg', dest='iekf', metavar='iekf_target', default='False')

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

epochs = int(args.epochs)

preproc_func = config.preproc_list[args.preproc]
skip_size = config.preproc_out_size[args.preproc]

iekf_trg = args.iekf=='True'

# Load and setup dataset
DS = convoyDataset(file_path=args.ds, iekf_targets=iekf_trg)

train, val = torch.utils.data.random_split(DS, [0.8,0.2])
train_loader = DataLoader(train, batch_size=128, shuffle=True)
val_loader = DataLoader(val, batch_size=128, shuffle=True)

# Scale input size based on the target input
if iekf_trg:
    init_batch, trg,_ = train[0]
else:
    init_batch, trg = train[0]
input_size = init_batch.shape[1]


# Instantiate the model
model = model(input_size, skipSize=skip_size, device=dev)

# Generate a timestamped directory to save results in based on the model name
model_name = model.model_name
path = util.generate_timestamped_path('models/' + model_name + "/")

trainer = modelTrainer(model, trgPreproc=preproc_func,iekf_trgs=iekf_trg)

print('Training Model.')
# Train model
tr, val_scr = trainer.train(train_loader, dev, val_dataloader=val_loader,epochs=epochs, save_path=path)

print('Saving training data.')
util.save_training_data(path, tr)
util.save_val_data(path, val_scr)

# Generate an interface to get outputs for plotting
modelIf = manifoldTrajectoryModel(model=model)

print('Plotting selected training set trajectories.')
for i in range(8): 
    if iekf_trg:
        inp, q, trg = train[i]
        target_input = preproc_func(trg)
    else:
        inp, q = train[i]
        target_input = preproc_func(q)

    q_hat, pred = modelIf.getModelOutput(inp, target_input)
    x_tick = plotting.get_min_change_x_tick(target, delta=0.01)

    plotting.plot_trajectory(q_hat, pred, target,save_dir=path, save_suffix='tr'+str(i))
    plotting.plot_state_viz(q_hat, pred, vert_x_tick=x_tick,save_dir=path, save_suffix='tr'+str(i))

    plotting.plot_state_error_viz(q_hat, pred, target, vert_x_tick=x_tick,save_dir=path, save_suffix='tr'+str(i))
    plotting.plot_true_and_pred(q_hat, pred, target, vert_x_tick=x_tick,save_dir=path, save_suffix='tr'+str(i))

print('Plotting selected validation set trajectories.')

for i in range(8):#len(train)): 
    if iekf_trg:
        inp, q, trg = val[i]
        target_input = preproc_func(trg)
    else:
        inp, q = val[i]
        target_input = preproc_func(q)

    q_hat, pred = modelIf.getModelOutput(inp, target_input)
    x_tick = plotting.get_min_change_x_tick(target, delta=0.01)

    plotting.plot_trajectory(q_hat, pred, target,save_dir=path, save_suffix='val'+str(i))
    plotting.plot_state_viz(q_hat, pred, vert_x_tick=x_tick,save_dir=path, save_suffix='val'+str(i))

    plotting.plot_state_error_viz(q_hat, pred, target, vert_x_tick=x_tick,save_dir=path, save_suffix='val'+str(i))
    plotting.plot_true_and_pred(q_hat, pred, target, vert_x_tick=x_tick,save_dir=path, save_suffix='val'+str(i))

plotting.plot_rewards(tr, val_scr, show=True, y_ax_txt='Negative Log Likelihood')
print(path)
