from common.se2 import *
import torch
from torch.optim import Adam
import tqdm
from common.util import plot_grad_flow
from common.preproc import noPreProc
from .losses.box_minus import *
import numpy as np

# A base class for trajectory models
# Keeps track of the previous observations
class modelTrainer():
    def __init__(self,
                 model,
                 max_grad=10,
                 mean_dim=4,
                 trgPreproc=noPreProc, 
                 leaderInp=True,
                 iekf_trgs=False
                 ):
        self.max_grad = max_grad
        self.model = model
        self.meanDim=mean_dim
        self.trgPreproc = trgPreproc
        self.iekfTrg = iekf_trgs

    # This loop trains the model for a given number of epochs
    # Returns the validation and training loss history
    def train(self, train_dataloader, device, val_dataloader=None, epochs=100, lr=0.005, gamma=0.995, l2_reg=0.0001, debug=False, save_path=None, loss=BoxMinusMatNLLLoss):
        # We use the Adam optimizer
        # We may change this to be configurable later
        opt = Adam(self.model.parameters(), lr=lr, weight_decay=l2_reg)

        # for debugging purposes
        # Allows us to track the root of NaNs in training
        if debug:
            torch.autograd.set_detect_anomaly(True)
        losses = []
        val_losses = []

        # Set benchamrks for saving during training
        best_avg_loss = float('inf')
        best_val_loss = float('inf')
        best_loss = float('inf')

        # Moving average window of size 10 for checking the average loss
        avg_weights = np.ones(10)/10

        # LR scheduling to decrease LR during training
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

        l = loss()
        
        # Setup model on training device
        self.model.to(device)

        # Set up a progress bar
        for i in (pbar :=tqdm.tqdm(range(epochs))):
            total_loss = 0
            # Set the model into training mode
            self.model.train()
            for batch in train_dataloader:
                # Split the batches into features and labels/targets


                # Preprocess the target
                if self.iekfTrg:
                    x, q, trg =batch[0], batch[1], batch[2]

                    target_input = self.trgPreproc(trg)
                else:
                    x, q = batch[0], batch[1]
                    target_input = self.trgPreproc(q)

                # Obtain output
                pred_means, pred_cov = self.model(x,target_input, train=not self.iekfTrg)

                # Reshape output into means and covariance matrices
                q_hat = self.constructMeans(pred_means)
                cov = self.constructCovariances(pred_cov)
             
                # Calculate the loss
                loss = l(q, pred_means, cov, device)

                # Backpropagate the error
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)

                # If we are debugging, we may wish to plot the gradients to ensure the whole network is learning, etc. 
                if debug:
                    plot_grad_flow(self.model.parameters())

                opt.step()
             
                # Track loss for plotting and comparison later
                total_loss += loss.detach().item()
                del x, q

            # Currently hardcoded limit for loss decrease
            if i < 850:
                scheduler.step()

            # Keep track of losses, clear GPU
            losses.append(total_loss/len(train_dataloader))
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda:0':
                torch.cuda.empty_cache()

            # Calculate the performance metrics and save the relevant NNs
            avg_loss = np.convolve(losses, avg_weights)[-1] 
            if save_path is not None and avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                self.saveModel(save_path, 'avg_')

            if save_path is not None and losses[-1] < best_loss:
                best_loss = losses[-1]
                self.saveModel(save_path, 'best_')

            # Get our validation loss
            val_loss = self.validate(val_dataloader, l, device)
            if save_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.saveModel(save_path, 'val_')

            val_losses.append(val_loss)

            # Update our TQDM progress bar
            pbar.set_description("Epoch Loss: {} | Val Loss: {}".format((total_loss/len(train_dataloader)), val_loss ))

        # Save the final model
        if save_path is not None:
            self.saveModel(save_path)

        return losses, val_losses

    # This function calculates the loss on the validation set without updating weights
    def validate(self, val_dataloader, l, device):
        val_loss = 0
        self.model.eval()
        # Run validation if we have allocated a dataloader for it
        if val_dataloader is not None:
            for batch in val_dataloader:
                # Split the batches into features and labels/targets
                if self.iekfTrg:
                    x, q, trg =batch[0].type(torch.float32).to(device), batch[1].type(torch.float32).to(device), batch[2].type(torch.float32).to(device)
                    target_input = self.trgPreproc(trg)
                else:
                    x, q = batch[0].type(torch.float32).to(device), batch[1].type(torch.float32).to(device)
                    target_input = self.trgPreproc(q)

              
                # Obtain output
                pred_means, pred_cov = self.model(x,target_input)
                # Reshape output into means and covariance matrices
                q_hat = self.constructMeans(pred_means)
                cov = self.constructCovariances(pred_cov)
              
                # Calculate the loss
                loss = l(q, pred_means, cov, device)
                #loss = l(q, pred_means)
                val_loss += loss.item()/len(val_dataloader)
                del x, q
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda:0':
                    torch.cuda.empty_cache()

        return val_loss

    # Transform a batch of mean vectors to a batch of mean SE2 representations
    def constructMeans(self, vecs):
        batch_list = []
        # Loop through each batch
        for batch in vecs:
            mean_list = []
            # Loop through each timestep in the batch
            for vec in batch:
                mean_list.append(self.constructMean(vec))
            batch_list.append(mean_list)
        return batch_list

    # Transform a single SE2 vector to an SE2 matrix
    def constructMean(self, vec):
        se2 = SE2(man_vec=vec)
        t_mat = se2.t_matrix
        return t_mat

    # A method that takes the NN output and reconstructs the covariance matrix
    def constructCovariances(self, vec):
        cov = torch.zeros((vec.shape[0], vec.shape[1], self.meanDim, self.meanDim))#, requires_grad=True)
        #cov = []
        for i in range(self.meanDim):
            auto_cov = torch.exp(vec[:,:,i])
            cov[:,:,i,i] = torch.exp(vec[:,:,i])

        # Keep track of our relevant pearson coeeficient
        pearsonCount = 0
        for i in range(self.meanDim):
            for j in range(i+1):
                # Skip if we already have the element from the diagonal covariance
                if i != j:
                    # Calculate the off-diagonal covariance matrix entry
                    cov[:,:,i,j] = torch.tanh(vec[:,:,self.meanDim + pearsonCount]) *torch.sqrt(torch.exp(vec[:,:,i])*torch.exp(vec[:,:,j]))
                    cov[:,:,j,i] = torch.tanh(vec[:,:,self.meanDim + pearsonCount]) *torch.sqrt(torch.exp(vec[:,:,i])*torch.exp(vec[:,:,j]))

                    pearsonCount += 1

        return cov

            
    # Save a model with a given name in the relevant filePath
    def saveModel(self, filePath, name=''):
        torch.save(self.model, filePath + name + 'model.pt')

    # Load the given filepath and model name
    def loadModel(self, filePath, name=''):
        self.model = torch.load(filePath + name + 'model.pt')





