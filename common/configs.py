from trajectory_models.trainable import *
from trajectory_models.trainable.losses import *
from simulations.bullet_sim.utils.enum_class import EnvType
from common.preproc import noPreProc, convertToCosSin
from controllers import *
from trajectory_models.wrappers import *
import torch
from stable_baselines3 import SAC
import stable_baselines3
from stable_baselines3.common.monitor import Monitor

# A dictionary linking script names to actual model objects
model_list={'SkipSeq2Seq':skipSeq2SeqModel,
            'Seq2Seq':Seq2SeqModel,
            'Transformer': TransAm,
            'SkipTransformer': skipTransAm
        }

# Models including a skip transformer
skip_models=[skipSeq2SeqModel,
             skipTransAm]

# Preprocessing functions and their output sizes
preproc_list={'noPreProc':noPreProc,
              'CosSin':convertToCosSin
              }

preproc_out_size = {'noPreProc':3,
                    'CosSin':4,
                    }

# Implemented box-minus losses for training
loss_list = {'box-minus-NLL': BoxMinusMatNLLLoss,
             'box-minus-MSE': BoxMinusMSELoss,
             }

# A list of state-feedback controllers
controller_list = {'pid': PIDStub,
                   'mpc': SimpleMPC,
                   'gmpc': GeometricMPC,
        }

# A list of model wrappers for the state-feedback loop
model_wrapper_list = {'stub': StubWrapper,
                      'neural-net': NeuralNetworkWrapper,
                      'iekf': DualIEKFWrapper,
                      }

bullet_models = { 'turtle' : EnvType.TURTLEBOT,
                 'scout': EnvType.SCOUT_MINI
        }

# Load a trajectory model and handle the varying inputs
# Returns the loaded and configured model
def load_traj_model(model_name, load_path, load_name, preproc_func, relative_state, dt=0.1):
    if model_name=='stub':
        return model_wrapper_list[model_name](relative_state=relative_state)
    if model_name=='nn' or model_name=='neural-net':
        model = torch.load(load_path + load_name + 'model.pt')
        model = model_wrapper_list[model_name](model, preproc_func, dt=dt)
        return model
    if model_name=='iekf':
        return model_wrapper_list[model_name](dt=dt)

# Given a name for an RL agent, an environment name, and a loaded model, construct the running
# A\agent that trains the requested RL agent
# Q and R correspond with noise for the states and inputs, respectively
def load_rl_agent(agent_name, env_name, model, Q, R, preproc): 
    agent = None
    env = env_list[env_name](model, Q, R)
    if agent_name in sb_agents:
        env = Monitor(env)
        print(env)
        agent = agent_list[agent_name]('MlpPolicy', env, verbose=1, buffer_size=100000)
    else:
        agent = agent_list[agent_name](env, model)  

    return agent, env

                      
