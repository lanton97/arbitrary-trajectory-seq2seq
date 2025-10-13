from ros_if.preproc import convertToCosSin
from ros_if.gmpc import GeometricMPC
from ros_if.nn_wrapper import NeuralNetworkWrapper
import numpy as np
import torch
import sys
from trajectory_models.trainable.seq2seq import skipSeq2SeqModel
sys.path.insert(0, 'vicon/trajectory_models/')


class SE2andGMPCIf():
    def __init__(self,model_path, pre_proc=convertToCosSin):
        self._obs_hist = []
        self._leader_vel_hist = []
        self._ctrl_hist = []
        model = skipSeq2SeqModel(8)
        model.load_state_dict(torch.load(model_path +  'model.pt', weights_only=True))
        model.eval()
        self._nn_model = NeuralNetworkWrapper(model, pre_proc, dt=0.1)
        self._controller = GeometricMPC(relativeState=True)

    def push_observation(self, ego_pos, relative_pos, leader_vel):
        if len(self._obs_hist) < 99:
            self._obs_hist.append(relative_pos)
            self._leader_vel_hist.append(leader_vel)
            self._ctrl_hist.append(np.array([0.0,0.0]))
        elif len(self._obs_hist) == 99:
            self._obs_hist.append(relative_pos)
            self._leader_vel_hist.append(leader_vel)
            self._ctrl_hist.append(np.array([0.0,0.0]))
            init_pos = np.array(ego_pos)
            self._nn_model.push_init(init_pos, self._obs_hist, self._ctrl_hist, self._leader_vel_hist)
        else:
            return self._nn_model.step(relative_pos, ego_pos, self._ctrl_hist[-1], leader_vel, None)

    def get_control(self):
        traj, covs = self._nn_model.traj
        ctrl = self._controller.demand(None, traj, None)
        self._ctrl_hist.append(ctrl(None,None,None))
        return ctrl#(None,None,None)

    def dump_history(self):
        return self._obs_hist, self._ctrl_hist
if __name__=="__main__":
    ros_if = SE2andGMPCIf('ros_if/models/' + 'avg_')
    ctrl = ros_if.get_control()
    print(ctrl)

