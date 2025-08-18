from ros_if.preproc import convertToCosSin
from ros_if.gmpc import GeometricMPC
from ros_if.nn_wrapper import NeuralNetworkWrapper
import numpy as np
import torch


class SE2andGMPCIf():
    def __init__(self,model_path, pre_proc=convertToCosSin):
        self._obs_hist = []
        self._leader_vel_hist = []
        self._ctrl_hist = []

        model = torch.load(model_path +  'model.pt')
        self._nn_model = NeuralNetworkWrapper(model, pre_proc, dt=0.1)
        self._controller = GeometricMPC(relativeState=True)

    def push_observation(self, ego_pos, relative_pos, leader_vel):
        if len(self._obs_hist) < 100:
            self._obs_hist.append(relative_pos)
            self._leader_vel_hist.append(leader_vel)
            self._ctrl_hist.append(np.array([0.0,0.0]))
        elif len(self._obs_hist) == 100:
            init_pos = np.array(ego_pos)
            self._nn_model.push_init(init_pos, self._obs_hist, self._ctrl_hist, self._leader_vel_hist)
        else:

            self._nn_model.step(ego_pos, relative_pos, self._ctrl_hist[-1], leader_vel_hist[-1], )

    def get_control(self):
        traj, covs = self._nn_model.traj
        # TEST 1 for controller
        traj = np.array([[0.05*(i+1), 0.1*(i+), 0.0] for i in range(100)])
        ctrl = self._controller.demand(None, traj, None)
        self._ctrl_hist.append(ctrl)
        return ctrl

    def dump_history(self):
        return self._obs_hist, self._ctrl_hist
if __name__=="__main__":
    ros_if = SE2andGMPCIf('ros_if/models/' + 'avg_')
    ctrl = ros_if.get_control()
    print(ctrl(None, None, None))

