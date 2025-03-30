import mujoco
import numpy as np
import random
from scipy.linalg import inv, eig
import neat
import pickle
#controller defined by a NN
class GenomeCtrl:
    def __init__(self, m:mujoco.MjModel, d: mujoco.MjData, network: neat.nn.FeedForwardNetwork):
        self.m = m
        self.d = d
        self.init_qpos = d.qpos.copy()

        # Control gains (using similar values to CircularMotion)
        self.kp = 50.0
        self.kd = 3.0

        #network
        self.network = network

    def CtrlUpdate(self):
        #angle of pendulum
        #print(self.d.qpos[6])
        forces = self.network.activate(self.d.qpos)
        for i in range(6):
            self.d.ctrl[i] = 10*forces[i]

        for i in range(1, 6):
            self.d.ctrl[i] += 150.0*(self.init_qpos[i] - self.d.qpos[i]) - 5.2 *self.d.qvel[i]
        
        #upward-facing joints
        self.d.ctrl[1] += - 20
        self.d.ctrl[3] += - 20

        return True
    

#Your final Control
class YourCtrl(GenomeCtrl):
    def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
        #load model
        with open("models/neat-model 2.pkl", "rb") as f:
            network = pickle.load(f)
        
        super().__init__(m, d, network)
        self.m = m
        self.d = d
        self.init_qpos = d.qpos.copy()

#definition of a Genome
class Genome:
    def __init__(self, id, network):
        self.id = id
        self.network = network
        self.fitness = 0