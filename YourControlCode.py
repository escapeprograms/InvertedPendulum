import mujoco
import numpy as np
import random
from scipy.linalg import inv, eig
import neat

#Your final Control
class YourCtrl:
    def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
        self.m = m
        self.d = d
        self.init_qpos = d.qpos.copy()

        # Control gains (using similar values to CircularMotion)
        self.kp = 50.0
        self.kd = 3.0


    def CtrlUpdate(self):
        #angle of pendulum
        #print(self.d.qpos[6])
        for i in range(6):
            self.d.ctrl[i] = 150.0*(self.init_qpos[i] - self.d.qpos[i]) - 5.2 *self.d.qvel[i]
        self.d.ctrl[0] = 1
        return True

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
            self.d.ctrl[i] = forces[i]

        return True
    
#definition of a Genome
class Genome:
    def __init__(self, id, network):
        self.id = id
        self.network = network
        self.fitness = 0