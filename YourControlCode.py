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
        # print(self.d.qpos[6])
        forces = self.network.activate(self.d.qpos)
        for i in range(6):
            self.d.ctrl[i] = 10*forces[i]

        # PD control
        for i in range(1, 6):
            self.d.ctrl[i] += 150.0*(self.init_qpos[i] - self.d.qpos[i]) - 5.2 *self.d.qvel[i]
        
        # basic-crutch control - resist gravity on upward-facing joints
        self.d.ctrl[1] += - 20
        self.d.ctrl[3] += - 20

        # # grav_comp control
        # p = self.grav_comp()
        # for i in range(6):
        #     self.d.ctrl[i] += p[i]
        # self.d.ctrl[4] += 0.2 # Stabilize joint 4

        # OSC for JUST y-axis joints - we want to keep x-axis joints fully based on the model,
        # since ideally they will not rotate and their torque will be handled by the z-axis rotation,
        # and we want to allow the model full control of the one z-axis joint for dealing with the mass.
        # for i in range(6):
        #     self.d.ctrl[i] += self.d.qfrc_bias[i]

        # Extra help to emphasize keeping the pendulum at the same height. Accounts for compounding integration error.
        # Used in OSC model 2 and in balancing with OSC demonstration (balanced with no external force).
        # self.d.ctrl[1] -= 0.4 * np.sin(self.d.qpos[1])
        # self.d.ctrl[3] -= 0.3 * np.sin(self.d.qpos[1] + self.d.qpos[3])
        # self.d.ctrl[4] -= 0.2 * np.sin(self.d.qpos[1] + self.d.qpos[3] + self.d.qpos[4])

        return True

    # Calculate OSC to oppose gravity, Coriolis, and centrifugal forces. Note: OSC is stored in Mujoco as qfrc_bias, so this function is not used.
    def osc(self):
        # Find the position of the end effector in the global frame
        ee_id = self.m.geom("mass").id
        ee_pos = self.d.geom("mass").xpos
        nv = self.m.nv
        JacP = np.zeros((3, nv), np.float64)
        JacR = np.zeros((3, nv), np.float64)
        mujoco.mj_jac(m=self.m, d=self.d, jacp=JacP, jacr=JacR, point=ee_pos, body=ee_id) # Jacobian for the end effector
        M = np.zeros((nv, nv), np.float64)
        mujoco.mj_fullM(m=self.m, dst=M, M=self.d.qM) # Mass matrix
        mujoco.mj_rne(m=self.m, d=self.d, flg_acc=0, result=self.d.qfrc_bias) # Compute Coriolis and gravity
        # Calculate OSC values
        # J = np.array([JacP, JacR])
        # Minv = np.linalg.pinv(M)
        # lam = np.linalg.inv(J@Minv@J.T)
        return(self.d.qfrc_bias)
    
    # Functions to account for Coriolis and gravity. Helper functions are modified from HW 2 and HW 3.
    def generate_transformation_matrix(self, xyz, rpy):
        # SO3 rotation matrix
        matx = np.array([[1, 0, 0],
                        [0, np.cos(rpy[0]), -np.sin(rpy[0])],
                        [0, np.sin(rpy[0]), np.cos(rpy[0])]])
        maty = np.array([[np.cos(rpy[1]), 0, np.sin(rpy[1])],
                        [0, 1, 0],
                        [-np.sin(rpy[1]), 0, np.cos(rpy[1])]])
        matz = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]), 0],
                        [np.sin(rpy[2]), np.cos(rpy[2]), 0],
                        [0, 0, 1]])
        R = np.matmul(np.matmul(matz, maty), matx)

        # SE3 transformation matrix
        T = np.zeros((4,4))
        T[:3, :3] = R
        T[0, 3] = xyz[0]
        T[1, 3] = xyz[1]
        T[2, 3] = xyz[2]
        T[3, 3] = 1.0
        return T

    def make_SE3_from_joint_to_mass(self):
        T = {}
        q = self.d.qpos
        T_0_1 = self.generate_transformation_matrix(xyz=[-0.6, 0.0, 0.0], rpy=[0,0,q[0]])
        T_0_2 = T_0_1@self.generate_transformation_matrix(xyz=[0.0, 0.0, 0.4], rpy=[0,q[1],0])
        T_0_3 = T_0_2@self.generate_transformation_matrix(xyz=[0.0, 0.0, 0.0], rpy=[q[2],0,0])
        T_0_4 = T_0_3@self.generate_transformation_matrix(xyz=[0.55, 0.0, 0.0], rpy=[0,q[3],0])
        T_0_5 = T_0_4@self.generate_transformation_matrix(xyz=[0.45, 0.0, 0.0], rpy=[0,q[4],0])
        T_0_6 = T_0_5@self.generate_transformation_matrix(xyz=[0.09, 0.0, 0.0], rpy=[q[5],0,0])
        T_0_p = T_0_6@self.generate_transformation_matrix(xyz=[0.22, 0.0, 0.3], rpy=[q[6],0,0])
        T = {0: T_0_1, 1: T_0_2, 2: T_0_3, 3: T_0_4, 4: T_0_5, 5: T_0_6, 6: T_0_p}
        return T
    
    # Calculate approximate force of gravity on each joint in the current orientation. Used in grav_comp models.
    def grav_comp(self):
        masses = [2.5, 2.5, 2.5, 1.8, 1.3, 0.5, 0.5]
        T = self.make_SE3_from_joint_to_mass()
        p = np.zeros((6,1))
        for i in range(5): # Only need to calculate joints 1, 3, and 4
            # Calculate the center of mass for all connected links
            # The only links that could be affected by gravity are ones that rotate along the y axis
            if i == 0 or i == 2 or i == 5:
                continue
            for j in range(i+1, 7):
                # Calculate the gravitational force from the mass of each link
                # Find distance in the direction of the joint axis via vector projection
                v_a = T[j][:3,3] - T[i][:3,3] # Distance in global x, y, z from joint i to j
                v_b = T[i+1][:3,3] - T[i][:3,3] # Direction of joint axis in global x, y, z
                # Need to consider cases i = 1 and i = 4, when joints do not point towards the next joint
                if i == 1 or i == 4:
                    v_b = T[i+2][:3,3] - T[i][:3,3]
                mag_b = np.sqrt(v_b[0] ** 2 + v_b[1] ** 2 + v_b[2] ** 2)
                proj_a_b = np.dot(v_a, v_b) * v_b / mag_b
                theta = np.arccos(v_b[2] / mag_b) # Angle between joint axis and z-axis
                # Calculate the torque applied to each joint
                p[i] -= np.sin(theta) * masses[j-1] * 9.81 * np.sqrt(proj_a_b[0] ** 2 + proj_a_b[1] ** 2 + proj_a_b[2] ** 2)
        return p

#Your final Control
class YourCtrl(GenomeCtrl):
    def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
        #load model from folder
        with open("models/basic-crutch/neat-model 7.pkl", "rb") as f:
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