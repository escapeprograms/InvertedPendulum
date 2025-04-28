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
        
        # #upward-facing joints
        self.d.ctrl[1] += - 20
        self.d.ctrl[3] += - 20

        # Oppose gravity - use this for models 3+
        # p = self.makeshift_grav()
        # for i in range(6):
        #     self.d.ctrl[i] += p[i]

        return True
    
    # Functions to account for Coriolis and gravity via the Jacobian. Helper functions are modified from HW 2 and HW 3.
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
        # T_ee_mass = self.generate_transformation_matrix(xyz=[0.22, 0.0, 0.0], rpy=[0,0,0])
        # T_5_mass = self.generate_transformation_matrix(xyz=[0.09, 0.0, 0.0], rpy=[q[5],0,0])#@T_ee_mass
        # T_4_mass = self.generate_transformation_matrix(xyz=[0.45, 0.0, 0.0], rpy=[0,q[4],0])@T_5_mass
        # T_3_mass = self.generate_transformation_matrix(xyz=[0.55, 0.0, 0.0], rpy=[0,q[3],0])@T_4_mass
        # T_2_mass = self.generate_transformation_matrix(xyz=[0.0, 0.0, 0.0], rpy=[q[2],0,0])@T_3_mass
        # T_1_mass = self.generate_transformation_matrix(xyz=[0.0, 0.0, 0.4], rpy=[0,q[1],0])@T_2_mass
        # T_0_mass = self.generate_transformation_matrix(xyz=[-0.6, 0.0, 0.0], rpy=[0,0,q[0]])@T_1_mass
        # T = {0: T_0_mass, 1: T_1_mass, 2: T_2_mass, 3: T_3_mass, 4: T_4_mass, 5: T_5_mass}#, 6: T_ee_mass}
        T_0_1 = self.generate_transformation_matrix(xyz=[-0.6, 0.0, 0.0], rpy=[0,0,q[0]])
        T_0_2 = T_0_1@self.generate_transformation_matrix(xyz=[0.0, 0.0, 0.4], rpy=[0,q[1],0])
        T_0_3 = T_0_2@self.generate_transformation_matrix(xyz=[0.0, 0.0, 0.0], rpy=[q[2],0,0])
        T_0_4 = T_0_3@self.generate_transformation_matrix(xyz=[0.55, 0.0, 0.0], rpy=[0,q[3],0])
        T_0_5 = T_0_4@self.generate_transformation_matrix(xyz=[0.45, 0.0, 0.0], rpy=[0,q[4],0])
        T_0_6 = T_0_5@self.generate_transformation_matrix(xyz=[0.09, 0.0, 0.0], rpy=[q[5],0,0])
        T_0_p = T_0_6@self.generate_transformation_matrix(xyz=[0.22, 0.0, 0.3], rpy=[q[6],0,0])
        T = {0: T_0_1, 1: T_0_2, 2: T_0_3, 3: T_0_4, 4: T_0_5, 5: T_0_6, 6: T_0_p}
        return T
    
    # Calculate Jacobian - using world Jacobian for (partial) OSC
    def Jacobian(self):

        T = self.make_SE3_from_joint_to_mass()
        S = {} # rotation axis at the local (joint) frame

        J=np.zeros((6, 6))
        S[1] = np.array([0, 0, 1])
        S[2] = np.array([0, 1, 0])
        S[3] = np.array([1, 0, 0])
        S[4] = np.array([0, 1, 0])
        S[5] = np.array([0, 1, 0])
        S[6] = np.array([1, 0, 0])
        for i in range(1, 7):
            mass = T[i]
            fr = T[0]
            w_i_mass = S[i]
            w_mass_mass = mass[:3,:3].T@w_i_mass
            w_fr_mass = fr[:3,:3]@w_mass_mass
            v_i_mass = np.cross(w_i_mass, mass[:3,3])
            v_mass_mass = mass[:3,:3].T@v_i_mass
            v_fr_mass = fr[:3,:3]@v_mass_mass
            vec = np.zeros((6,1))
            vec[:3,0] = w_fr_mass
            vec[3:,0] = v_fr_mass
            J[:,i-1:i] = vec
        return J
    
    # Compensate for Coriolis and gravity. Mass matrix is currently approximated as a diagonal matrix.
    # This doesn't work, likely because the mass matrix is not diagonal.
    def gravity_coriolis(self):
        J = self.Jacobian()
        M = np.zeros((6, 6))
        masses = [2.5, 2.3, 1.8, 1.3, 0.5, 0.2, 0.2]
        for i in range(6):
            M[i,i] = masses[i]
        # Calculate lambda
        lam = np.linalg.inv(J@np.linalg.inv(M)@J.T)
        print("Lambda: ", lam)
        # Calculate grtavitational forces
        G = np.zeros((6,1))
        for i in range(6):
            G[i,0] = masses[i]*9.81 # Not sure if this is correct. Purposefully leaving this positive so our torque opposes gravity.
        # Ignoring Coriolis for now
        # Calculate torque applied to each joint by gravity
        p = lam@J@np.linalg.inv(M)@G
        return p
    
    # Currently being used, approximates the torque applied to each (y-axis) joint by gravity.
    def makeshift_grav(self):
        masses = [2.5, 2.5, 2.5, 1.8, 1.3, 0.5, 0.2] # Per link
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
                # Need to consider case i = 1, when shoulder joint points towards joint 3, not 2
                if i == 1:
                    v_b = T[i+2][:3,3] - T[i][:3,3]
                mag_b = np.sqrt(v_b[0] ** 2 + v_b[1] ** 2 + v_b[2] ** 2)
                proj_a_b = np.dot(v_a, v_b) * v_b / mag_b
                theta = np.arccos(v_b[2] / mag_b) # Angle between joint axis and z-axis
                # Calculate the torque applied to each joint based on the angle to the z-axis.
                p[i] -= np.sin(theta) * masses[j-1] * 9.81 * np.sqrt(proj_a_b[0] ** 2 + proj_a_b[1] ** 2 + proj_a_b[2] ** 2)
        return p

#Your final Control
class YourCtrl(GenomeCtrl):
    def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
        #load model from folder
        with open("models/neat-model 7.pkl", "rb") as f:
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