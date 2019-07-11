import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import time
import math
import numpy as np
from IPython import embed
class pdController(object):
    
    def __init__(self, bipedId, clientId, timeStep):
        self.bipedId = bipedId
        self.clientId = clientId
        self.kp = [0, 0, 0, 500, 500, 500, 500, 200, 200]
        self.kd = [0, 0, 0, 50, 50, 50, 50, 20, 20]
        self.kp = np.diagflat(self.kp)
        self.kd = np.diagflat(self.kd)
        self.init_kp = self.kp.copy()
        self.init_kd = self.kd.copy()
        self.timeStep = timeStep
        self.num_joints = p.getNumJoints(self.bipedId, physicsClientId=self.clientId)

    def getPosandVel(self):

        pos=[]
        vel=[]
        for i in range(self.num_joints):
            state = p.getJointState(self.bipedId, i, physicsClientId=self.clientId)
            pos.append(state[0])
            vel.append(state[1])
        
        return np.array(pos), np.array(vel)

    def getForce(self, targetPos, vel):
        pos_now, vel_now = self.getPosandVel()
        targetPos = np.array([0]*3 + targetPos.tolist())
        targetPos[0] = pos_now[0]
        vel = np.array([vel, 0, 0, 0, 0, 0, 0, 0, 0])
        pos_part = self.kp.dot(pos_now - targetPos)
        vel_part = self.kd.dot(vel_now)
        M = p.calculateMassMatrix(self.bipedId, pos_now.tolist())
        M = np.array(M)
        M = (M + self.kd * self.timeStep)
        c = p.calculateInverseDynamics(self.bipedId, pos_now.tolist(), vel_now.tolist(), [0]*9)
        c = np.array(c)
        b = -pos_part - vel_part -self.kp.dot(vel_now)*self.timeStep - c
        qddot = np.linalg.solve(M, b)      
        tau = -pos_part - vel_part - self.kd.dot(qddot) * self.timeStep-self.kp.dot(vel_now)*self.timeStep + self.kd.dot(vel)

        return tau


class bipedEnv(gym.Env):
    metadata = {'render.modes' : ['human']}


    def __init__(self, renders = True):

        self.renders = renders

        #set renders to True to render the biped environment
        if(self.renders == True):
            self.clientId = p.connect(p.GUI)
        else:
            self.clientId = p.connect(p.DIRECT)

        #simulation parameters
        self.timeStep = 0.002
        self.num_substep = 30
        self.num_step = 0
        self.configure()
        for i in range(p.getNumJoints(self.bipedId)):
            print(p.getJointInfo(self.bipedId, i))
        #load the biped model
       
        self.pdCon = pdController(self.bipedId, self.clientId, self.timeStep)

        #gym settings
        self.action_space = spaces.Box(low = -1, high = 1, shape=(6, ))
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (21, ))
        self.scale_action = np.array([math.pi/2, math.pi/2, math.pi/2, math.pi/2
        , 0.5, 0.5])
        self.episode_length = 50
        self.vel =0
        self.vel_target = 1.0
        self.pos_current = 0
        self.pos_prev = 0

        #reward weights
        self.w_action = 1
        self.w_velocity = 3
        self.w_live = 1
        self.live_bonus = 4
        self.w_upright = 1
        self.w_foot_lift =0.0
        self.w_jump = 0.0
       
        #Symmetry Matrix
        self.M_action=np.array(
        [[0, 1, 0, 0, 0, 0], 
        [1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1, 0]])

        self.M_state = np.zeros((21, 21))

        #init M_state
        self.M_state[0,0] = 1
        self.M_state[1:7, 1:7] = self.M_action
        self.M_state[7, 7] = 1
        self.M_state[8:14, 8:14] = self.M_action
        self.M_state[14, 14] = 1
        self.M_state[15:19, 15:19] = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]]
        )
        self.M_state[19, 19] = 1
        self.M_state[20, 20] = 1


        #curriculum setting
        self.kpandkd_start = self.pdCon.init_kp[2,2]
        self.kpandkd_end = self.kpandkd_start*0.75*0.75

    def step(self, action):

        kpandkd = self.kpandkd_start - (self.num_step)/(self.episode_length)*(self.kpandkd_start - self.kpandkd_end)
        self.pdCon.kp[2, 2] =kpandkd
        self.pdCon.kd[2, 2] =0.1*kpandkd
        self.pdCon.kd[0, 0] =kpandkd

        self.vel = np.min([2*self.num_step*self.timeStep*self.num_substep, self.vel_target])
        self.pos_prev =  p.getLinkState(self.bipedId, 2,  physicsClientId=self.clientId)[0][1]

        action_real = action * self.scale_action      
        joint_idx = np.arange(self.pdCon.num_joints).tolist()
        for i in range(self.num_substep):
            forces = self.pdCon.getForce(action_real, self.vel).tolist()
            p.setJointMotorControlArray(self.bipedId, joint_idx, p.TORQUE_CONTROL, forces= forces)
            p.stepSimulation(physicsClientId=self.clientId)
            if(self.renders==True):
                pos_root = p.getLinkState(self.bipedId, 2,  physicsClientId=self.clientId)[0]
                p.resetDebugVisualizerCamera(2, 90, 0, [1, pos_root[1], 1 ], self.clientId)
                time.sleep(self.timeStep)
        self.num_step += 1

        self.pos_current =  p.getLinkState(self.bipedId, 2,  physicsClientId=self.clientId)[0][1]
        obv = self.getObv()
        rwd_action, rwd_live, rwd_upright, rwd_vel, rwd_foot, rwd_jump = self.getRwd(action, obv)
        done = self.getDone(obv)
        rwd = self.w_action*rwd_action + self.w_live* rwd_live + self.w_upright*rwd_upright + self.w_velocity*rwd_vel + self.w_foot_lift*rwd_foot + self.w_jump*rwd_jump
        info={}
        info["rwd_action"] = rwd_action
        info["rwd_live"] = rwd_live
        info["rwd_upright"] = rwd_upright
        info["rwd_vel"] = rwd_vel
        info["rwd_foot"] = rwd_foot
        info["rwd_jump"] = rwd_jump
        return obv, rwd, done, info

    def setKpandKd(self,kd):
        self.pdCon.init_kp[2,2] = kd
        self.pdCon.init_kd[0,0] = kd
        self.pdCon.init_kd[2,2] = 0.1*kd

        self.kpandkd_start = self.pdCon.init_kp[2,2]
        self.kpandkd_end = self.kpandkd_start*0.75*0.75


    def reset(self):
        
        #reset the biped to initial pose and velocity
        start_pose=[0, 0, 0., 0.05, 0, 0, 0, 0, 0]
        for i in range(p.getNumJoints(self.bipedId, physicsClientId=self.clientId)):
            p.resetJointState(self.bipedId, i, start_pose[i], 0, physicsClientId= self.clientId)
        self.num_step = 0
        self.pdCon.kp = self.pdCon.init_kp.copy()
        self.pdCon_kd = self.pdCon.init_kd.copy()
        return self.getObv()

    def getObv(self):
        obv=[]
        pos, vel =self.pdCon.getPosandVel()
        obv += pos[-7:].tolist()
        obv += vel[-7:].tolist()
        pos_root = np.array(p.getLinkState(self.bipedId, 2, physicsClientId=self.clientId)[0])
        pos_leftfoot = np.array(p.getLinkState(self.bipedId, 8, physicsClientId=self.clientId)[0]) - pos_root
        pos_rightfoot = np.array(p.getLinkState(self.bipedId, 7, physicsClientId=self.clientId)[0]) - pos_root
        obv.append(pos_root[2])
        obv += [pos_leftfoot[1], pos_leftfoot[2]]
        obv += [pos_rightfoot[1], pos_rightfoot[2]]
        root_vel = p.getLinkState(self.bipedId, 2, computeLinkVelocity=1, physicsClientId=self.clientId)[6][1]
        obv += [root_vel]
        obv+=[self.vel]
        obv = np.array(obv)
        return obv

    def getRwd(self, action, obv):
        vel = (self.pos_current - self.pos_prev)/(self.num_substep*self.timeStep)
        rwd_vel = -abs(vel - self.vel)
        rwd_action = -np.linalg.norm(action)
        rwd_upright = -abs(obv[0])
        pos_leftleg = np.array(p.getLinkState(self.bipedId, 4, physicsClientId=self.clientId)[0])
        pos_rightleg = np.array(p.getLinkState(self.bipedId, 3, physicsClientId=self.clientId)[0])
   
        rwd_foot = np.min([np.max([pos_leftleg[2], pos_rightleg[2]]) ,0.7]) - 0.7
        rwd_jump = np.max([obv[14]-1.1, 0])

        #print("vel:{}".format(rwd_vel))
        #print("action:{}".format(rwd_action))
        #print("live:{}".format(rwd_live))
        #print("upright:{}".format(rwd_upright))
        
        return rwd_action, self.live_bonus, rwd_upright, rwd_vel, rwd_foot, rwd_jump

    def getDone(self, obv):

        return not((self.num_step<self.episode_length) and (obv[0]> -0.4) and (obv[0]<0.2) and  (np.isfinite(obv).all())
        and (abs(obv[14]-1.09)<0.2))

    def getSymmetryState(self, state):

        return self.M_state.dot(state)

    def getSymmetryAction(self, action):

        return self.M_action.dot(action)

    def setRenders(self, renders):
        self.renders = renders
    

    def render(self, mode='human'):
        pass


    def close(self):
        pass


    def configure(self):
        p.setTimeStep(self.timeStep)
        p.setGravity(0, 0, -10)
        self.bipedId = p.loadURDF("./human_model/biped2d_pybullet.urdf", basePosition=[0, 0, 1.095], useFixedBase=True, flags=p.URDF_MAINTAIN_LINK_ORDER )
        self.planeId = p.loadURDF("./human_model/plane.urdf", globalScaling = 2)

        #clear the torque in every joint
        for i in range(p.getNumJoints(self.bipedId, physicsClientId=self.clientId)):
            p.setJointMotorControl2(self.bipedId, i, p.POSITION_CONTROL, force=0, physicsClientId=self.clientId)

        



if __name__ == "__main__":
    env=bipedEnv()
    env.reset()
    action_list=None
    obv_list=None
    para = []
    name=['right_hip', 'left_hip', 'right_knee', 'left_knee', 'right_ankle', 'left_ankle']
    for i in range(6):
        para.append(p.addUserDebugParameter(name[i], -1, 1, 0))
    while(1):
        action=[]
        for i in range(6):
            action.append(p.readUserDebugParameter(para[i]))
        #action = [0]*6
        action=np.array(action)  
        #action = env.action_space.sample()      
        obv, rwd, done, info =env.step(action)
        done = False
        if(done == True):
            env.reset()
         
 
       

