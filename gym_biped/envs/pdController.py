import numpy as np
import pybullet as p
class pdController(object):

    def __init__(self, bipedId, clientId, timeStep):
        self.bipedId = bipedId
        self.clientId = clientId
        self.kp = [0, 0, 0, 500, 500, 500, 500, 200, 200]
        self.kd = [0, 0, 0, 50, 50, 50, 50, 20, 20]
        self.kp = np.diagflat(self.kp)
        self.kd = np.diagflat(self.kd)
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
