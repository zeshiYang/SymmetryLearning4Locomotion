from runner import *
from PPO import *
from tensorboardX  import SummaryWriter
from model import *

if __name__ == "__main__":
    import pybullet_envs
    import gym
    from gym_biped.envs.bipedEnv import *
    #env = gym.make("Walker2d-v2")
    env = bipedEnv(False)
    env.setKpandKd(1000)

    #env.isRender = False
    env.reset()
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space, hidden=[128, 64])
    critic = Critic(env.observation_space.shape[0], 0, 3/(1-0.99), hidden =[128, 64])
    s_norm = Normalizer(env.observation_space.shape[0])
    PPO("./Walker2d-Bullet-curriculum-5-50", env, actor, critic, s_norm, 0.2, 2000, 64, 3e-4, 3e-4, 0.5)