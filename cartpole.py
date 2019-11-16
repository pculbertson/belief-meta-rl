import gym, torch
from utils import ReplayBuffer
from models import TransitionNet, RewardNet
env = gym.make('CartPole-v1')
s = env.reset()
dim_actions = env.action_space.shape[0] if env.action_space.shape else 1 #if discrete return dim = 1
dim_obs = env.observation_space.shape[0] if env.observation_space.shape else 1
rb = ReplayBuffer(100,dim_obs,dim_actions)
for _ in range(1000):
    #env.render()
    a = env.action_space.sample()
    sp, r, d, _ = env.step(a) # take a random action
    rb.add_sample(s,a,r,sp,d)
    s = env.reset() if d else sp
env.close()