import gym, torch
import numpy as np
from utils.buffer import ReplayBuffer
from utils.distributions import log_transition_probs, log_rew_probs
from utils.misc import eval_policy
from .mb_models import TransitionNet, RewardNet
from .mb_policies import RandomShooting, CrossEntropy

env = gym.make('Pendulum-v0')
#env = gym.make('SemisuperPendulumNoise-v0')
if env.action_space.shape:
    dim_actions = env.action_space.shape[0]
    discrete_actions = False
else:
    dim_actions = env.action_space.n
    discrete_actions = True
dim_obs = env.observation_space.shape[0] if env.observation_space.shape else 1

print(dim_actions, dim_obs)

buffer_size = 10000
if env.action_space.shape:
    rb = ReplayBuffer(buffer_size,dim_obs,dim_actions)
else:
    rb = ReplayBuffer(buffer_size,dim_obs,1)

num_epochs = 5000
global_iters = 0
num_train_steps = 30
train_iters = 50

num_traj = 200
traj_length = 12
num_iters = 10
elite_frac = 0.2
smoothing = 0.00

max_ep_length = 200
random_episodes = 10

if env.action_space.shape:
    init_action_dist = torch.distributions.MultivariateNormal(torch.zeros(dim_actions),torch.from_numpy((env.action_space.high-env.action_space.low)*0.2)*torch.eye(dim_actions))
    action_dist = [init_action_dist]*(traj_length-1)
else:
    init_action_dist = torch.distributions.Categorical(logits=torch.ones(env.action_space.n))
    action_dist = [init_action_dist]*(traj_length-1)


policy = CrossEntropy(None, None, action_dist, num_traj, traj_length, num_iters, elite_frac, dim_obs, dim_actions, None, None, None, smoothing, None, true_dyn=True, env=env)

t_losses = np.array([])
r_losses = np.array([])
rewards = []

for epoch in range(num_epochs):
    action_dist = [init_action_dist]*(traj_length-1)
    s, d, ep_rew = env.reset(), False, 0.
    dyn_error, rew_error = 0, 0
    ep_step = 0
    while not d and ep_step < max_ep_length:
        #print(env.state)
        action_dist = policy.new_action_dist(s)
        a = action_dist.pop(0).sample()
        #a = action_dist.pop(0).sample()
           
        if env.action_space.shape:
            sp, r, d, _ = env.step(a) # take a random action
        else:
            pass
            #sp, r, d, _ = env.step(int(a)) # take a random action
            
        #print(sp,a,r)
        
        action_dist = [torch.distributions.MultivariateNormal(dist.mean,10.*dist.covariance_matrix) for dist in action_dist]
        action_dist.append(init_action_dist)
        policy.set_action_dist(action_dist)
        
        print(r,d)
        ep_rew += r
        global_iters += 1
        ep_step += 1
        s = sp
    rewards.append(ep_rew)
    print(ep_rew)
    print('this really is the ep_rew, damn')
env.close()