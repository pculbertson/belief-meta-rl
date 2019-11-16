import gym, torch
import numpy as np
from utils import ReplayBuffer, log_transition_probs
from models import TransitionNet, RewardNet

env = gym.make('CartPole-v1')
dim_actions = env.action_space.shape[0] if env.action_space.shape else 1 #if discrete return dim = 1
dim_obs = env.observation_space.shape[0] if env.observation_space.shape else 1

buffer_size = 10000
rb = ReplayBuffer(buffer_size,dim_obs,dim_actions)

num_epochs = 1000
global_iters = 0

trans_cov_type='dense'
rew_cov = True

trans_net = TransitionNet(dim_obs,dim_actions,cov_type=trans_cov_type)
rew_net = RewardNet(dim_obs,dim_actions,cov=rew_cov)

learning_rate = 1e-5
t_optimizer = torch.optim.Adam(trans_net.parameters(),lr=learning_rate)
batch_size = 32

print_freq = 100

for _ in range(num_epochs):
    s, d = env.reset(), False
    while not d:
        a = env.action_space.sample()
        sp, r, d, _ = env.step(a) # take a random action
        rb.add_sample(s,a,r,sp,d)
        s = env.reset() if d else sp
        
        if rb.size() >= batch_size:
            t_optimizer.zero_grad()
            samps = rb.random_batch(batch_size)
            ins = torch.from_numpy(np.concatenate((samps['o'],samps['a']),axis=1)).float()
            t_outs = trans_net(ins)
            t_means, t_covs = t_outs[:,:dim_obs], t_outs[:,dim_obs:]
            t_loss = torch.mean(-log_transition_probs(t_means,t_covs,torch.from_numpy(samps['op']).float(),cov_type=trans_cov_type))
            t_loss.backward()
            t_optimizer.step()
            
            if global_iters % print_freq == 0:
                print(t_loss)
                
        global_iters += 1
        
env.close()