import gym, torch
import numpy as np
from utils.buffer import ReplayBuffer
from utils.distributions import log_transition_probs, log_rew_probs, get_cov_mat
from utils.misc import eval_policy
from .mb_models import TransitionNet, RewardNet
from .mb_policies import RandomShooting, CrossEntropy, Scenario

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
num_train_steps = 50
train_iters = 50

trans_cov_type='scalar'
rew_cov = False
trans_hs=300
rew_hs=20

max_logvar = 10.

state_noise = 1e-4
rew_noise = 1e-4

trans_net = TransitionNet(dim_obs,dim_actions,cov_type=trans_cov_type,hs=trans_hs).to(device)
rew_net = RewardNet(dim_obs,dim_actions,cov=rew_cov,hs=rew_hs).to(device)
max_variance = 1.0

t_learning_rate = 1e-3
t_optimizer = torch.optim.Adam(trans_net.parameters(),lr=t_learning_rate)

r_learning_rate = 2e-3
r_optimizer = torch.optim.Adam(rew_net.parameters(),lr=r_learning_rate)
batch_size = 256

num_traj = 200
#traj_length = 12
traj_length = 50
num_iters = 10

a_lr = 1.
init_action_dist = torch.cat((torch.zeros((dim_actions,traj_length-1)),
    torch.ones((dim_actions,traj_length-1))),axis=0).data.to(device)
action_dist = torch.autograd.Variable(init_action_dist,requires_grad=True)
action_cov_type = 'diag'
a_opt = torch.optim.Adam([action_dist],lr=a_lr)

max_ep_length = 200
random_episodes = 1

if env.action_space.shape:
    init_action_dist = torch.distributions.MultivariateNormal(torch.zeros(dim_actions),torch.from_numpy((env.action_space.high-env.action_space.low)*1.)*torch.eye(dim_actions))
else:
    init_action_dist = torch.distributions.Categorical(logits=torch.ones(env.action_space.n))

#policy = Scenario(trans_net, rew_net, action_dist, a_opt, num_traj, traj_length, num_iters, dim_obs, dim_actions, trans_cov_type, rew_cov, action_cov_type, max_logvar, device)
policy = Scenario(trans_net, rew_net, action_dist, a_opt, num_traj, traj_length, num_iters, dim_obs, dim_actions, trans_cov_type, rew_cov, action_cov_type, max_logvar, device,true_dyn=True,env=env)

t_losses = np.array([])
r_losses = np.array([])
rewards = []

for epoch in range(num_epochs):
    if (rb.size() >= batch_size):
        for step in range(num_train_steps):
            t_optimizer.zero_grad()
            samps = rb.random_batch(batch_size)
            if discrete_actions:
                a_one_hot = torch.squeeze(torch.nn.functional.one_hot(torch.from_numpy(samps['a']).long())).float()
                ins = torch.cat((torch.from_numpy((samps['o'])).float(),a_one_hot),axis=1).to(device)
            else:
                ins = torch.cat((torch.from_numpy((samps['o'])).float(),torch.from_numpy(samps['a']).float()),axis=1).to(device)
            t_outs = trans_net(ins)
            t_means, t_covs = t_outs[:,:dim_obs], t_outs[:,dim_obs:]
            t_covs = torch.clamp(t_covs,-max_logvar,max_logvar)
            #t_loss = torch.nn.MSELoss()(t_means,torch.from_numpy(samps['op']).float())
            t_loss = torch.mean(-log_transition_probs(t_means,t_covs,torch.from_numpy(samps['op']-samps['o']).float().to(device),cov_type=trans_cov_type))
            t_loss.backward()
            #torch.nn.utils.clip_grad_norm(trans_net.parameters(),0.1)
            t_optimizer.step()
            t_losses = np.append(t_losses, t_loss.cpu().data.numpy())

            #train reward model
            r_optimizer.zero_grad()
            r_outs = rew_net(ins)
            #r_loss = torch.nn.MSELoss()(r_outs,torch.from_numpy(samps['r']).float().to(device))
            #r_loss = torch.mean(torch.cdist(r_outs,torch.from_numpy(samps['r']).float().to(device),p=1))
            r_loss = torch.mean(torch.norm(r_outs-torch.from_numpy(samps['r']).float().to(device),p=1))
            #torch.nn.utils.clip_grad_norm(rew_net.parameters(),0.1)
            r_loss.backward()
            r_optimizer.step()
            r_losses = np.append(r_losses, r_loss.cpu().data.numpy())
    if r_losses.size != 0:
        print(t_losses[-1],r_losses[-1])
    action_dist = [init_action_dist]*(traj_length-1)
    s, d, ep_rew = env.reset(), False, 0.
    dyn_error, rew_error = 0, 0
    ep_step = 0
    while not d and ep_step < max_ep_length:
        #new_action_dist = policy.new_action_dist(np.array(s))
        #if discrete_actions:
            #a = new_action_dist.pop(0).sample().cpu().numpy()
        #else:
            #a = policy.new_action_dist(np.array(s))[0].rsample()
            #a = new_action_dist.pop(0).rsample().cpu().numpy()
            #a = new_action_dist[0].mean.cpu().numpy()
        #a = env.action_space.sample()
        
        if epoch < random_episodes:
            a = np.array(env.action_space.sample())
        else:
            #print(s)
            action_dist = policy.new_action_dist(s)
            a = action_dist[:dim_actions,0]
            a = a + torch.matmul(get_cov_mat(action_dist[dim_actions:,0].view(1,-1,dim_actions),dim_actions,action_cov_type,device),torch.randn_like(a))
            a = a.view(dim_actions).detach().cpu().numpy()

            print(r)
            new_action_dist = torch.cat((action_dist[:dim_actions,1].view(dim_actions,1),
                                                                 torch.ones((dim_actions,1),requires_grad=True).to(device)),axis=0)
            action_dist = torch.autograd.Variable(torch.cat((action_dist[:,1:],new_action_dist),axis=1),requires_grad=True)
            policy.set_action_dist(action_dist)
        
        if env.action_space.shape:
            sp, r, d, _ = env.step(a) # take a random action
        else:
            sp, r, d, _ = env.step(int(a)) # take a random action
        s_n = s+np.random.multivariate_normal(np.zeros(dim_obs),state_noise*np.eye(dim_obs))
        sp_n = sp+np.random.multivariate_normal(np.zeros(dim_obs),state_noise*np.eye(dim_obs))
        r_n = r+np.random.normal(0.,rew_noise)
        rb.add_sample(s_n,a,r_n,sp_n,d)
        #if epoch >= random_episodes:
            #print('sp: ', sp, ' r: ', r)
        
        if discrete_actions:
            a_one_hot = torch.squeeze(torch.nn.functional.one_hot(torch.from_numpy(a).long(),num_classes=dim_actions)).float()
            net_in = torch.cat((torch.from_numpy(s).float(),a_one_hot)).view(1,dim_obs+dim_actions).to(device)
        else:
            net_in = torch.cat((torch.from_numpy(s).float(),torch.from_numpy(a).float())).view(1,dim_obs+dim_actions).to(device)
        t_out = torch.squeeze(trans_net(net_in))
        t_mean, t_cov =  t_out[:dim_obs], t_out[dim_obs:]
        r_out = torch.squeeze(rew_net(net_in))
        
        dyn_error += torch.nn.MSELoss()(t_mean,torch.from_numpy(sp-s).float().to(device))
        rew_error += torch.nn.MSELoss()(r_out,torch.from_numpy(np.array(r)).float().to(device))
        
        s = sp
                    
        ep_rew += r
        global_iters += 1
        ep_step += 1
    rewards.append(ep_rew)
    print(rewards[-1],dyn_error,rew_error)
env.close()