import gym, torch, meta_env
import numpy as np
from utils.buffer import ReplayBuffer
from belief_models import SingleEncoder, TransitionNet, RewardNet
from utils.distributions import get_cov_mat, log_transition_probs, log_rew_probs, product_of_gaussians

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#setup environment
env = gym.make('MetaPendulum-v0')
if env.action_space.shape:
    dim_actions = env.action_space.shape[0]
    discrete_actions = False
else:
    dim_actions = env.action_space.n
    discrete_actions = True
dim_obs = env.observation_space.shape[0] if env.observation_space.shape else 1

#setup buffer
buffer_size = 10000
if env.action_space.shape:
    rb = ReplayBuffer(buffer_size,dim_obs,dim_actions)
else:
    rb = ReplayBuffer(buffer_size,dim_obs,1)

#training hyperparameters
num_epochs = 5000
global_iters = 0
num_train_steps = 1
max_logvar = 1.0
state_noise = 1e-4
rew_noise = 1e-4
max_variance = 1.0
random_episodes=num_epochs
max_ep_length = 300

#model hyperparameters
trans_cov_type='diag'
trans_hs=200

rew_hs=200

encoder_type = 'single'
encoder_cov_type='diag'
encoder_hs = 200
latent_dim=30
latent_prior = torch.distributions.MultivariateNormal(torch.zeros(latent_dim),torch.eye(latent_dim))

trans_net = TransitionNet(dim_obs,dim_actions,latent_dim,cov_type=trans_cov_type,hs=trans_hs).to(device)
rew_net = RewardNet(dim_obs,dim_actions,latent_dim,hs=rew_hs).to(device)
if encoder_type == 'single':
    encoder = SingleEncoder(dim_obs,dim_actions,latent_dim,cov_type=encoder_cov_type,hs=encoder_hs)
#TODO: code multi-step encoder [RNN]

#training parameters
t_learning_rate = 1e-3
t_optimizer = torch.optim.Adam(trans_net.parameters(),lr=t_learning_rate)

r_learning_rate = 1e-3
r_optimizer = torch.optim.Adam(rew_net.parameters(),lr=r_learning_rate)

q_learning_rate = 1e-3
q_optimizer = torch.optim.Adam(encoder.parameters(),lr=q_learning_rate)

batch_size = 50
batch_length = 50

#planner hyperparameters
num_traj = 1000
traj_length = 30
num_iters = 5
elite_frac = 0.1

#TODO: code planner

losses = np.array([])
rewards = []

for epoch in range(num_epochs):
    rb.new_episode()
    if epoch > 0:
        for step in range(num_train_steps):
            samps = rb.random_sequences(batch_size,batch_length)
            
            q_optimizer.zero_grad()
            t_optimizer.zero_grad()
            r_optimizer.zero_grad()
            
            kl_divs, log_ts, log_rs = torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(batch_size)
        
            batch_means, batch_precs, batch_prod_means, batch_prod_precs = [], [], [], []
            for i in range(batch_size):
                s,a_rb,r,sp = [torch.from_numpy(samps[i][k]).float() for k in ['o','a','r','op']]
                if discrete_actions:
                    a = torch.squeeze(torch.nn.functional.one_hot(torch.from_numpy(a_rb).long(),num_classes=dim_actions)).float()
                else:
                    a = a_rb
                q_ins = torch.cat((s,a,r,sp),axis=1)
                q_outs = encoder(q_ins)
                q_means = q_outs[:,:latent_dim]
                q_precs = torch.inverse(get_cov_mat(q_outs[:,latent_dim:],dim_obs,encoder_cov_type,device))
                batch_means.append(q_means)
                batch_precs.append(q_precs)
                means, precs = product_of_gaussians(q_means.view(1,-1,latent_dim),q_precs.view(1,-1,latent_dim,latent_dim))
                batch_prod_means.append(means)
                batch_prod_precs.append(precs)
                
            means, precs = torch.stack(batch_prod_means), torch.stack(batch_prod_precs)
            
            thetas = torch.squeeze(means + torch.matmul(torch.inverse(precs),torch.randn_like(means)))#shape: [BxL]
            for i in range(batch_size):
                s,a_rb,r,sp = [torch.from_numpy(samps[i][k]).float() for k in ['o','a','r','op']]
                if discrete_actions:
                    a = torch.squeeze(torch.nn.functional.one_hot(torch.from_numpy(a_rb).long(),num_classes=dim_actions)).float()
                else:
                    a = a_rb
                net_ins = torch.cat((s,a,thetas[i,:].expand(batch_length,latent_dim)),axis=1)
                t_outs = trans_net(net_ins)
                r_outs = rew_net(net_ins)
                t_means, t_covs = t_outs[:,:dim_obs], t_outs[:,dim_obs:]
                log_ts[i] = torch.sum(log_transition_probs(t_means,t_covs,sp,cov_type=trans_cov_type,device=device))
                log_rs[i] = torch.sum(log_rew_probs(r_outs[:,0],r_outs[:,1],r))
                q_dist = torch.distributions.MultivariateNormal(batch_means[i],precision_matrix=batch_precs[i])
                kl_divs[i] = torch.sum(torch.distributions.kl.kl_divergence(q_dist,latent_prior.expand([batch_length])))
            
            loss = torch.mean(kl_divs - log_ts - log_rs)
            
            loss.backward()
            
            q_optimizer.step()
            t_optimizer.step()
            r_optimizer.step()
        losses = np.append(losses,loss.cpu().detach().numpy())
        print(loss)

    s, d, ep_rew = env.reset(), False, 0.
    dyn_error, rew_error = 0, 0
    ep_step = 0
    while not d and ep_step < max_ep_length:
        if epoch < random_episodes:
            a = np.array(env.action_space.sample())
        #else:
            #a = policy.get_action(s)
        sp, r, d, _ = env.step(a) # take a random action
        s_n = s+np.random.multivariate_normal(np.zeros(dim_obs),state_noise*np.eye(dim_obs))
        sp_n = sp+np.random.multivariate_normal(np.zeros(dim_obs),state_noise*np.eye(dim_obs))
        r_n = r+np.random.normal(0.,rew_noise)
        rb.add_sample(s_n,a,r_n,sp_n,d)
                    
        ep_rew += r
        global_iters += 1
        ep_step += 1
        s = sp
    rewards.append(ep_rew)
env.close()