import gym, torch, meta_env
import numpy as np
from utils.buffer import ReplayBuffer
from belief_models import LSTMEncoder, TransitionNet, RewardNet
from utils.distributions import get_cov_mat, log_transition_probs, log_rew_probs, product_of_gaussians, gaussian_product_posterior
from belief_policies import LSTMRandomShooting

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
num_train_steps = 20
max_logvar = 10.
state_noise = 1e-3
rew_noise = 1e-3
random_episodes=0
max_ep_length = 200
episode_repeat = 3

#model hyperparameters
trans_cov_type='diag'
trans_hs=200

rew_hs=200

encoder_type = 'single'
encoder_cov_type='diag'
encoder_hs = 200
encoder_num_layers = 1
latent_dim=30
latent_prior = torch.distributions.MultivariateNormal(torch.zeros(latent_dim),torch.eye(latent_dim))
test_batch_size = 100

trans_net = TransitionNet(dim_obs,dim_actions,latent_dim,cov_type=trans_cov_type,hs=trans_hs).to(device)
rew_net = RewardNet(dim_obs,dim_actions,latent_dim,hs=rew_hs).to(device)
encoder = LSTMEncoder(dim_obs,dim_actions,latent_dim,cov_type=encoder_cov_type,hs=encoder_hs,num_layers=encoder_num_layers)

#training parameters
t_learning_rate = 1e-3
t_optimizer = torch.optim.Adam(trans_net.parameters(),lr=t_learning_rate)

r_learning_rate = 1e-2
r_optimizer = torch.optim.Adam(rew_net.parameters(),lr=r_learning_rate)

q_learning_rate = 1e-3
q_optimizer = torch.optim.Adam(encoder.parameters(),lr=q_learning_rate)

#planner hyperparameters
num_traj = 300
traj_length = 20
num_iters = 5
elite_frac = 0.1

batch_size = 50
batch_length = traj_length

if env.action_space.shape:
    init_action_dist = torch.distributions.MultivariateNormal(torch.zeros(dim_actions),torch.from_numpy((env.action_space.high-env.action_space.low)**2)*torch.eye(dim_actions))
    action_dist = [init_action_dist]*(traj_length-1)
else:
    init_action_dist = torch.distributions.Categorical(logits=torch.ones(env.action_space.n))
    action_dist = [init_action_dist]*(traj_length-1)
    
policy = LSTMRandomShooting(trans_net, rew_net, encoder, init_action_dist, num_traj, traj_length, dim_obs, dim_actions, latent_dim, trans_cov_type, False, encoder_cov_type, max_logvar, device, det=False)
losses = np.array([])
rewards = []

for epoch in range(num_epochs):
    if epoch % episode_repeat == 0:
        print('new params!')
        m, l = np.random.uniform(0.1,50.), np.random.uniform(0.5,10.)
        env.set_params(m,l)
    rb.new_episode()
    if epoch > 0:
        for step in range(num_train_steps):
            samps = rb.random_sequences(batch_size,batch_length)
            
            q_optimizer.zero_grad()
            t_optimizer.zero_grad()
            r_optimizer.zero_grad()
            
            kl_divs, log_ts, log_rs = torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(batch_size)
        
            #assuming all sequences are full-length
            s,a_rb,r,sp = [torch.stack([torch.from_numpy(samp[k]).float() for samp in samps]) for k in ['o','a','r','op']]
            
            if discrete_actions:
                a = torch.squeeze(torch.nn.functional.one_hot(torch.from_numpy(a_rb).long(),num_classes=dim_actions)).float()
            else:
                a = a_rb
            
            q_ins = torch.cat((s,a,r,sp),axis=2)
            q_outs, _ = encoder(q_ins)
            q_means = torch.squeeze(q_outs)[:,:,:latent_dim]
            q_precs = torch.inverse(get_cov_mat(q_outs[:,:,latent_dim:],dim_obs,encoder_cov_type,device))
                
            post_means, post_precs = product_of_gaussians(q_means,q_precs)
            products_zipped = [product_of_gaussians(q_means[:,:t],q_precs[:,:t,:,:]) for t in range(1,batch_length+1)]
            stepwise_means = torch.squeeze(torch.stack([a[0] for a in products_zipped],axis=1))
            stepwise_precs = torch.stack([a[1] for a in products_zipped],axis=1)
            
            thetas = (post_means + torch.matmul(torch.inverse(post_precs),torch.randn_like(post_means))).view(batch_size,1,-1)#shape: [BxL]
            
            net_ins = torch.cat((s,a,thetas.expand(batch_size,batch_length,latent_dim)),axis=2)
            t_outs = trans_net(net_ins)
            r_outs = rew_net(net_ins)
            r_means, r_covs = r_outs[:,:,0], torch.clamp(r_outs[:,:,1],-max_logvar,max_logvar)
            t_means, t_covs = t_outs[:,:,:dim_obs], torch.clamp(t_outs[:,:,dim_obs:],-max_logvar,max_logvar)
            log_ts = torch.sum(log_transition_probs(t_means,t_covs,sp,cov_type=trans_cov_type,device=device),1)
            t_mse = torch.nn.MSELoss()(t_means,sp)
            log_rs = -torch.sum(torch.nn.MSELoss(reduction='none')(r_outs[:,:,0],torch.squeeze(r)),1)
        
            prior_dist = torch.distributions.MultivariateNormal(stepwise_means[:,:-1,:],precision_matrix=stepwise_precs[:,:-1,:,:])
            post_dist = torch.distributions.MultivariateNormal(stepwise_means[:,1:,:],precision_matrix=stepwise_precs[:,1:,:,:])
            kl_divs = torch.sum(torch.distributions.kl.kl_divergence(post_dist,prior_dist))
            
            loss = torch.mean(kl_divs - log_ts - log_rs)
            
            loss.backward()
            
            q_optimizer.step()
            t_optimizer.step()
            r_optimizer.step()
            print(step)
        losses = np.append(losses,loss.cpu().detach().numpy())
        print(torch.mean(kl_divs),torch.mean(log_ts),torch.mean(log_rs))
        print('mse: ', t_mse)
        print(loss)

    s, d, ep_rew = env.reset(), False, 0.
    dyn_error, rew_error = 0, 0
    ep_step = 0
    latent_mean, latent_prec = torch.zeros(latent_dim).view(1,latent_dim,1), torch.eye(latent_dim).view(1,latent_dim,latent_dim)
    hidden = None
    while not d and ep_step < max_ep_length:
        
        if epoch < random_episodes or epoch % episode_repeat == 0:
            a = np.array(env.action_space.sample())
        else:
            if hidden is None:
                a = policy.get_action(s,hidden)
            else:
                a = policy.get_action(s,[h.expand(encoder_num_layers,num_traj,encoder_hs) for h in hidden])
        
        sp, r, d, _ = env.step(a) # take a random action
        s_n = s+np.random.multivariate_normal(np.zeros(dim_obs),state_noise*np.eye(dim_obs))
        sp_n = sp+np.random.multivariate_normal(np.zeros(dim_obs),state_noise*np.eye(dim_obs))
        r_n = r+np.random.normal(0.,rew_noise)
        rb.add_sample(s_n,a,r_n,sp_n,d)
        
        codes = torch.squeeze(latent_mean.expand(test_batch_size,latent_dim,1) + torch.matmul(torch.inverse(latent_prec).expand(test_batch_size,latent_dim,latent_dim),torch.randn(test_batch_size,latent_dim,1)))
        s_torch = torch.from_numpy(np.array(s)).float().expand(test_batch_size,dim_obs)
        a_torch = torch.from_numpy(np.array(a)).float().expand(test_batch_size,dim_actions)
        net_ins = torch.cat((s_torch,a_torch,codes),axis=1)
        
        sp_hats = trans_net(net_ins)[:,:dim_obs]
        r_hats = rew_net(net_ins)[:,0]
        
        t_err = torch.mean(torch.nn.MSELoss()(torch.from_numpy(sp).float().expand(test_batch_size,dim_obs),sp_hats))
        r_err = torch.mean(torch.nn.MSELoss()(torch.from_numpy(np.array([r])).float().expand(test_batch_size,1),r_hats))
        
        dyn_error += t_err
        rew_error += r_err
        
        q_in = torch.cat([torch.from_numpy(np.array(k)).float() for k in [s,a,[r],sp]]).view(1,1,-1)
        q_out, hidden = encoder(q_in,hidden)
        new_mean, new_prec = torch.squeeze(q_out)[:latent_dim].view(1,latent_dim,1), torch.inverse(get_cov_mat(torch.squeeze(q_out)[latent_dim:],dim_obs,encoder_cov_type,device)).view(1,latent_dim,latent_dim)     
        latent_mean, latent_prec = gaussian_product_posterior(latent_mean,latent_prec,new_mean,new_prec)
        
        if ep_step == 5:
            print('trans: ', sp, torch.mean(sp_hats,0))
            print('r: ', r, torch.mean(r_hats,0))    
        
        ep_rew += r
        global_iters += 1
        ep_step += 1
        s = sp
    rewards.append(ep_rew)
    print('ep_rew: ', ep_rew)
    print('traj errors: ', dyn_error,rew_error)
    #print(t_err,r_err)
env.close()