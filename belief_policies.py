import torch
import numpy as np
from utils.distributions import get_batch_mvnormal, get_cov_mat, gaussian_product_posterior

class ScenarioRandomShooting():
    """implements random shooting for MBRL"""
    def __init__(self, transition, reward, action_dist, num_traj, traj_length, ns, na, num_codes, latent_dim, t_cov_type, r_cov, max_logvar, device, det=False):
        self._transition = transition
        self._reward = reward
        self._action_dist = action_dist
        self._num_traj = num_traj
        self._traj_length = traj_length
        self._ns = ns
        self._na = na
        self._latent_dim = latent_dim
        self._t_cov_type = t_cov_type
        self._r_cov = r_cov
        self._max_logvar = max_logvar
        self._device = device
        self._num_codes = num_codes
        self._det = det
        
    def get_action(self, state,codes):
        states = torch.zeros([self._num_traj,self._num_codes,self._ns,self._traj_length]).to(self._device)
        states[:,:,:,0] = torch.from_numpy(state).float().to(self._device)
        if type(self._action_dist) == torch.distributions.Categorical:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,1,self._traj_length-1).to(self._device)
        else:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,self._na,self._traj_length-1).to(self._device)
        rewards = torch.zeros([self._num_traj,self._num_codes,self._traj_length-1])
        
        for i in range(self._traj_length-1):
            if type(self._action_dist) == torch.distributions.Categorical:
                actions_input = torch.squeeze(torch.nn.functional.one_hot(actions[:,:,i]).float())
            else:
                actions_input = actions[:,:,i]
            states[:,:,:,i+1], rewards[:,:,i] = self._next_state_rew(states[:,:,:,i],actions_input,codes)
                
        total_reward = torch.sum(torch.mean(rewards,1),-1)
        best_traj = torch.argmax(total_reward)
        return actions[best_traj,:,0].cpu().numpy()
        
        
    def _next_state_rew(self, states, actions, codes):
        """helper function to unroll dynamics (batched)"""
        ins = torch.cat((states,actions.unsqueeze(1).expand(self._num_traj,self._num_codes,self._na),codes.unsqueeze(0).expand(self._num_traj,self._num_codes,self._latent_dim)),axis=-1).view(-1,self._na+self._ns+self._latent_dim).to(self._device)
        t_outs = self._transition(ins)
        t_means, t_covs = t_outs[:,:self._ns], t_outs[:,self._ns:]
        t_covs_clamped = torch.clamp(t_covs,-self._max_logvar,self._max_logvar).to(self._device)
        cov_mat = get_cov_mat(t_covs,self._ns,self._t_cov_type,self._device)
        r_outs = self._reward(ins)
        if self._det:
            sp = t_means
            rews = r_outs
        else:
            sp = t_means.to(self._device) + torch.squeeze(torch.matmul(cov_mat,torch.randn_like(t_means).view(-1,self._ns,1)))
            
            if self._r_cov:
                rews = torch.distributions.Normal(r_outs[:,0],torch.exp(r_outs[:,1])).rsample()
            else:
                rews = r_outs[:,0]
        return (sp.view(self._num_traj,self._num_codes,self._ns), torch.squeeze(rews).view(self._num_traj,self._num_codes))
    
class LSTMRandomShooting():
    """implements random shooting for RNNEncoder"""
    def __init__(self, transition, reward, encoder, action_dist, num_traj, traj_length, ns, na, latent_dim, t_cov_type, r_cov, q_cov_type, max_logvar, device, det=False, elite_frac=0.1):
        self._transition = transition
        self._reward = reward
        self._encoder = encoder
        self._action_dist = action_dist
        self._num_traj = num_traj
        self._traj_length = traj_length
        self._ns = ns
        self._na = na
        self._latent_dim = latent_dim
        self._t_cov_type = t_cov_type
        self._r_cov = r_cov
        self._q_cov_type = q_cov_type
        self._max_logvar = max_logvar
        self._device = device
        self._num_elite = round(elite_frac*num_traj)
        self._det = det
        
    def get_action(self,state,hidden,code_means,code_precs):
        states = torch.zeros([self._num_traj,self._ns,self._traj_length]).to(self._device)
        states[:,:,0] = torch.from_numpy(state).float().to(self._device)
        if type(self._action_dist) == torch.distributions.Categorical:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,1,self._traj_length-1)
        else:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,self._na,self._traj_length-1).to(self._device)
        rewards = torch.zeros([self._num_traj,self._traj_length-1]).to(self._device)
        for i in range(self._traj_length-1):
            if type(self._action_dist) == torch.distributions.Categorical:
                actions_input = torch.squeeze(torch.nn.functional.one_hot(actions[:,:,i]).float())
            else:
                actions_input = actions[:,:,i]
            
            codes = code_means.expand(self._num_traj,self._latent_dim,1).cuda()+torch.matmul(torch.inverse(code_precs).cuda(),torch.randn(self._num_traj,self._latent_dim,1).cuda())
            codes = codes.view(self._num_traj,self._latent_dim)
            
            states[:,:,i+1], rewards[:,i] = self._next_state_rew(states[:,:,i],actions_input,codes)
            
            q_ins = torch.cat((states[:,:,i],actions[:,:,i],rewards[:,i].unsqueeze(1),states[:,:,i+1]),axis=1).view(self._num_traj,1,-1).to(self._device)
            q_outs, hidden = self._encoder(q_ins,hidden)
            new_means = torch.squeeze(q_outs)[:,:self._latent_dim].unsqueeze(-1) 
            new_precs = torch.inverse(get_cov_mat(torch.squeeze(q_outs)[:,self._latent_dim:],self._ns,self._q_cov_type,self._device))
            code_means, code_precs = gaussian_product_posterior(code_means,code_precs,new_means,new_precs)
                
        total_reward = torch.sum(rewards,1)
        elite_indices = torch.argsort(total_reward,descending=True)[:self._num_elite]
        #best_traj = torch.argmax(total_reward)
        return torch.mean(actions[elite_indices,:,0],0).cpu().numpy()
        
        
    def _next_state_rew(self, states, actions, codes):
        """helper function to unroll dynamics (batched)"""
        #print(states.shape,actions.shape,codes.shape)
        ins = torch.cat((states,actions,codes),axis=1).to(self._device)
        t_outs = self._transition(ins)
        t_means, t_covs = t_outs[:,:self._ns], t_outs[:,self._ns:]
        t_covs_clamped = torch.clamp(t_covs,-self._max_logvar,self._max_logvar).to(self._device)
        cov_mat = get_cov_mat(t_covs,self._ns,self._t_cov_type,self._device)
        r_outs = self._reward(ins)
        if self._det:
            sp = t_means
            rews = r_outs
        else:
            sp = t_means.to(self._device) + torch.squeeze(torch.matmul(cov_mat,torch.randn_like(t_means).view(self._num_traj,self._ns,1)))
            
            if self._r_cov:
                rews = torch.distributions.Normal(r_outs[:,0],torch.exp(r_outs[:,1])).rsample()
            else:
                rews = r_outs[:,0]
        return (sp, torch.squeeze(rews))
    
    def rollout(self,state,hidden,action_seq,roll_length,num_roll,init_code_mean,
                init_code_prec,policy=None,resample_belief=False):
        
        states = torch.zeros([num_roll,self._ns,roll_length]).to(self._device)
        states[:,:,0] = torch.from_numpy(state).float().to(self._device)
        
        if policy:
            actions = torch.zeros(num_roll,self._na,roll_length).to(self._device)
        else:
            actions = action_seq
        
        rewards = torch.zeros([num_roll,roll_length-1]).to(self._device)
        
        if resample_belief:
            code_means = torch.zeros(num_roll,roll_length,self._latent_dim,1)
            code_precs = torch.zeros(num_roll,roll_length,self._latent_dim,self._latent_dim)
            code_means[:,0,:,:] = init_code_mean
            code_precs[:,0,:,:] = init_code_prec
            codes = torch.zeros(num_roll,self._latent_dim,roll_length)
            codes[:,:,0] = init_code_mean.expand(num_roll,self._latent_dim) + \
                torch.squeeze(torch.matmul(
                torch.inverse(init_code_prec).expand(num_roll,self._latent_dim,self._latent_dim),
                torch.randn(num_roll,self._latent_dim,1)))
        else:
            codes = init_code_mean.expand(num_roll,self._latent_dim) \
                + torch.squeeze(torch.matmul(
                    torch.inverse(init_code_prec).expand(num_roll,self._latent_dim,self._latent_dim),
                    torch.randn(num_roll,self._latent_dim,1)))
            codes = codes.expand(num_roll,self._latent_dim,roll_length)
        
        for i in range(self._traj_length-1):
            if policy:
                actions[:,:,i] = policy(states[:,:,i])
            if resample_belief:
                pass
                                
            
            states[:,:,i+1], rewards[:,i] = self._next_state_rew(states[:,:,i],actions[:,:,i],codes[:,:,i])
            
            q_ins = torch.cat((states[:,:,i],actions[:,:,i],rewards[:,i].unsqueeze(1),states[:,:,i+1]),axis=1).view(self._num_traj,1,-1).to(self._device)
            q_outs, hidden = self._encoder(q_ins,hidden)
            new_means = torch.squeeze(q_outs)[:,:self._latent_dim].unsqueeze(-1) 
            new_precs = torch.inverse(get_cov_mat(torch.squeeze(q_outs)[:,self._latent_dim:],self._ns,self._q_cov_type,self._device))
            code_means[:,i+1,:,:], code_precs[:,i+1,:,:] = gaussian_product_posterior(code_means,code_precs,new_means,new_precs)
                
        return states, rewards, code_means, code_precs
    
    
class CrossEntropy():
    """implements cross-entropy method for MBRL"""
    def __init__(self, transition, reward, action_dists, num_traj, traj_length, num_iters, elite_frac, ns, na, t_cov_type, r_cov, max_logvar, device):
        self._transition = transition
        self._reward = reward
        self._action_dists = action_dists #list of action distributions
        self._num_traj = num_traj
        self._traj_length = traj_length
        self._num_iters = num_iters
        self._elite_frac = elite_frac
        self._num_elite = round(elite_frac*num_traj)
        self._ns = ns
        self._na = na
        self._t_cov_type = t_cov_type
        self._r_cov = r_cov
        self._max_logvar = max_logvar
        self._device = device
        if type(self._action_dists[0]) == torch.distributions.Categorical:
            self._discrete_action = True
        else:
            self._discrete_action = False
        
    def new_action_dist(self, state):
        curr_action_dist = self._action_dists
        for iteration in range(self._num_iters):
            states = torch.zeros([self._num_traj,self._ns,self._traj_length])
            states[:,:,0] = torch.from_numpy(state).float().to(self._device)
            #sample action distribution
            if self._discrete_action:
                actions = torch.cat([dist.expand((self._num_traj,1)).sample() for dist in curr_action_dist],axis=1)
            else:
                actions = torch.cat([dist.expand((self._num_traj,self._na)).rsample() for dist in curr_action_dist],axis=1).view(self._num_traj,self._na,self._traj_length-1)
            #shoot trajectories forward, get rewards
            rewards = torch.zeros([self._num_traj,self._traj_length-1])
            for i in range(self._traj_length-1):
                states[:,:,i+1], rewards[:,i] = self._next_state_rew(states[:,:,i],actions[:,:,i])
            #take elite fraction, refit action distributions
            total_rew = torch.sum(rewards,1)
            elite_indices = torch.argsort(total_rew,descending=True)[:self._num_elite]
            curr_action_dist = self._fit_action_dists(actions[elite_indices,:,:])
        return curr_action_dist
    
    def set_action_dist(self,action_dist):
        self._action_dist = action_dist
        
    def _next_state_rew(self, states, actions):
        if self._discrete_action:
            pass
        else:
            ins = torch.cat((states,actions),axis=1).to(self._device)
            t_out = self._transition(ins)
            r_outs = self._reward(ins)
            
            t_means, t_covs = t_out[:,:self._ns], t_out[:,self._ns:]
            t_covs_clamped = torch.clamp(t_covs,-self._max_logvar,self._max_logvar).to(self._device)
            cov_mat = get_cov_mat(t_covs_clamped,self._ns,self._t_cov_type,self._device)
            
            sp = t_means + torch.squeeze(torch.matmul(cov_mat,torch.randn_like(t_means).view(self._num_traj,self._ns,1)))
            
            if self._r_cov:
                r_logvar_clamped = torch.clamp(r_outs[:,1],-self._max_logvar,self._max_logvar).to(self._device)
                rews = torch.distributions.Normal(r_outs[:,0],torch.exp(r_logvar_clamped)).rsample()
            else:
                rews = r_outs
            
            return (sp, torch.squeeze(rews))
            
    
    def _fit_action_dists(self,elite_actions):
        action_dists = []
        if self._discrete_action:
            pass
        else: #assume train of Gaussian action distributions
            #print('actions: ', elite_actions)
            action_means = torch.mean(elite_actions,0)
            action_variances = torch.var(elite_actions,0)
            #print('means: ', action_means)
            #print('variances: ', action_variances)
            action_dists = [torch.distributions.MultivariateNormal(action_means[:,t],action_variances[:,t]*torch.eye(self._na)) for t in range(self._traj_length-1)]
            #for t in range(self._traj_length-1):
                #action_mean = torch.mean(elite_actions[:,:,t],0)
                #action_variance = torch.var(elite_actions[:,:,t]-action_mean,0)
                #action_dists.append(torch.distributions.MultivariateNormal(action_mean,action_variance*torch.eye(self._na)))
        return action_dists
                