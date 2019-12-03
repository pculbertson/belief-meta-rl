import torch
import numpy as np
from utils.distributions import get_batch_mvnormal, get_cov_mat, gaussian_product_posterior

class RandomShooting():
    """implements random shooting for MBRL"""
    def __init__(self, transition, reward, encoder, action_dist, num_traj, traj_length, ns, na, latent_dim, t_cov_type, r_cov, q_cov_type, max_logvar, device, code_type, det=False):
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
        self._code_type = code_type
        self._det = det
        
    def get_action(self, state):
        states = torch.zeros([self._num_traj,self._ns,self._traj_length])
        states[:,:,0] = torch.from_numpy(state).float().to(self._device)
        if type(self._action_dist) == torch.distributions.Categorical:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,1,self._traj_length-1)
        else:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,self._na,self._traj_length-1)
        rewards = torch.zeros([self._num_traj,self._traj_length-1])
        if self._code_type == 'thompson':
            #encoder is single code pre-sampled for episode
            codes = self._encoder.expand([self._num_traj,self.latent_dim,self._traj_length-1])
        elif self._code_type == 'scenarios':
            #encoder is code distribution for current time, sample a set of codes
            codes = self._encoder.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,self._latent_dim,self._traj_length)
        elif self._code_type == 'resample':
            code_means, code_precs = torch.zeros(self._num_traj,self._latent_dim), torch.eye(self._latent_dim).expand(self._num_traj,self._latent_dim,self._latent_dim)
        for i in range(self._traj_length-1):
            if type(self._action_dist) == torch.distributions.Categorical:
                actions_input = torch.squeeze(torch.nn.functional.one_hot(actions[:,:,i]).float())
            else:
                actions_input = actions[:,:,i]
            
            if self._code_type == 'resample':
                #encoder is network, resample codes @ each time with new data
                codes = code_means+torch.squeeze(torch.matmul(torch.inverse(code_precs),torch.randn_like(code_means).unsqueeze(2)))
            
            states[:,:,i+1], rewards[:,i] = self._next_state_rew(states[:,:,i],actions_input,codes)
            
            if self._code_type == 'resample':
                q_ins = torch.cat((states[:,:,i],actions[:,:,i],rewards[:,i].unsqueeze(1),states[:,:,i+1]),axis=1)
                q_outs = self._encoder(q_ins)
                new_means, new_precs = q_outs[:,:self._latent_dim], torch.inverse(get_cov_mat(q_outs[:,self._latent_dim:],self._ns,self._q_cov_type,self._device))
                code_means, code_precs = gaussian_product_posterior(code_means,code_precs,new_means,new_precs)
                
        total_reward = torch.sum(rewards,1)
        best_traj = torch.argmax(total_reward)
        return actions[best_traj,:,0].numpy()
        
        
    def _next_state_rew(self, states, actions, codes):
        """helper function to unroll dynamics (batched)"""
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
    
class LSTMRandomShooting():
    """implements random shooting for RNNEncoder"""
    def __init__(self, transition, reward, encoder, action_dist, num_traj, traj_length, ns, na, latent_dim, t_cov_type, r_cov, q_cov_type, max_logvar, device, det=False):
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
        self._det = det
        
    def get_action(self,state,hidden):
        states = torch.zeros([self._num_traj,self._ns,self._traj_length])
        states[:,:,0] = torch.from_numpy(state).float().to(self._device)
        if type(self._action_dist) == torch.distributions.Categorical:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,1,self._traj_length-1)
        else:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,self._na,self._traj_length-1)
        rewards = torch.zeros([self._num_traj,self._traj_length-1])
        code_means, code_precs = torch.zeros(self._num_traj,self._latent_dim,1), torch.eye(self._latent_dim).expand(self._num_traj,self._latent_dim,self._latent_dim)
        for i in range(self._traj_length-1):
            if type(self._action_dist) == torch.distributions.Categorical:
                actions_input = torch.squeeze(torch.nn.functional.one_hot(actions[:,:,i]).float())
            else:
                actions_input = actions[:,:,i]
            
            codes = torch.squeeze(code_means+torch.matmul(torch.inverse(code_precs),torch.randn_like(code_means)))
            
            states[:,:,i+1], rewards[:,i] = self._next_state_rew(states[:,:,i],actions_input,codes)
            
            q_ins = torch.cat((states[:,:,i],actions[:,:,i],rewards[:,i].unsqueeze(1),states[:,:,i+1]),axis=1).view(self._num_traj,1,-1)
            q_outs, hidden = self._encoder(q_ins,hidden)
            new_means, new_precs = torch.squeeze(q_outs)[:,:self._latent_dim].unsqueeze(-1), torch.inverse(get_cov_mat(torch.squeeze(q_outs)[:,self._latent_dim:],self._ns,self._q_cov_type,self._device))
            code_means, code_precs = gaussian_product_posterior(code_means,code_precs,new_means,new_precs)
                
        total_reward = torch.sum(rewards,1)
        best_traj = torch.argmax(total_reward)
        return actions[best_traj,:,0].numpy()
        
        
    def _next_state_rew(self, states, actions, codes):
        """helper function to unroll dynamics (batched)"""
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
                