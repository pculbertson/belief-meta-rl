import torch
import numpy as np
from utils.distributions import get_batch_mvnormal, get_cov_mat

class RandomShooting():
    """implements random shooting for MBRL"""
    def __init__(self, transition, reward, action_dist, num_traj, traj_length, ns, na, t_cov_type, r_cov, max_logvar, device, det=False):
        self._transition = transition
        self._reward = reward
        self._action_dist = action_dist
        self._num_traj = num_traj
        self._traj_length = traj_length
        self._ns = ns
        self._na = na
        self._t_cov_type = t_cov_type
        self._r_cov = r_cov
        self._max_logvar = max_logvar
        self._device = device
        self._det = det
        
    def get_action(self, state):
        #states, rews, actions = [torch.from_numpy(state).float().repeat(self._num_traj,1)],[],[]
        states = torch.zeros([self._num_traj,self._ns,self._traj_length])
        states[:,:,0] = torch.from_numpy(state).float().to(self._device)
        if type(self._action_dist) == torch.distributions.Categorical:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,1,self._traj_length-1)
        else:
            actions = self._action_dist.expand((self._num_traj,1)).sample([self._traj_length-1]).view(self._num_traj,self._na,self._traj_length-1)
        rewards = torch.zeros([self._num_traj,self._traj_length-1])
        for i in range(self._traj_length-1):
            if type(self._action_dist) == torch.distributions.Categorical:
                actions_input = torch.squeeze(torch.nn.functional.one_hot(actions[:,:,i]).float())
            else:
                actions_input = actions[:,:,i]
            #print((states[:,:,i].shape,actions_input.shape))
            states[:,:,i+1], rewards[:,i] = self._next_state_rew(states[:,:,i],actions_input)
        total_reward = torch.sum(rewards,1)
        best_traj = torch.argmax(total_reward)
        return actions[best_traj,:,0].numpy()
        
        
    def _next_state_rew(self, states, actions):
        """helper function to unroll dynamics (batched)"""
        ins = torch.cat((states,actions),axis=1).to(self._device)
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
                rews = r_outs
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
                