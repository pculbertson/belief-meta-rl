import torch
import numpy as np
from utils.distributions import get_batch_mvnormal, get_cov_mat
from utils.env_utils import pendulum_next_state_rew

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
        return (sp+states.to(self._device), torch.squeeze(rews))
    
    
class CrossEntropy():
    """implements cross-entropy method for MBRL"""
    def __init__(self, transition, reward, action_dist, num_traj, traj_length, num_iters, elite_frac, ns, na, t_cov_type, r_cov, max_logvar, smoothing, device, true_dyn=False, env=None):
        self._transition = transition
        self._reward = reward
        self._action_dist = action_dist #list of action distributions
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
        self._smoothing = smoothing
        self._device = device
        self._true_dyn = true_dyn
        self._env = env
        if type(self._action_dist[0]) == torch.distributions.Categorical:
            self._discrete_action = True
        else:
            self._discrete_action = False

    def new_action_dist(self, state):
        #print([dist.covariance_matrix for dist in self._action_dist])
        curr_action_dist = self._action_dist
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
            total_rew = torch.mean(rewards,1)
            elite_indices = torch.argsort(total_rew,descending=True)[:self._num_elite]
            if iteration == 0:
                pass
                #print('sp_hat: ', torch.mean(states[elite_indices,:,1],axis=0), ' r_hat: ', torch.mean(rewards[elite_indices,0]))
                #print('iter: ', iteration, ' rew: ', torch.mean(total_rew))
            elif iteration == self._num_iters-1:
                pass
                #print('iter: ', iteration, ' rew: ', torch.mean(total_rew[elite_indices]))
                #print('sp_hat: ', torch.mean(states[elite_indices,:,1],axis=0), ' r_hat: ', torch.mean(rewards[elite_indices,0]))
            curr_action_dist = self._fit_action_dist(actions[elite_indices,:,:],curr_action_dist)
        #sp, r = pendulum_next_state_rew(states[:,:,0],curr_action_dist[0].mean,self._env)
        #print(sp[0,:],curr_action_dist[0].mean,r[0])
        #print(np.arccos(states[0,0,0].numpy()),states[0,2,0].numpy())
        return curr_action_dist
    
    def set_action_dist(self,action_dist):
        self._action_dist = action_dist
        
    def _next_state_rew(self, states, actions):
        if self._true_dyn:
            sp, r = pendulum_next_state_rew(states,actions,self._env)
            return torch.from_numpy(sp).float(), torch.from_numpy(r).float()
        if self._discrete_action:
            pass
        else:
            ins = torch.cat((states,actions),axis=1).to(self._device)
            t_out = self._transition(ins)
            r_outs = self._reward(ins)
            
            t_means, t_covs = t_out[:,:self._ns], t_out[:,self._ns:]
            t_covs_clamped = torch.clamp(t_covs,-self._max_logvar,self._max_logvar).to(self._device)
            cov_mat = get_cov_mat(t_covs_clamped,self._ns,self._t_cov_type,self._device)
            
            #sp = t_means.to(self._device) + states.to(self._device)
            sp = t_means + torch.squeeze(torch.matmul(cov_mat,torch.randn_like(t_means).view(self._num_traj,self._ns,1))) + states.to(self._device)
            
            if self._r_cov:
                r_logvar_clamped = torch.clamp(r_outs[:,1],-self._max_logvar,self._max_logvar).to(self._device)
                rews = torch.distributions.Normal(r_outs[:,0],torch.exp(r_logvar_clamped)).rsample()
            else:
                rews = r_outs
            
            return (sp, torch.squeeze(rews))
            
    
    def _fit_action_dist(self,elite_actions,prev_dist):
        action_dist = []
        if self._discrete_action:
            pass
        else: #assume train of Gaussian action distributions
            action_means = torch.mean(elite_actions,0)
            action_variances = torch.squeeze(torch.var(elite_actions,0))
            for t in range(self._traj_length-1):
                new_mean = (1-self._smoothing)*action_means[:,t] + self._smoothing*prev_dist[t].mean
                new_var = (1-self._smoothing)*action_variances[t]*torch.eye(self._na) + self._smoothing*(prev_dist[t].covariance_matrix+ torch.eye(self._na))
                action_dist.append(torch.distributions.MultivariateNormal(new_mean,new_var))
            #action_dist = [torch.distributions.MultivariateNormal(action_means[:,t],torch.eye(self._na)) for t in range(self._traj_length-1)]
        return action_dist
                
class Scenario():
    """implements cross-entropy method for MBRL"""
    def __init__(self, transition, reward, action_dist, a_opt, num_traj, traj_length, num_iters, ns, na, t_cov_type, r_cov, a_cov_type, max_logvar, device, true_dyn=False, env=None):
        self._transition = transition
        self._reward = reward
        self._action_dist = action_dist #should be a Torch variable of size n_traj x na + na_cov
        self._a_opt = a_opt
        self._num_traj = num_traj
        self._traj_length = traj_length
        self._num_iters = num_iters
        self._ns = ns
        self._na = na
        self._t_cov_type = t_cov_type
        self._r_cov = r_cov
        self._a_cov_type = a_cov_type
        self._max_logvar = max_logvar
        self._device = device
        self._true_dyn = true_dyn
        self._env = env
        if type(self._action_dist[0]) == torch.distributions.Categorical:
            self._discrete_action = True
        else:
            self._discrete_action = False

    def new_action_dist(self, state):
        """take in current state, roll out scenarios & take gradient step on policy"""
        curr_action_dist = self._action_dist
        orig_action_dist = curr_action_dist.data.detach().cpu().numpy()
        for iteration in range(self._num_iters):
            self._a_opt.zero_grad()
            states = torch.zeros([self._num_traj,self._ns,self._traj_length])
            states[:,:,0] = torch.from_numpy(state).float().to(self._device)
            
            action_means = curr_action_dist[:self._na,:].view(1,self._traj_length-1,self._na,1).expand(self._num_traj,self._traj_length-1,self._na,1)
            action_covs = get_cov_mat(self._action_dist[self._na:,:].view(-1,self._na),self._na,self._a_cov_type,self._device).expand(self._num_traj,self._traj_length-1,self._na,self._na)
            
            actions = action_means# + torch.matmul(action_covs,torch.randn_like(action_means)).to(self._device)
            
            #shoot trajectories forward, get rewards
            rewards = torch.zeros([self._num_traj,self._traj_length-1])
            for i in range(self._traj_length-1):
                states[:,:,i+1], rewards[:,i] = self._next_state_rew(states[:,:,i].to(self._device),actions[:,i,:].view(self._num_traj,self._na))
            #take elite fraction, refit action distributions
            loss = torch.mean(rewards**2)
            print(loss)
            loss.backward()
            self._a_opt.step()
            
        #print(pendulum_next_state_rew(states[0,:,0].view(1,self._ns),curr_action_dist[:self._na,0].view(1,self._na),self._env))
        print(torch.norm(curr_action_dist-torch.from_numpy(orig_action_dist).float().cuda()))
        return curr_action_dist
    
    def set_action_dist(self,action_dist):
        self._action_dist = action_dist
        
    def _next_state_rew(self, states, actions):
        if self._true_dyn:
            sp, r = pendulum_next_state_rew(states,actions,self._env)
            #return torch.from_numpy(sp.cpu()).float(), torch.from_numpy(r).float()
            return sp, r
        if self._discrete_action:
            pass
        else:
            ins = torch.cat((states,actions),axis=1).to(self._device)
            t_out = self._transition(ins)
            r_outs = self._reward(ins)
            
            t_means, t_covs = t_out[:,:self._ns], t_out[:,self._ns:]
            t_covs_clamped = torch.clamp(t_covs,-self._max_logvar,self._max_logvar).to(self._device)
            cov_mat = get_cov_mat(t_covs_clamped,self._ns,self._t_cov_type,self._device)
            
            #sp = t_means.to(self._device) + states.to(self._device)
            sp = t_means + torch.squeeze(torch.matmul(cov_mat,torch.randn_like(t_means).view(self._num_traj,self._ns,1))) + states.to(self._device)
            
            if self._r_cov:
                r_logvar_clamped = torch.clamp(r_outs[:,1],-self._max_logvar,self._max_logvar).to(self._device)
                rews = torch.distributions.Normal(r_outs[:,0],torch.exp(r_logvar_clamped)).rsample()
            else:
                rews = r_outs
            
            return (sp, torch.squeeze(rews))