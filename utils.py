import numpy as np
import torch

class ReplayBuffer:
    """class used to store experience. adapted from PEARL."""
    def __init__(self,max_replay_buffer_size,obs_dim,action_dim):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = size = max_replay_buffer_size
        
        self._o = np.zeros((size, obs_dim))
        self._a = np.zeros((size, action_dim))
        self._op = np.zeros((size, obs_dim))
        self._r = np.zeros((size,1))
        self._d = np.zeros((size,1))
        
        self.clear()
        
    def add_sample(self,o,a,r,op,d):
        """adds sample to replay buffer"""
        self._o[self._top] = o
        self._a[self._top] = a
        self._r[self._top] = r
        self._op[self._top] = op
        self._d[self._top] = d
        
        self._advance()
        
    def _advance(self):
        """helper function to advance buffer index"""
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
            
    def clear(self):
        """resets replay buffer counts"""
        self._top = 0
        self._size = 0
        
    def size(self):
        """returns buffer size"""
        return self._size
    
    def sample_data(self,indices):
        """returns dictionary of (o,a,r,o',d) samples @ specified indices"""
        return dict(
            o=self._o[indices],
            a=self._a[indices],
            r=self._r[indices],
            op=self._op[indices],
            d=self._d[indices]
        )
    
    def random_batch(self, batch_size):
        """samples random transition batch"""
        indices = np.random.randint(0,self._size,batch_size)
        return self.sample_data(indices)
    
def get_batch_mvnormal(means, covs, cov_type='diag',device="cpu"):
    """return torch multivariate normal for sampling/prob evaluation"""
    if cov_type=='diag':
        cov_mat = torch.diag_embed(torch.exp(covs))
        batch_mvnormal = torch.distributions.MultivariateNormal(means,cov_mat)
    elif cov_type=='dense':
        batch_size, ns = covs.shape[0], means.shape[1]
        cov_mat = torch.zeros(batch_size,ns,ns)
        cov_mat[:,torch.tril(torch.ones(ns,ns))==1] = covs
        batch_mvnormal = torch.distributions.MultivariateNormal(means,scale_tril=cov_mat)
    elif cov_type=='scalar':
        batch_size, ns = covs.shape[0], means.shape[1]
        cov_mat = torch.exp(covs.to(device)).reshape(batch_size,1,1)*(torch.eye(ns).to(device))
        batch_mvnormal = torch.distributions.MultivariateNormal(means.to(device),cov_mat)
    elif cov_type=='fixed':
        cov_mat = covs
        batch_mvnormal = torch.distributions.MultivariateNormal(means,cov_mat)
    return batch_mvnormal

def get_cov_mat(covs, ns, cov_type='diag',device="cpu"):
    if cov_type=='diag':
        cov_mat = torch.diag_embed(torch.exp(covs))
    elif cov_type=='dense':
        batch_size = covs.shape[0]
        cov_mat = torch.zeros(batch_size,ns,ns)
        cov_mat[:,torch.tril(torch.ones(ns,ns))==1] = covs
    elif cov_type=='scalar':
        batch_size = covs.shape[0]
        cov_mat = (torch.exp(covs)).reshape(batch_size,1,1)*(torch.eye(ns).to(device))
    elif cov_type=='fixed':
        cov_size = 1e-2
        cov_mat = cov_size*torch.eye(ns)
    return cov_mat
    
def log_transition_probs(means, covs, ops, cov_type='diag',device="cpu"):
    """given network outputs, calculates log probability for next observations"""
    batch_mvnormal = get_batch_mvnormal(means,covs,cov_type,device)    
    return batch_mvnormal.log_prob(ops.to(device))

def log_rew_probs(means, covs, rews):
    dists = torch.distributions.Normal(means,torch.exp(covs))
    return dists.log_prob(rews)

def eval_policy(env,policy,init_action_dist,traj_length,discrete_actions,num_evals):
    rews = []
    for i in range(num_evals):
        action_dist = [init_action_dist]*(traj_length-1)
        s, d, ep_rew = env.reset(), False, 0.
        while not d:
            action_dist = policy.new_action_dist(np.array(s))
            if discrete_actions:
                a = action_dist.pop(0).sample().numpy()
            else:
                a = action_dist.pop(0).rsample()
            if env.action_space.shape:
                sp, r, d, _ = env.step(a) # take a random action
            else:
                sp, r, d, _ = env.step(int(a)) # take a random action
            ep_rew += r
            action_dist.append(init_action_dist)
            policy.set_action_dist(action_dist)
        rews.append(ep_rew)
    return torch.mean(torch.stack(rews))