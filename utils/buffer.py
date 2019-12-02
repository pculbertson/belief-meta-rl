import numpy as np
import random

class ReplayBuffer:
    """class used to store experience. adapted from PEARL."""
    def __init__(self,max_replay_buffer_size,obs_dim,action_dim):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = size = max_replay_buffer_size
        self._ep_starts = [] #list of episode start indices
        
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
        if self._top in self._ep_starts:
            self._ep_starts.remove(self._top)
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
        """samples batch of unordered transitions"""
        indices = np.random.randint(0,self._size,batch_size)
        return self.sample_data(indices)
    
    def random_sequences(self, batch_size, seq_length=-1):
        """samples batch of [batch_size] sequences; returns full episode if [seq_length] = -1"""
        sequences = []
        for i in range(batch_size):
            ep_num = random.randint(0,len(self._ep_starts)-1)
            ep_start = self._ep_starts[ep_num]
            if seq_length == -1: #full episode length
                if (ep_num == len(self._ep_starts)-1):
                    #handle episode wraparound
                    if (self._size == self._max_replay_buffer_size):
                        ep_range = list(range(ep_start,self._size)) + list(range(0,self._ep_starts[0]))
                    else:
                        ep_range = list(range(ep_start,self._top))
                else: 
                    ep_range = list(range(ep_start,self._ep_starts[ep_num+1]))
            else:
                if (ep_num == len(self._ep_starts)-1):
                    if (self._size == self._max_replay_buffer_size):
                        if self._size - ep_start >= seq_length:
                            ep_range = list(range(ep_start,ep_start+seq_length))
                        else:
                            remainder = seq_length - (self._size - ep_start)
                            ep_end = min(remainder,self._ep_starts[0])
                            ep_range = list(range(ep_start,self._size)) + list(range(0,ep_end))
                    else:
                        ep_end = min(ep_start+seq_length,self._top)
                        ep_range = list(range(ep_start,ep_end))
                else:
                    ep_end = min(ep_start+seq_length,self._ep_starts[ep_num+1])
                    ep_range = list(range(ep_start,ep_end))
            sequences.append(self.sample_data(ep_range))
        return sequences
                
    def new_episode(self):
        """starts new episode in replay buffer"""
        self._ep_starts.append(self._top)