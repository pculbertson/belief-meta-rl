import numpy as np
import torch

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