import numpy as np
import gym

def true_next_state_rew(env, num_traj, dim_obs, states, actions):
    sp = np.zeros((num_traj, dim_obs))
    rew = np.zeros((num_traj))
    for i in range(num_traj):
        env.state = np.array([np.arccos(states[i,0].numpy()),states[i,-1].numpy()])
        sp[i,:], rew[i], _, _  = env.step(actions[i,:])
    return sp, rew

def pendulum_next_state_rew(states,actions,env):
    th, thdot = np.arccos(states[:,0].numpy()), states[:,2].numpy()

    g = env.g
    m = env.m
    l = env.l
    dt = env.dt

    u = np.squeeze(np.clip(actions.numpy(), -env.max_torque, env.max_torque))
    costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
    
    newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
    newth = th + newthdot*dt
    newthdot = np.clip(newthdot, -env.max_speed, env.max_speed) #pylint: disable=E1111

    sp = np.stack((newth,newthdot),axis=1)
    return pendulum_get_obs(sp), -costs

def angle_normalize(x):
    return np.mod(x+np.pi, 2*np.pi) - np.pi

def pendulum_get_obs(sp):
    theta, thetadot = sp[:,0], sp[:,1]
    return np.stack((np.cos(theta), np.sin(theta), thetadot),axis=1)