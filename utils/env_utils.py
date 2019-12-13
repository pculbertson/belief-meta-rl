import numpy as np
import torch
import gym

def pendulum_next_state_rew(states,actions,env):
    th, thdot = torch.atan2(states[:,1],states[:,0]), states[:,2]

    g = env.g
    m = env.m
    l = env.l
    dt = env.dt

    u = torch.squeeze(torch.clamp(actions, -env.max_torque, env.max_torque))
    costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
    
    newthdot = thdot + (-3*g/(2*l) * torch.sin(th + np.pi) + 3./(m*l**2)*u) * dt
    newth = th + newthdot*dt
    newthdot = torch.clamp(newthdot, -env.max_speed, env.max_speed) #pylint: disable=E1111

    sp = torch.stack((newth,newthdot),axis=1)
    return pendulum_get_obs(sp), -costs

def angle_normalize(x):
    return torch.fmod(x+np.pi, 2*np.pi) - np.pi

def pendulum_get_obs(sp):
    theta, thetadot = sp[:,0], sp[:,1]
    return torch.stack((torch.cos(theta), torch.sin(theta), thetadot),axis=1)