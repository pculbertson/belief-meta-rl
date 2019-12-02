from gym.envs.registration import register

register(
    id='PointRobot-v0',
    entry_point='meta_env.envs:PointEnv',
)
register(
    id='SparsePointRobot-v0',
    entry_point='meta_env.envs:SparsePointEnv',
)
register(
    id='MetaPendulum-v0',
    entry_point='meta_env.envs:PendulumEnv',
)