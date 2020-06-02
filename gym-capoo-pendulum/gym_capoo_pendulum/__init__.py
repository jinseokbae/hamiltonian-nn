from gym.envs.registration import register

register(
    id='capoo-pendulum-v0',
    entry_point='gym_capoo_pendulum.envs:CapooPendulumEnv',
)
# register(
#     id='capoo-pendulum-extrahard-v0',
#     entry_point='gym_capoo_pendulum.envs:FooExtraHardEnv',
# )