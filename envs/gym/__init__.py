from gym.envs.registration import register


register(
    id="MBRLHalfCheetah-v0",
    entry_point="envs.gym.half_cheetah:HalfCheetahEnv",
    kwargs={"frame_skip": 5},
    max_episode_steps=1000,
)

register(
    id="MBRLCheetahRun-v0",
    entry_point="envs.gym.cheetahrun:CheetahRunEnv",
    kwargs={"frame_skip": 5},
    max_episode_steps=1000,
)

register(
    id="MBRLWalker2d-v0",
    entry_point="envs.gym.walker2d:Walker2dEnv",
    kwargs={"frame_skip": 4},
    max_episode_steps=1000,
)

register(
    id="MBRLAnt-v0",
    entry_point="envs.gym.ant:AntEnv",
    kwargs={"frame_skip": 5},
    max_episode_steps=1000,
)

register(
    id="MBRLHopper-v0",
    entry_point="envs.gym.hopper:HopperEnv",
    kwargs={"frame_skip": 4},
    max_episode_steps=1000,
)

register(
    id="MBRLDoor-binary-v0",
    entry_point="envs.gym.binary:DoorBinaryEnv",
    max_episode_steps=1000,
)

register(
    id="MBRLPen-binary-v0",
    entry_point="envs.gym.binary:PenBinaryEnv",
    max_episode_steps=1000,
)

register(
    id="MBRLRelocate-binary-v0",
    entry_point="envs.gym.binary:RelocateBinaryEnv",
    max_episode_steps=1000,
)

register(
    id="MBRLKitchen-partial-v0",
    entry_point="envs.gym.kitchen:KitchenEnv",
    kwargs={"kitchen_env_name": "kitchen-partial-v0"},
    max_episode_steps=400,
)

register(
    id="MBRLKitchen-complete-v0",
    entry_point="envs.gym.kitchen:KitchenEnv",
    kwargs={"kitchen_env_name": "kitchen-complete-v0"},
    max_episode_steps=400,
)

register(
    id="MBRLKitchen-mixed-v0",
    entry_point="envs.gym.kitchen:KitchenEnv",
    kwargs={"kitchen_env_name": "kitchen-mixed-v0"},
    max_episode_steps=400,
)

env_name_to_gym_registry = {
    "half_cheetah": "MBRLHalfCheetah-v0",
    "ant": "MBRLAnt-v0",
    "hopper": "MBRLHopper-v0",
    "walker2d": "MBRLWalker2d-v0",
    "cheetah_run": "MBRLCheetahRun-v0",
    "door": "MBRLDoor-binary-v0",
    "pen": "MBRLPen-binary-v0",
    "relocate": "MBRLRelocate-binary-v0",
    "kitchen_partial": "MBRLKitchen-partial-v0",
    "kitchen_complete": "MBRLKitchen-complete-v0",
    "kitchen_mixed": "MBRLKitchen-mixed-v0",
}
