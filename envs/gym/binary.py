import numpy as np
import gym
import mj_envs  # noqa


class DoorBinaryEnv:
    def __init__(self):
        self.env = gym.make("door-binary-v0")

    def cost_np_vec(self, obs, acts, next_obs):
        assert obs.shape == (39,)
        assert self.reward_type == "binary"  # -1/0 rewards
        _qp_1_minus2 = obs[:27]  # noqa
        _latch_pos = obs[27]  # noqa
        door_pos = obs[28]
        _palm_pos = obs[29:32]  # noqa
        _handle_pos = obs[32:35]  # noqa
        _palm_pos_minus_handle_pos = obs[35:38]  # noqa
        _door_open = obs[38]  # noqa

        goal_achieved = True if door_pos >= 1.35 else False

        reward = goal_achieved - 1
        return -reward


class PenBinaryEnv:
    def __init__(self):
        self.env = gym.make("pen-binary-v0")

    def cost_np_vec(self, obs, acts, next_obs):
        assert obs.shape == (45,)
        assert self.reward_type == "binary"  # -1/0 rewards
        _qp_0_minus6 = obs[:24]  # noqa
        _obj_pos = obs[24:27]  # noqa
        _obj_vel = obs[27:33]  # noqa
        obj_orien = obs[33:36]
        desired_orien = obs[36:39]
        obj_pos_minus_desired_pos = obs[39:42]
        _obj_orien_minus_desired_orien = obs[42:45]  # noqa

        dist = np.linalg.norm(obj_pos_minus_desired_pos)
        orien_similarity = np.dot(obj_orien, desired_orien)

        goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False

        reward = goal_achieved - 1
        return -reward


class RelocateBinaryEnv:
    def __init__(self):
        self.env = gym.make("relocate-binary-v0")

    def cost_np_vec(self, obs, acts, next_obs):
        assert obs.shape == (39,)
        assert self.reward_type == "binary"  # -1/0 rewards

        _qp_0_minus6 = obs[:30]  # noqa
        _palm_pos_minus_obj_pos = obs[30:33]  # noqa
        _palm_pos_minus_target_pos = obs[33:36]  # noqa
        obj_pos_minus_target_pos = obs[36:39]

        goal_achieved = (
            True if np.linalg.norm(obj_pos_minus_target_pos) < 0.1 else False
        )

        reward = goal_achieved - 1
        return -reward
