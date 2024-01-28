import numpy as np
import gym
import d4rl


class KitchenEnv:
    def __init__(self, kitchen_env_name: str):
        self.env = gym.make(kitchen_env_name)
        self.env.REMOVE_TASKS_WHEN_COMPLETE = False

    def cost_np_vec(self, obs, acts, next_obs):
        assert obs.shape == (60,)
        assert self.goal_concat

        qp = obs[:9]
        obj_qp = obs[9:30]
        goal = obs[30:60]

        assert not self.REMOVE_TASKS_WHEN_COMPLETE

        reward_dict, score = self._get_reward_n_score(
            {
                "qp": qp,
                "obj_qp": obj_qp,
                "goal": goal,
            }
        )
        reward = reward_dict["r_total"]
        return -reward

    def __getattr__(self, item):
        return getattr(self.env, item)
