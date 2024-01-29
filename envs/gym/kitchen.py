import numpy as np
import gym
import d4rl


class KitchenEnv:
    def __init__(self, kitchen_env_name: str):
        self.env = gym.make(kitchen_env_name)
        self.env.REMOVE_TASKS_WHEN_COMPLETE = False

    def cost_np_vec(self, obs, acts, next_obs):
        assert len(obs.shape) == 2 and obs.shape[1] == 60
        assert self.goal_concat

        qp = obs[:, :9]
        obj_qp = obs[:, 9:30]
        goal = obs[:, 30:60]

        assert not self.REMOVE_TASKS_WHEN_COMPLETE

        # reward_dict, score = self._get_reward_n_score(
        #     {
        #         "qp": qp,
        #         "obj_qp": obj_qp,
        #         "goal": goal,
        #     }
        # )
        # reward = reward_dict["r_total"]
        reward = np.stack(
            [
                self._get_reward_n_score(
                    {
                        "qp": qp[i],
                        "obj_qp": obj_qp[i],
                        "goal": goal[i],
                    }
                )[0]["r_total"]
                for i in range(len(qp))
            ]
        )
        return -reward

    def is_done(self, obs, next_obs):
        return False

    def __getattr__(self, item):
        return getattr(self.env, item)
