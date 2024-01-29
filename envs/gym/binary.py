import os
import numpy as np
import gym
import mj_envs  # noqa

AWAC_DATA_DIR = "/iris/u/maxsobolmark/awac-data"


def process_expert_dataset(expert_datset):
    """This is a mess, but works"""
    all_observations = []
    all_next_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []

    for x in expert_datset:
        all_observations.append(
            np.vstack([xx["state_observation"] for xx in x["observations"]])
        )
        all_next_observations.append(
            np.vstack([xx["state_observation"] for xx in x["next_observations"]])
        )
        all_actions.append(np.vstack([xx for xx in x["actions"]]))
        # for some reason rewards has an extra entry, so in rlkit they just remove the last entry: https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L84
        all_rewards.append(x["rewards"][:-1])
        all_terminals.append(x["terminals"])

    return {
        "observations": np.concatenate(all_observations, dtype=np.float32),
        "next_observations": np.concatenate(all_next_observations, dtype=np.float32),
        "actions": np.concatenate(all_actions, dtype=np.float32),
        "rewards": np.concatenate(all_rewards, dtype=np.float32),
        "terminals": np.concatenate(all_terminals, dtype=np.float32),
    }


def process_bc_dataset(bc_dataset):
    final_bc_dataset = {k: [] for k in bc_dataset[0] if "info" not in k}

    for x in bc_dataset:
        for k in final_bc_dataset:
            final_bc_dataset[k].append(x[k])

    return {
        k: np.concatenate(v, dtype=np.float32).squeeze()
        for k, v in final_bc_dataset.items()
    }


class BinaryEnv:
    def get_dataset(
        self,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
        remove_terminals=True,
        include_bc_data=True,
    ):
        env_prefix = self.env.spec.name[len("MBRL") :].split("-")[0].lower()

        expert_dataset = np.load(
            os.path.join(
                os.path.expanduser(AWAC_DATA_DIR), f"{env_prefix}2_sparse.npy"
            ),
            allow_pickle=True,
        )

        # this seems super random, but I grabbed it from here: https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/launchers/experiments/awac/awac_rl.py#L124 and here https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L153
        dataset_split = 0.9
        last_train_idx = int(dataset_split * len(expert_dataset))

        dataset_dict = process_expert_dataset(expert_dataset[:last_train_idx])

        if include_bc_data:
            bc_dataset = np.load(
                os.path.join(
                    os.path.expanduser(AWAC_DATA_DIR), f"{env_prefix}_bc_sparse4.npy"
                ),
                allow_pickle=True,
            )

            # this seems super random, but I grabbed it from here: https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/launchers/experiments/awac/awac_rl.py#L124 and here https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L153
            bc_dataset_split = 0.9
            bc_dataset = bc_dataset[: int(bc_dataset_split * len(bc_dataset))]
            bc_dataset = process_bc_dataset(bc_dataset)

            dataset_dict = {
                k: np.concatenate([dataset_dict[k], bc_dataset[k]])
                for k in dataset_dict
            }

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        if remove_terminals:
            dataset_dict["terminals"] = np.zeros_like(dataset_dict["terminals"])

        dones[-1] = True

        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones_float"] = dones.astype(np.float32)
        return dataset_dict

    def is_done(self, obs, next_obs):
        return False

    def __getattr__(self, item):
        return getattr(self.env, item)


class DoorBinaryEnv(BinaryEnv):
    def __init__(self):
        self.env = gym.make("door-binary-v0")

    def cost_np_vec(self, obs, acts, next_obs):
        assert len(obs.shape) == 2 and obs.shape[1] == 39
        assert self.reward_type == "binary"  # -1/0 rewards
        _qp_1_minus2 = obs[:, :27]  # noqa
        _latch_pos = obs[:, 27]  # noqa
        door_pos = obs[:, 28]
        _palm_pos = obs[:, 29:32]  # noqa
        _handle_pos = obs[:, 32:35]  # noqa
        _palm_pos_minus_handle_pos = obs[:, 35:38]  # noqa
        _door_open = obs[:, 38]  # noqa

        goal_achieved = door_pos >= 1.35

        reward = goal_achieved - 1
        return -reward


class PenBinaryEnv(BinaryEnv):
    def __init__(self):
        self.env = gym.make("pen-binary-v0")

    def cost_np_vec(self, obs, acts, next_obs):
        assert len(obs.shape) == 2 and obs.shape[1] == 45
        assert self.reward_type == "binary"  # -1/0 rewards
        _qp_0_minus6 = obs[:, :24]  # noqa
        _obj_pos = obs[:, 24:27]  # noqa
        _obj_vel = obs[:, 27:33]  # noqa
        obj_orien = obs[:, 33:36]
        desired_orien = obs[:, 36:39]
        obj_pos_minus_desired_pos = obs[:, 39:42]
        _obj_orien_minus_desired_orien = obs[:, 42:45]  # noqa

        dist = np.linalg.norm(obj_pos_minus_desired_pos, axis=1)
        orien_similarity = (obj_orien * desired_orien).sum(axis=1)

        goal_achieved = np.logical_and(dist < 0.075, orien_similarity > 0.95)

        reward = goal_achieved - 1
        return -reward


class RelocateBinaryEnv(BinaryEnv):
    def __init__(self):
        self.env = gym.make("relocate-binary-v0")

    def cost_np_vec(self, obs, acts, next_obs):
        assert len(obs.shape) == 2 and obs.shape[1] == 39
        assert self.reward_type == "binary"  # -1/0 rewards

        _qp_0_minus6 = obs[:, :30]  # noqa
        _palm_pos_minus_obj_pos = obs[:, 30:33]  # noqa
        _palm_pos_minus_target_pos = obs[:, 33:36]  # noqa
        obj_pos_minus_target_pos = obs[:, 36:39]

        goal_achieved = np.linalg.norm(obj_pos_minus_target_pos, axis=1) < 0.1

        reward = goal_achieved - 1
        return -reward
