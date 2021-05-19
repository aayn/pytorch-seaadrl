import numpy as np
import gym

import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs

env_name = "PongNoFrameskip-v4"
seed = 1
num_procs = 2
gamma = 0.99
log_dir = "/tmp/gym"
device = "cuda:0"


def test_prediction():
    envs = make_vec_envs(env_name, seed, num_procs, gamma, log_dir, device, False)
    _ = envs.reset()
    for _ in range(np.random.randint(5, 10)):
        _ = envs.step(torch.tensor([[0], [1]]))

    next_actions = torch.tensor([[0, 1], [1, 0], [1, 1]]).reshape((-1, 2, 1))
    print(next_actions)

    def predict(envs, actions):
        clone_fns = envs.get_attr("clone_full_state")
        init_env_states = [cf() for cf in clone_fns]
        # for uenv in unwrapped_envs:
        #     init_env_states.append()

        pred_ram_states = []

        for a in actions:
            _ = envs.step(a)
            pred_ram_states.append(
                [uenv._get_ram() for uenv in envs.get_attr("unwrapped")]
            )
        
        restore_fns = envs.get_attr("restore_full_state")
        for rs, rf in zip(init_env_states, restore_fns):
            rf(rs)
        
        return pred_ram_states

    pred_ram_states = predict(envs, next_actions)
    

    for i, a in enumerate(next_actions):
        _ = envs.step(a)
        ram_states = [uenv._get_ram() for uenv in envs.get_attr("unwrapped")]
        for prs, rs in zip(pred_ram_states[i], ram_states):
            print(any(prs - rs))


def main():
    torch.set_num_threads(1)
    envs = make_vec_envs(env_name, seed, num_procs, gamma, log_dir, device, False)
    print(envs.action_space.n)
    states = envs.reset()
    print(envs.observation_space.shape)
    # print(envs.get_images())
    for uenv in envs.get_attr("unwrapped"):
        print(uenv._get_ram().shape)


if __name__ == "__main__":
    # main()
    test_prediction()
