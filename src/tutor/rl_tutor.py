import numpy as np

import functools
import gym
import numpy as np

from torch import nn
import pfrl
from pfrl.agents import PPO as ppo
from pfrl import experiments
from pfrl import utils as pfutils
import logging

import torch
import copy

import utils

class MyEnv(gym.Env):
    def __init__(self, env):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = [-float("inf"), float("inf")]
        self.env = env

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()


class RLTutor:
    def __init__(
        self,
        env,
        num_iteration,
        num_envs,
        num_timesteps,
        seed,
        gamma,
        lambd,
        value_func_coef,
        entropy_coef,
        clip_eps,
    ):
        self.name = "RLTutor"
        self.raw_env = env
        self.num_iteration = num_iteration
        self.num_envs = num_envs
        self.num_timesteps = num_timesteps

        self.sum_steps = self.num_envs * self.num_timesteps * self.num_iteration
        self.update_interval = self.num_envs * self.num_timesteps
        self.eval_interval = self.sum_steps + 1

        self.agent = None

        self.process_seeds = np.arange(self.num_envs) + seed * self.num_envs

        # Only for getting timesteps, and obs-action spaces
        sample_env = self._make_env(1, False)
        obs_space = sample_env.observation_space
        action_space = sample_env.action_space
        print("Observation space:", obs_space)
        print("Action space:", action_space)
        del sample_env
        self.obs_size = obs_space.low.size
        self.action_size = action_space.n

        self.gamma = gamma
        self.lambd = lambd
        self.value_func_coef = value_func_coef
        self.entropy_coef = entropy_coef
        self.clip_eps = clip_eps

        self._init_agent()

    def _lecun_init(self, layer, gain=1):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pfrl.initializers.init_lecun_normal(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        else:
            pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
            pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
        return layer

    def _init_agent(self):
        model = pfrl.nn.RecurrentSequential(
            self._lecun_init(nn.GRU(self.obs_size, 512, 1)),
            pfrl.nn.Branched(
                nn.Sequential(
                    self._lecun_init(nn.Linear(512, self.action_size), 1e-2),
                    pfrl.policies.SoftmaxCategoricalHead(),
                ),
                self._lecun_init(nn.Linear(512, 1)),
            ),
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-8)
        self.agent = ppo(
            model,
            opt,
            obs_normalizer=None,
            gpu=0,
            gamma=self.gamma,
            lambd=self.lambd,
            phi=lambda x: x,
            value_func_coef=self.value_func_coef,
            entropy_coef=self.entropy_coef,
            update_interval=self.update_interval,
            minibatch_size=self.num_timesteps,
            epochs=10,
            clip_eps=self.clip_eps,
            clip_eps_vf=None,
            standardize_advantages=True,
            recurrent=True,
            max_recurrent_sequence_len=None,
            act_deterministically=False,
            max_grad_norm=None,
            value_stats_window=200,
            entropy_stats_window=200,
            value_loss_stats_window=200,
            policy_loss_stats_window=200,
        )

    def _make_env(self, idx, test):
        process_seed = int(self.process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        # Set a random seed used in PFRL
        pfutils.set_random_seed(env_seed)

        env = copy.deepcopy(self.raw_env)
        gym_env = MyEnv(env)
        # Cast observations to float32 because our model uses float32
        gym_env = pfrl.wrappers.CastObservationToFloat32(gym_env)
        return gym_env

    def _make_batch_env(self, test):
        env_list = []
        for idx, env in enumerate(range(self.num_envs)):
            a = functools.partial(self._make_env, idx, test)
            env_list.append(a)
        b = pfrl.envs.MultiprocessVectorEnv(env_list)
        return b

    def train(self, session_num, output_dir, logger, lr):
        logger = logger or logging.getLogger(__name__)

        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(self.sum_steps, lr, 0, lr_setter)
        )

        reward = utils.train_agent_batch(
            agent=self.agent,
            env=self._make_batch_env(False),
            steps=self.sum_steps,
            outdir=output_dir,
            session_num=session_num,
            num_steps=self.num_timesteps,
            num_items=self.raw_env.n_items,
            log_interval=self.num_envs * 200,
            step_hooks=step_hooks,
            logger=logger,
        )

        assert len(reward) == self.num_iteration
        return reward

    def act(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
            obs = obs.to("cuda")
            obs = obs.to(torch.float32)
        elif isinstance(obs, list):
            obs = torch.Tensor(obs)
            obs = obs.to("cuda")
            obs = obs.to(torch.float32)
        else:
            raise TypeError("Unsupported type of data")

        with self.agent.eval_mode():
            [action] = self.agent.batch_act([obs])

        return action

    def load_agent(self, save_path):
        self.agent.load(save_path)
