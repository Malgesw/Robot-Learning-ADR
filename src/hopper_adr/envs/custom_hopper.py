"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, xml_path=None):
        MujocoEnv.__init__(self, 4, xml_path)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(
            self.sim.model.body_mass[1:])    # Default link masses

        # Source environment has an imprecise torso mass (1kg shift)
        if domain == 'source':
            self.sim.model.body_mass[1] -= 1.0

    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        # print(self.sample_parameters())
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """
        lower_bound = self.original_masses[1:]*0.4
        upper_bound = self.original_masses[1:]*1.2
        # sample body_mass.shape - 1 values for the movable masses
        return np.random.uniform(lower_bound, upper_bound)

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.sim.model.body_mass[1:])
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[2:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (
            np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        # self.set_random_parameters()
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + \
            self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


class ADRCallback(BaseCallback):
    def __init__(self, train_env, test_env, thresh_high=1400, thresh_low=800, buffer_size=10, prob_adr=1, verbose=0):
        super(ADRCallback, self).__init__(verbose)

        self.train_env = train_env
        self.test_env = test_env
        self.buffer_size = buffer_size
        self.thresh_high = thresh_high
        self.thresh_low = thresh_low
        self.prob_adr = prob_adr

        model = self.train_env.unwrapped.model
        self.original_masses = np.copy(model.body_mass[2:])

        # id: [lower bound, upper bound] for each parameter
        self.params = {i: [self.original_masses[i]*0.4, self.original_masses[i]*1.2]
                       for i in range(0, len(self.original_masses))}
        self.performance_buffers = {param: [[], []] for param in self.params}
        self.entropies = []
        self.count = 0

    def randomize_environment(self):

        lower_bound = [bounds[0] for bounds in self.params.values()]
        upper_bound = [bounds[1] for bounds in self.params.values()]
        model = self.train_env.unwrapped.model
        model.body_mass[2:] = np.random.uniform(lower_bound, upper_bound)

    def adjust_boundary(self, param_id, avg_reward):
        bounds = self.params[param_id]
        if avg_reward >= self.thresh_high:
            # expand the boundaries
            bounds[0] *= 0.95
            bounds[1] *= 1.05
        elif avg_reward <= self.thresh_low:
            # shrink the boundaries
            bounds[0] *= 1.05
            bounds[1] *= 0.95
        original_mass = self.original_masses[param_id]
        # bounds[0] = max(0.1, bounds[0])
        # bounds[1] = min(2.0, bounds[1])
        bounds[0] = max(original_mass*0.1, bounds[0])
        bounds[1] = min(original_mass*2, bounds[1])

    def _on_step(self) -> bool:
        if "dones" in self.locals:
            # Check if any environment in the vectorized env is done
            if any(self.locals["dones"]):
                self.randomize_environment()

        return True

    def _on_rollout_end(self) -> None:
        if np.random.rand() < self.prob_adr:
            # perform ADR
            self.randomize_environment()
            chosen_param_id = np.random.choice(list(self.params.keys()))
            model = self.train_env.unwrapped.model
            buffer_pos = 0  # 0 -> lower bounds, 1 -> upper bounds
            if np.random.rand() < 0.5:
                # assign lower bound
                model.body_mass[2:][chosen_param_id] = self.params[chosen_param_id][0]
            else:
                # assign upper bound
                model.body_mass[2:][chosen_param_id] = self.params[chosen_param_id][1]
                buffer_pos = 1

            mean_reward, _ = evaluate_policy(
                self.model, self.train_env, n_eval_episodes=50)
            buffer = self.performance_buffers[chosen_param_id][buffer_pos]
            buffer.append(mean_reward)
            if len(buffer) >= self.buffer_size:
                avg = np.mean(buffer)
                buffer.clear()
                self.adjust_boundary(param_id=chosen_param_id, avg_reward=avg)

            entropy = 0
            for param, (lower, upper) in self.params.items():
                # Log difference for each parameter
                log_diff = np.log(upper - lower)
                entropy += log_diff
            entropy = entropy / len(self.params)
            self.entropies.append(entropy)
            self.count = self.count + 1
            print(self.params)
            print(self.performance_buffers)
            print(entropy)

    def _on_training_end(self) -> None:

        try:
            counts = [count+1 for count in range(self.count)]
            fig = plt.figure(figsize=(10, 10))
            plt.plot(counts, self.entropies)
            plt.xlabel("Rollouts")
            plt.ylabel("ADR Entropy")
            plt.show()
        except:
            print("Exception")


"""
    Registered environments
"""
gym.envs.register(
    id="CustomHopper-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
)

gym.envs.register(
    id="CustomHopper-source-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.envs.register(
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"}
)
