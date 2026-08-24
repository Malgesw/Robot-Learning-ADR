import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils, spaces
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from .custom_hopper import CustomHopper

import matplotlib.pyplot as plt


class CustomHopperWithObstacles(CustomHopper):
    def __init__(self, domain=None):

        super().__init__(domain=domain, xml_path="assets/hopperWithObs.xml")

        self.obstacle_1_id = self.get_geom_id('obstacle_1')
        # self.obstacle_2_id = self.get_geom_id('obstacle_2')

        # Store the original obstacle positions
        # self.original_obstacle_positions = np.copy(
        #    self.sim.model.geom_pos[[self.obstacle_1_id, self.obstacle_2_id], :])

        # Update observation space to include obstacles positions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.sim.data.qpos) - 1 + len(self.sim.data.qvel) + 1,), dtype=np.float32
        )

        if domain == 'source':
            # Set specific obstacle positions for source domain (x-axis pos with a +1 offset)
            self.sim.model.geom_pos[self.get_geom_id('obstacle_1')][0] += 1
            # self.sim.model.geom_pos[self.get_geom_id('obstacle_2')][0] += 1
            # self.sim.model.geom_size[self.get_geom_id('obstacle_1')][2] += 0.5

        # Store the original obstacle positions
        # self.original_obstacle_positions = np.copy(
            # self.sim.model.geom_pos[[self.obstacle_1_id], :])
        self.original_obstacle_position = self.sim.model.geom_pos[self.get_geom_id(
            'obstacle_1')][0]
        # self.original_height = np.copy(
        #    self.sim.model.geom_size[self.get_geom_id('obstacle_1')][2])

    def get_geom_id(self, geom_name):
        return self.sim.model.geom_name2id(geom_name)

    def set_random_parameters(self):
        self.set_parameters(self.sample_parameters())
        self.set_obstacle_position(self.sample_obstacle_position())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""
        lower_bound = self.original_masses[1:] * 0.4
        upper_bound = self.original_masses[1:] * 1.2
        return np.random.uniform(lower_bound, upper_bound)

    def sample_obstacle_position(self):
        """Sample new obstacle position"""
        # lower_bound = self.original_obstacle_positions[:, 0] - 0.5  # - 0.5
        # upper_bound = self.original_obstacle_positions[:, 0] + 0.5  # + 0.5
        lower_bound = self.original_obstacle_position - 0.5
        upper_bound = self.original_obstacle_position + 0.5
        return np.random.uniform(lower_bound, upper_bound)

    def set_parameters(self, task):
        super().set_parameters(task)

    def get_parameters(self):
        masses = np.array(self.sim.model.body_mass[1:])
        # obstacles = np.array(self.sim.model.geom_pos[[self.get_geom_id(
        #    'obstacle_1'), self.get_geom_id('obstacle_2')], 0])
        obstacle = np.array(
            self.sim.model.geom_pos[self.get_geom_id('obstacle_1')])
        # height = np.array(
        #    self.sim.model.geom_pos[self.get_geom_id('obstacle_1')])
        return np.concatenate([masses, obstacle])

    def set_obstacle_position(self, new_position):
        """Set new obstacle positions"""
        self.sim.model.geom_pos[self.get_geom_id(
            'obstacle_1')][0] = new_position
        # self.sim.model.geom_pos[self.get_geom_id(
        #   'obstacle_2')][0] = new_positions[1]

    # def set_obstacle_height(self, new_height):
    #    self.sim.model.geom_size[self.get_geom_id('obstacle_1')]

    def has_collision_with_obstacles(self):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if geom1 in self.get_obstacle_ids() or geom2 in self.get_obstacle_ids():
                return True
        return False

    # def get_obstacle_ids(self):
    #    return [self.get_geom_id('obstacle_1'), self.get_geom_id('obstacle_2')]

    def step(self, a):
        """Step the simulation to the next timestep"""
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        # alive_bonus = 2.5
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        # reward -= 1e-4 * np.square(a).sum()

        obs_x_pos = self.sim.model.geom_pos[self.get_geom_id('obstacle_1')][0]

        if posafter > obs_x_pos:
            reward += posafter

        # obs_x_pos = [self.sim.model.geom_pos[obs_id][0]
        # for obs_id in self.get_obstacle_ids()]

        # if posafter > min(obs_x_pos):
        #    reward += 0.5*posafter
        #    if posafter > max(obs_x_pos):
        #        reward += posafter

        # collision_penalty = 0
        # if self.has_collision_with_obstacles():
        #    collision_penalty = 0.5

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (
            np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state, including obstacle positions"""
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],  # Hopper position
            self.sim.data.qvel.flat,      # Hopper velocity
            # self.sim.model.geom_pos[self.get_geom_id(
            #    'obstacle_1')].flat,  # Obstacle 1 position (3D)
            # self.sim.model.geom_pos[self.get_geom_id(
            #    'obstacle_2')].flat   # Obstacle 2 position (3D)
            [self.sim.model.geom_pos[self.get_geom_id(
                'obstacle_1')][0]]  # Obstacle 1 x-position
            # [self.sim.model.geom_pos[self.get_geom_id(
            #   'obstacle_2')][0]]   # Obstacle 2 x-position
        ])
        return obs

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
        super().viewer_setup()


class ADRCallbackObs(BaseCallback):
    def __init__(self, train_env, test_env, thresh_high=900, thresh_low=300, buffer_size=10, prob_adr=1, verbose=0):
        super(ADRCallbackObs, self).__init__(verbose)

        self.train_env = train_env
        self.test_env = test_env
        self.buffer_size = buffer_size
        self.thresh_high = thresh_high
        self.thresh_low = thresh_low
        self.prob_adr = prob_adr

        model = self.train_env.envs[0].unwrapped.model
        self.original_masses = np.copy(model.body_mass[2:])
        self.obstacle_1_id = self.get_geom_id('obstacle_1')
        # self.obstacle_2_id = self.get_geom_id('obstacle_2')
        # save only x positions
        # self.original_obstacle_positions = np.copy(
        #    model.geom_pos[[self.obstacle_1_id, self.obstacle_2_id], 0])
        self.original_obstacle_position = np.copy(
            model.geom_pos[self.obstacle_1_id][0])
        # id: [lower bound, upper bound] for each parameter
        self.params = {f"mass_{i}": [self.original_masses[i]*0.8, self.original_masses[i]*1.1]
                       for i in range(0, len(self.original_masses))}
        # add obstacles pos x to parameters
        # for i in range(1):
        self.params["obstacle_1_x"] = [
            self.original_obstacle_position - 0.5,
            self.original_obstacle_position + 0.5
        ]

        self.performance_buffers = {
            param: [[], []] for param in self.params}
        self.entropies = []
        self.count = 0

    def get_geom_id(self, geom_name):
        return self.train_env.envs[0].unwrapped.model.geom_name2id(geom_name)

    def randomize_environment(self):

        masses_lower_bound = [bounds[0] for param,
                              bounds in self.params.items() if "mass" in param]
        masses_upper_bound = [bounds[1] for param,
                              bounds in self.params.items() if "mass" in param]
        obs_lower_bound = [bounds[0] for param,
                           bounds in self.params.items() if "obstacle" in param]
        obs_upper_bound = [bounds[1] for param,
                           bounds in self.params.items() if "obstacle" in param]

        model = self.train_env.envs[0].unwrapped.model
        model.body_mass[2:] = np.random.uniform(
            masses_lower_bound, masses_upper_bound)

        # while (model.geom_pos[self.obstacle_2_id][0] - model.geom_pos[self.obstacle_1_id][0] < 1.2):
        #   model.geom_pos[[self.obstacle_1_id, self.obstacle_2_id],
        #                  0] = np.random.uniform(obs_lower_bound, obs_upper_bound)
        # print("distance between obstacles: {}".format(
        #    model.geom_pos[self.obstacle_2_id][0] - model.geom_pos[self.obstacle_1_id][0]))
        model.geom_pos[self.obstacle_1_id][0] = np.random.uniform(
            obs_lower_bound, obs_upper_bound)

    def adjust_boundary(self, param_id, avg_reward):
        bounds = self.params[param_id]
        if avg_reward >= self.thresh_high:
            # self.thresh_high += 100
            # Expand the boundaries
            if "mass" in param_id:
                bounds[0] *= 0.95
                bounds[1] *= 1.05
            elif "obstacle" in param_id:
                # Handle positive and negative bounds for obstacles
                bounds[0] = bounds[0] * \
                    1.05 if bounds[0] < 0 else bounds[0] * 0.95
                bounds[1] = bounds[1] * \
                    0.95 if bounds[1] < 0 else bounds[1] * 1.05
        elif avg_reward <= self.thresh_low:
            # Shrink the boundaries
            if "mass" in param_id:
                bounds[0] *= 1.05
                bounds[1] *= 0.95
            elif "obstacle" in param_id:
                # Handle positive and negative bounds for obstacles
                bounds[0] = bounds[0] * \
                    0.95 if bounds[0] < 0 else bounds[0] * 1.05
                bounds[1] = bounds[1] * \
                    1.05 if bounds[1] < 0 else bounds[1] * 0.95

        if "mass" in param_id:
            mass_id = int(param_id.split('_')[1])
            original_mass = self.original_masses[mass_id]
            # bounds[0] = max(0.1, bounds[0])
            # bounds[1] = min(2.0, bounds[1])
            bounds[0] = max(original_mass*0.1, bounds[0])
            bounds[1] = min(original_mass*2, bounds[1])
        elif "obstacle" in param_id:
            # obs_id = int(param_id.split('_')[1]) - 1
            original_x = self.original_obstacle_position
            bounds[0] = max(original_x - 1, bounds[0])
            bounds[1] = min(original_x + 1, bounds[1])

    def _on_step(self) -> bool:
        if "dones" in self.locals:
            # Check if any environment in the vectorized env is done
            if any(self.locals["dones"]):
                self.randomize_environment()
                # counts = [count+1 for count in range(self.count)]
                # print(counts)
                # print(self.entropies)

                # print('env randomized')
        return True

    def _on_rollout_end(self) -> None:
        if np.random.rand() < self.prob_adr:
            # perform ADR
            self.randomize_environment()
            chosen_param_id = np.random.choice(
                list(self.params.keys()))
            model = self.train_env.envs[0].unwrapped.model
            buffer_pos = 0  # 0 -> lower bounds, 1 -> upper bounds
            prob = np.random.rand()
            if "mass" in chosen_param_id:
                chosen_mass_index = int(chosen_param_id.split('_')[1])
                if prob < 0.5:
                    # assign lower bound
                    model.body_mass[2:][chosen_mass_index] = self.params[chosen_param_id][0]
                else:
                    # assing upper bound
                    model.body_mass[2:][chosen_mass_index] = self.params[chosen_param_id][1]
                    buffer_pos = 1
            elif "obstacle" in chosen_param_id:
                chosen_obs_index = int(
                    chosen_param_id.split('_')[1]) - 1
                if prob < 0.5:
                    # assign lower bound
                    model.geom_pos[self.obstacle_1_id +
                                   chosen_obs_index][0] = self.params[chosen_param_id][0]
                else:
                    # assign upper bound
                    model.geom_pos[self.obstacle_1_id +
                                   chosen_obs_index][0] = self.params[chosen_param_id][1]
                    buffer_pos = 1

            # callback = RandomizeObstaclesCallback(self.train_env)
            mean_reward, _ = evaluate_policy(
                self.model, self.train_env, n_eval_episodes=50)
            buffer = self.performance_buffers[chosen_param_id][buffer_pos]
            buffer.append(mean_reward)
            if len(buffer) >= self.buffer_size:
                avg = np.mean(buffer)
                buffer.clear()
                self.adjust_boundary(
                    param_id=chosen_param_id, avg_reward=avg)

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


class RandomizeObstaclesCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(RandomizeObstaclesCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        return True

    def __call__(self, locals_, globals_):
        """
        Randomize obstacle positions at the start of each episode.
        """
        if locals_["dones"][0]:  # 'dones' is True at the end of an episode
            self.env.envs[0].unwrapped.set_obstacle_position(
                self.env.envs[0].unwrapped.sample_obstacle_position()
            )


"""
    Registered environments
"""
gym.envs.register(
    id="CustomHopperWithObstacles-v0",
    entry_point="%s:CustomHopperWithObstacles" % __name__,
    max_episode_steps=500,

)

gym.envs.register(
    id="CustomHopperWithObstacles-source-v0",
    entry_point="%s:CustomHopperWithObstacles" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.envs.register(
    id="CustomHopperWithObstacles-target-v0",
    entry_point="%s:CustomHopperWithObstacles" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"}
)
