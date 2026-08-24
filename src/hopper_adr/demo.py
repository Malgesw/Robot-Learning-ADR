"""Roll out a random policy on the Gym Hopper environment.

Useful as a smoke test that the environment and its MuJoCo assets are
installed and registered correctly before running actual training.
"""
import gym
from hopper_adr.envs.custom_hopper import *


def main():
    render = False

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    # masses of each link of the Hopper
    print('Dynamics parameters:', env.get_parameters())

    n_episodes = 500

    for ep in range(n_episodes):
        done = False
        state = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            # Step the simulator to the next timestep
            state, reward, done, info = env.step(action)

            if render:
                env.render()


if __name__ == '__main__':
    main()
