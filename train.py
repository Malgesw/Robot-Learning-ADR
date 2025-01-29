"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
from env.custom_hopper import *
from env.custom_hopper_obs import CustomHopperWithObstacles, ADRCallbackObs, RandomizeObstaclesCallback

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dirs = {'log_dir': './logs', 'models_dir': './models',
        'images_dir': './images', 'test_log_dir': './test_logs',
        'log_dir_obs': './logs_obs', 'test_log_dir_obs': './test_logs_obs',
        'udr_log_dir': './udr_logs', 'udr_test_log_dir': './udr_test_logs',
        'udr_log_dir_obs': './udr_logs_obs/', 'udr_test_log_dir_obs': './udr_test_logs_obs',
        'adr_log_dir': './adr_logs', 'adr_test_log_dir': './adr_test_logs',
        'adr_log_dir_obs': './adr_logs_obs/', 'adr_test_log_dir_obs': './adr_test_logs_obs'}


def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)


def create_model(args, env):
    if args.algo == 'ppo':
        model = PPO("MlpPolicy", env, learning_rate=args.lr,
                    batch_size=args.batch_size, n_epochs=args.num_epochs, seed=args.seed, verbose=1)
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model


def load_model(args, env):

    env = VecNormalize.load(
        './models/vecNormalize{}UDR{}ADR{}.pkl'.format(args.train_env, args.udr, args.adr), env)
    env.training = False
    env.norm_reward = False

    if args.algo == 'ppo':
        model = PPO.load('./models/{}{}Timesteps{}Lr{}Epochs{}Bsize{}UDR{}ADR{}'
                         .format(args.algo, args.train_env, args.total_timesteps, args.lr, args.num_epochs, args.batch_size, args.udr, args.adr), env=env)
        print('./models/{}{}Timesteps{}Lr{}Epochs{}Bsize{}UDR{}ADR{}'.format(
            args.algo, args.train_env, args.total_timesteps, args.lr, args.num_epochs, args.batch_size, args.udr, args.adr))
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model, env


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, args, title="Learning Curve"):
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x[x <= args.total_timesteps], y[x <= args.total_timesteps])
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed ({})".format(args.algo))
    plt.savefig('./images/trainingResults{}{}Timesteps{}Lr{}Epochs{}Bsize{}UDR{}ADR{}.png'
                .format(args.algo, args.train_env, args.total_timesteps, args.lr, args.num_epochs, args.batch_size, args.udr, args.adr))
    plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true',
                        help="Perform a test directly")
    parser.add_argument("--test_env", type=str, default="CustomHopper-target-v0",
                        help="Testing environment [CustomHopper-source-v0, CustomHopper-target-v0, CustomHopperWithObstacles-source-v0, CustomHopperWithObstacles-target-v0]")
    parser.add_argument("--train_env", type=str, default="CustomHopper-source-v0",
                        help="Training environment [CustomHopper-source-v0, CustomHopper-target-v0, CustomHopperWithObstacles-source-v0, CustomHopperWithObstacles-target-v0]")
    parser.add_argument("--total_timesteps", type=int, default=25000,
                        help="The total number of samples to train on")
    parser.add_argument(
        "--render_test", action='store_true', help="Render test")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--algo', default='ppo',
                        type=str, help='RL Algo [ppo]')
    parser.add_argument('--lr', default=0.0003,
                        type=float, help='Learning rate')
    parser.add_argument('--gradient_steps', default=-1, type=int,
                        help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='Batch size')
    parser.add_argument('--num_epochs', default=10,
                        type=int, help='Training epochs')
    parser.add_argument('--test_episodes', default=100,
                        type=int, help='# episodes for test evaluations')
    parser.add_argument("--udr", action='store_true',
                        help="Use Uniform Domain Randomization")
    parser.add_argument("--adr", action='store_true',
                        help="Use Automatic Domain Randomization")
    args = parser.parse_args()

    set_seed(args.seed)
    env = gym.make(args.train_env)  # source domain
    t_env = gym.make(args.test_env)  # target domain

    for dir in dirs.values():
        os.makedirs(dir, exist_ok=True)

    obs_string = ''
    t_obs_string = ''
    if "Obstacles" in args.train_env:
        obs_string = '_obs'
    if "Obstacles" in args.test_env:
        t_obs_string = '_obs'

    if args.udr:
        env = Monitor(env, dirs['udr_log_dir{}'.format(obs_string)])
        t_env = Monitor(t_env, dirs['udr_test_log_dir{}'.format(t_obs_string)])
    elif args.adr:
        env = Monitor(env, dirs['adr_log_dir{}'.format(obs_string)])
        t_env = Monitor(t_env, dirs['adr_test_log_dir{}'.format(t_obs_string)])
    else:
        env = Monitor(env, dirs['log_dir{}'.format(obs_string)])
        t_env = Monitor(t_env, dirs['test_log_dir{}'.format(t_obs_string)])

    if not args.test:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    t_env = DummyVecEnv([lambda: t_env])

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    # masses of each link of the Hopper
    # print('Dynamics parameters:', env.envs[0].get_parameters())
    # print('Dynamics parameters: ', env.get_parameters())

    if not args.test:

        if args.adr:
            if "Obstacles" in args.train_env:
                callback = ADRCallbackObs(train_env=env, test_env=t_env)
            else:
                callback = ADRCallback(train_env=env, test_env=t_env)
        else:
            callback = None

        model = create_model(args, env)
        checkpoint_callback = CheckpointCallback(
            save_freq=200000, save_path='./model_checkpoints/')
        if not args.adr:
            model.learn(total_timesteps=args.total_timesteps,
                        callback=checkpoint_callback)
        else:
            model.learn(total_timesteps=args.total_timesteps,
                        callback=[callback, checkpoint_callback])
        env.save(
            './models/vecNormalize{}UDR{}ADR{}.pkl'.format(args.train_env, args.udr, args.adr))
        model.save('./models/{}{}Timesteps{}Lr{}Epochs{}Bsize{}UDR{}ADR{}'
                   .format(args.algo, args.train_env, args.total_timesteps, args.lr, args.num_epochs, args.batch_size, args.udr, args.adr))

        if args.udr:
            plot_results(dirs['udr_log_dir{}'.format(obs_string)], args)
        elif args.adr:
            plot_results(dirs['adr_log_dir{}'.format(obs_string)], args)
        else:
            plot_results(dirs['log_dir{}'.format(obs_string)], args)

        # mean_reward, std_reward = evaluate_policy(
         #   model, t_env, n_eval_episodes=args.test_episodes, render=args.render_test)
        # print("Test reward (avg +/- std): ({} +/- {}) - Num episodes: {}".format(
        #    mean_reward, std_reward, args.test_episodes))
    else:

        model, t_env = load_model(args, t_env)
        model = PPO.load(
            './model_checkpoints/rl_model_600000_steps.zip', env=t_env)
        if "Obstacles" in args.test_env:
            # test_callback = RandomizeObstaclesCallback(t_env)
            test_callback = None
        else:
            test_callback = None
        mean_reward, std_reward = evaluate_policy(
            model, t_env, n_eval_episodes=args.test_episodes, render=args.render_test, callback=test_callback)
        print("Test reward (avg +/- std): ({} +/- {}) - Num episodes: {}".format(
            mean_reward, std_reward, args.test_episodes))


if __name__ == '__main__':
    main()
