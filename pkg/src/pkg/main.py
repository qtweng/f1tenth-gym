import time
import gym
import numpy as np
import concurrent.futures
import os
import sys

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from functools import reduce

from yaml import scan
# Get ./src/ folder & add it to path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# import your drivers here
from pkg.drivers import PureFTG, DisparityExtender, GapFollower

# choose your drivers here (1-4)
drivers = [PureFTG()]

# choose your racetrack here (SILVERSTONE, SILVERSTONE_OBS)
RACETRACK = 'map0'

root_path = reduce(lambda path, _: os.path.dirname(path), range(3), os.path.dirname(os.path.realpath(__file__)))
env_path = os.path.join(root_path, 'gym', 'f110_gym', 'envs')
map_path = os.path.join(root_path, 'pkg', 'src', 'pkg', 'maps')
sys.path.append(env_path)
from f110_env import F110Env

def make_env( rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        
        env = F110Env(map_path + '/' + RACETRACK, '.png', len(drivers))
        #env.seed(seed + rank)
        return env
    set_random_seed(seed+rank)
    return _init

class GymRunner(object):

    def __init__(self, racetrack, drivers):
        self.racetrack = racetrack
        self.drivers = drivers

    def run(self):
        # load map
        # env = gym.make('f110_gym:f110-v0',
        #                map="{}/maps/{}".format(current_dir, RACETRACK),
        #                map_ext=".png", num_agents=len(drivers))
        # print(f'Initializing env with: {map_path + "/" + RACETRACK, ".png", len(drivers)}')
                
        env = F110Env(map_path + '/' + RACETRACK, '.png', len(drivers))
        #env = SubprocVecEnv([make_env(i) for i in range(8)])
        #env = DummyVecEnv([make_env(i) for i in range(12)])
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env,norm_reward=False )
        env = VecMonitor(env)
        # check_env(env)
        
        # noise objects for td3
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma = 0.01 * np.ones(n_actions))

        # modesl
        #model = PPO('MultiInputPolicy', env, learning_rate=0.03, batch_size=256, verbose=2)
        #model = SAC("MultiInputPolicy", env, verbose=2)
        model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=2)
        model.learn(total_timesteps=400000, log_interval=1)
    
        while True:
            print("done")
            #model.save("ppo_f1tenth")
            model.save("td3_f1tenth")

            #model = PPO.load("ppo_f1tenth")
            #model = SAC.load("sac_f1tenth")
            model = TD3.load("td3_f1tenth")
            model.set_env(env)
            model.learn(total_timesteps=400000, log_interval=1)

        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render(mode='human_fast')
            if done:
                obs = env.reset()

        # specify starting positions of each agent
        # poses = np.array([[0. + (i * 0.75), 0. - (i*1.5), np.radians(60)] for i in range(len(drivers))])

        # obs, step_reward, done, info = env.reset(poses=poses)
        # env.render()

        # laptime = 0.0
        # start = time.time()

        # while not done:
        #     actions = []
        #     futures = []
        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         for i, driver in enumerate(drivers):
        #             futures.append(executor.submit(driver.process_lidar, obs['scans'][i]))
        #             print(len( obs['scans'][i]))
        #     for future in futures:
        #         speed, steer = future.result()
        #         actions.append([steer, speed])
        #     actions = np.array(actions)
        #     obs, step_reward, done, info = env.step(actions)
        #     laptime += step_reward
        #     env.render(mode='human_fast')

        # print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    runner = GymRunner(RACETRACK, drivers)
    runner.run()
