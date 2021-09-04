import time
import gym
import numpy as np
import concurrent.futures
import os
import sys

from stable_baselines3 import PPO
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
RACETRACK = 'SILVERSTONE'

root_path = reduce(lambda path, _: os.path.dirname(path), range(3), os.path.dirname(os.path.realpath(__file__)))
env_path = os.path.join(root_path, 'gym', 'f110_gym', 'envs')
map_path = os.path.join(root_path, 'pkg', 'src', 'pkg', 'maps')
sys.path.append(env_path)
from f110_env import F110Env


class GymRunner(object):

    def __init__(self, racetrack, drivers):
        self.racetrack = racetrack
        self.drivers = drivers

    def run(self):
        # load map
        # env = gym.make('f110_gym:f110-v0',
        #                map="{}/maps/{}".format(current_dir, RACETRACK),
        #                map_ext=".png", num_agents=len(drivers))
        print(f'Initializing env with: {map_path + "/" + RACETRACK, ".png", len(drivers)}')
        env = F110Env(map_path + '/' + RACETRACK, '.png', len(drivers))


        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=10000)
        print("done")

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
