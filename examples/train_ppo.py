import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace

from numba import njit

from pyglet.gl import GL_POINTS

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecCheckNan
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def main():
    """
    main entry point
    """

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312,
            'tlad': 0.82461887897713965, 'vgain': 1.375}  # 0.90338203837889}

    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4, starts=np.array([[conf.sx, conf.sy, conf.stheta]]), checkpoints=[(-1.043387605536935, 6.182245854975881), (
        -11.119006505707265, 12.196284277915074), (-23.699395891476858, 16.020208825506806), (-35.28917215489346, 24.549809980247254), (-47.28584233778769, 7.649503337057018), (-34.33850049129457, -7.868348672295764), (-9.921228138520522, -7.54480924235594), (conf.sx, conf.sy)])
    env.add_render_callback(render_callback)

    # set up logger
    check_env(env)
    # env_id = "f110_gym:f110-v0"
    # num_cpu = 4  # Number of processes to use
    # env = VecMonitor(env)
    # env = VecCheckNan(env, raise_exception=True)

    model = PPO.load("ppo_f110v0", env=env)

    # Instantiate the agent
    model = PPO('MlpPolicy',
                env,
                verbose=2,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                learning_rate=0.0003,
                max_grad_norm=0.5)
    # # Train the agent and display a progress bar
    # model.learn(total_timesteps=int(2e6), progress_bar=True)
    # Save the agent
    # model.save("ppo_f110v0")

    model.set_env(env)
    obs = env.reset()
    env.render()

    laptime = 0.0
    start = time.time()
    done = False

    while not done:
        action = model.predict(obs)
        print(action)
        obs, step_reward, done, info = env.step(action[0])
        env.render(mode='human')

    print('Sim elapsed time:', env.current_time,
          'Real elapsed time:', time.time()-start)


if __name__ == '__main__':
    main()
