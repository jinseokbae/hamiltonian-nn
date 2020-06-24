# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import gym
# import gym_capoo_pendulum
import pdb
import scipy, scipy.misc
from skimage.transform import resize

import os, sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import to_pickle, from_pickle


def get_theta(obs):
    '''Transforms coordinate basis from the defaults of the gym pendulum env.'''
    theta = np.arctan2(obs[0], -obs[1])
    theta = theta + np.pi / 2
    theta = theta + 2 * np.pi if theta < -np.pi else theta
    theta = theta - 2 * np.pi if theta > np.pi else theta
    return theta


def preproc(X, side):
    '''Crops, downsamples, desaturates, etc. the rgb pendulum observation.'''
    X = X[...,0][226:-106,166:-166] - X[...,1][226:-106,166:-166]
    return resize(X, [int(side), side])


# def preproc(X, side):
#     '''Crops, downsamples, desaturates, etc. the rgb pendulum observation.'''
#     X = X[..., 0][440:-220, 330:-330] - X[..., 1][440:-220, 330:-330]
#     return resize(X, [int(side), side]) / 255.


def sample_gym(seed=0, timesteps=103, trials=2, side=28, min_angle=0., max_angle=np.pi / 6,
               verbose=False, env_name='Pendulum-v0'):
    gym_settings = locals()
    if verbose:
        print("Making a dataset of pendulum pixel observations.")
        print("Edit 5/20/19: you may have to rewrite the `preproc` function depending on your screen size.")
    # env = gym.make(env_name)
    env = gym.make('gym_capoo_pendulum:capoo-pendulum-v0')
    env.reset()
    env.seed(seed)

    canonical_coords, frames = [], []
    for step in range(trials * timesteps):

        if step % timesteps == 0:
            angle_ok = False

            while not angle_ok:
                obs = env.reset()
                theta_init = np.abs(get_theta(obs))
                if verbose:
                    print("\tCalled reset. Max angle= {:.3f}".format(theta_init))
                if theta_init > min_angle and theta_init < max_angle:
                    angle_ok = True

            if verbose:
                print("\tRunning environment...")

        frames.append(preproc(env.render('rgb_array'), side))
        print('Step: {}'.format(step))
        if step == 0 :
            import imageio
            temp = frames[0]*255
            imageio.imwrite('./check_cropped.png',temp.astype('uint8'))
        obs, _, _, _ = env.step([0.])
        theta, dtheta = get_theta(obs), obs[-1]

        # The constant factor of 0.25 comes from saying plotting H = PE + KE*c
        # and choosing c such that total energy is as close to constant as
        # possible. It's not perfect, but the best we can do.
        canonical_coords.append(np.array([theta, 0.25 * dtheta]))

    canonical_coords = np.stack(canonical_coords).reshape(trials * timesteps, -1)
    frames = np.stack(frames).reshape(trials * timesteps, -1)
    env.close()
    return canonical_coords, frames, gym_settings

def make_ball_dataset(test_split=0.2, **kwargs):
    frame_num = 303
    episode_num = 200
    timesteps = 100
    canonical_coords, frames= [], []
    for episode in range(0, episode_num):
        print('Loading episode {0:03d}...'.format(episode))
        coords_dir = 'dataset/ball/coords_64/coords_episode_{0:03d}.txt'.format(episode)
        frames_dir = 'dataset/ball/images_64/ball_episode_{0:03d}.txt'.format(episode)

        myCoords = np.genfromtxt(coords_dir, delimiter=',')
        myCoords = myCoords.reshape(4,-1).astype(np.float32)


        canonical_coords.append(myCoords[:, 100:].T) # myCoords.T.shape = (303,4)
        myCoords = None

        myFrames = np.genfromtxt(frames_dir, delimiter=',').astype(np.uint8)
        frames.append(myFrames[100: ,:]) # myPixels.shape = (303, 64*64*3)
        myFrames = None

    canonical_coords = np.stack(canonical_coords, axis=0).astype(np.float32) # coords.shape = (200,303,4) -> (E,T,(x,y,xdot,ydot))
    frames = np.stack(frames, axis=0).astype(np.uint8) # pixels.shape = (200,303,64*64*3) -> (E,T,Image)
    coords, dcoords = [], []
    pixels, dpixels = [], []
    next_pixels, next_dpixels = [], []
    count = 0
    for cc, pix in zip (canonical_coords, frames):
        print('Processing episode {0:03d}'.format(count))
        count += 1
        cc = cc[1:]
        dcc = cc[1:] - cc[:-1]
        cc = cc[1:]

        p = np.concatenate([pix[1:], pix[1:]-pix[:-1]], axis=-1)

        dp = p[1:] - p[:-1]
        p = p[1:]

        next_p, next_dp = p[1:], dp[1:]
        p, dp = p[:-1], dp[:-1]
        cc, dcc = cc[:-1], dcc[:-1]

        coords.append(cc)
        dcoords.append(dcc)
        pixels.append(p)
        dpixels.append(dp)
        next_pixels.append(next_p)
        next_dpixels.append(next_dp)

    # concatenate across trials
    data = {'coords': coords, 'dcoords': dcoords,
            'pixels': pixels, 'dpixels': dpixels,
            'next_pixels': next_pixels, 'next_dpixels': next_dpixels}
    data = {k: np.concatenate(v) for k, v in data.items()}

    # make a train/test split
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data

    settings = {}
    settings['env_name'] = 'one-ball-64'
    settings['verbose'] = True
    settings['max_speed'] = (10, 10)
    settings['min_speed'] = (-10, -10)
    settings['side'] = 64
    settings['trials'] = episode_num
    settings['timesteps'] = timesteps
    settings['seed'] = 0
    data['meta'] = settings

    return data


def make_gym_dataset(test_split=0.2, **kwargs):
    '''Constructs a dataset of observations from an OpenAI Gym env'''
    canonical_coords, frames, gym_settings = sample_gym(**kwargs)
    coords, dcoords = [], []  # position and velocity data (canonical coordinates)
    pixels, dpixels = [], []  # position and velocity data (pixel space)
    next_pixels, next_dpixels = [], []  # (pixel space measurements, 1 timestep in future)

    trials = gym_settings['trials']

    for cc, pix in zip(np.split(canonical_coords, trials), np.split(frames, trials)):
        # calculate cc offsets
        cc = cc[1:]
        dcc = cc[1:] - cc[:-1]
        cc = cc[1:]
        # concat adjacent frames to get velocity information
        # now the pixel arrays have same information as canonical coords
        # ...but in a different (highly nonlinear) basis
        p = np.concatenate([pix[1:], pix[1:]-pix[:-1]], axis=-1)

        dp = p[1:] - p[:-1]
        p = p[1:]

        # calculate the same quantities, one timestep in the future
        next_p, next_dp = p[1:], dp[1:]
        p, dp = p[:-1], dp[:-1]
        cc, dcc = cc[:-1], dcc[:-1]

        # append to lists
        coords.append(cc);
        dcoords.append(dcc)
        pixels.append(p)
        dpixels.append(dp)
        next_pixels.append(next_p)
        next_dpixels.append(next_dp)

    # concatenate across trials

    data = {'coords': coords, 'dcoords': dcoords,
            'pixels': pixels, 'dpixels': dpixels,
            'next_pixels': next_pixels, 'next_dpixels': next_dpixels}
    data = {k: np.concatenate(v) for k, v in data.items()}

    # make a train/test split
    split_ix = int(data['coords'].shape[0] * test_split)
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[split_ix:], v[:split_ix]
    data = split_data
    gym_settings['timesteps'] -= 3  # from all the offsets computed above
    data['meta'] = gym_settings
    pdb.set_trace()

    return data


def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns a dataset bult on top of OpenAI Gym observations. Also constructs
    the dataset if no saved version is available.'''

    if experiment_name == "pendulum":
        env_name = "Pendulum-v0"
    elif experiment_name == "acrobot":
        env_name = "Acrobot-v1"
    elif experiment_name =="ball":
        env_name = "one-ball-64"
    else:
        assert experiment_name in ['pendulum']

    path = '{}/{}-pixels-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        if env_name == "one-ball-64":
            data = make_ball_dataset(**kwargs)
        elif env_name == "Pendulum-v0":
            data = make_gym_dataset(**kwargs)
        print(path)
        to_pickle(data, path)

    return data


### FOR DYNAMICS IN ANALYSIS SECTION ###
def hamiltonian_fn(coords):
    k = 1.9  # this coefficient must be fit to the data
    q, p = np.split(coords, 2)
    H = k * (1 - np.cos(q)) + p ** 2  # pendulum hamiltonian
    return H


def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords, 2)
    S = -np.concatenate([dpdt, -dqdt], axis=-1)
    return S