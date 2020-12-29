#!/bin/bash 

#PiPy packages
import os
import argparse
import pprint as pp
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Flatten, GRU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K

import threading
from threading import Thread, Lock
import time

from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from scipy.io import savemat
import datetime
import json
import matplotlib.pyplot as plt

#Local Packages
from env.SEIR_v0_2 import SEIR_v0_2
from RL_algo.PPO import AC_model, PPOAgent

#0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#/home/kosaraju/anaconda3/envs/tf-gpu/bin/python $DIR/run.py --gamma=

# save
#
#

#---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for PPO agent')
    #loading the environment to get it default params
    env = SEIR_v0_2()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    #--------------------------------------------------------------------
    #general params
    parser.add_argument('--summary_dir', help='directory for saving and loading model and other data', default='./results')
    parser.add_argument('--use_gpu', help='weather to use gpu or not', type = bool, default=True)
    parser.add_argument('--save_model', help='Saving model from summary_dir', type = bool, default=True)
    parser.add_argument('--load_model', help='Loading model from summary_dir', type = bool, default=True)
    parser.add_argument('--random_seed', help='seeding the random number generator', default=1754)

    #PPO agent params
    parser.add_argument('--max_episodes', help='max number of episodes', type = int, default=1200)
    parser.add_argument('--exp_name', help='Name of the experiment', default='seir')
    parser.add_argument('--gamma', help='models the long term returns', type =float, default=0.95)
    parser.add_argument('--traj_per_episode', help='trajectories per episode', type = int, default=10)

    #model/env paramerterss
    parser.add_argument('--sim_length', help='Total number of days', type = int, default=140)
    parser.add_argument('--sampling_time', help='Sampling time (in days) used for the environment', type = int, default=7)
    parser.add_argument('--discretization_time', help='discretization time (in minutes) used for the environment ', type = int, default=5)
    parser.add_argument('--env_weight', help='0-Social cost, 1-economic cost', type = float, default=0.7)

    #Network parameters
    parser.add_argument('--params', help='Hiden layer parameters', type = int, default=400)
    parser.add_argument('--lr', help='learning rate', type = float, default=3e-4)
    parser.add_argument('--EPOCHS', help='Number of epochs for training',type =int, default=10)
    parser.add_argument('--EPSILON', help='Clip parameter of PPO algorithm, between 0-1',type =float, default=0.1)
    parser.add_argument('--C', help='Controls the entropy, exploration',type =float, default=5e-2)
    parser.add_argument('--rnn', help='Use reccurent neural networks?', type = bool, default=True)
    parser.add_argument('--rnn_steps', help='if rnn = True, then how many time steps do we see backwards',type =int, default=2)

    args = vars(parser.parse_args())

    #saving the arguments to a text file
    try:
        args_path = args['summary_dir']+'/args.txt'
        with open(args_path, 'w') as file:
            file.write(json.dumps(args)) # use `json.loads` to do the reverse
    except:
        pass

    pp.pprint(args)
    try:
        os.makedir(args['summary_dir'])
    except:
        pass
    
    #setting random seed
    np.random.seed(args['random_seed'])
    random.seed(args['random_seed'])
    tf.random.set_seed(args['random_seed'])

    env = SEIR_v0_2(
        discretizing_time = args['discretization_time'], 
        sampling_time     = args['sampling_time'], 
        sim_length        = args['sim_length']
        )
    test_env = SEIR_v0_2(
        discretizing_time = args['discretization_time'], 
        sampling_time     = args['sampling_time'], 
        sim_length        = args['sim_length'])
    

    env.weight, test_env.weight= args['env_weight'], args['env_weight']
    env.seed(args['random_seed'])
    test_env.seed(args['random_seed'])

    agent = PPOAgent(
        env              =   env, 
        test_env         =   test_env, 
        exp_name         =   args['exp_name'], 
        EPSIODES         =   args['max_episodes'], 
        lr               =   args['lr'], 
        EPOCHS           =   args['EPOCHS'],
        path             =   args['summary_dir'],
        gamma            =   args['gamma'],
        traj_per_episode =   args['traj_per_episode'],
        EPSILON          =   args['EPSILON'],
        C                =   args['C'],
        rnn              =   args['rnn'],
        rnn_steps        =   args['rnn_steps']
        )       
    
    # try:
    #     agent.load()
    # except:
    #     pass

    agent.run()

    #testing
    print("testing the agent")
    agent.test(savefig_filename='fin_plot.pdf')
    data = dict(
        states   = agent.test_env.state_trajectory,
        actions  = agent.test_env.action_trajectory,
        rewards  = agent.test_env.rewards,
        scores   = agent.scores,
        averages = agent.averages
    )
    file_name = args['summary_dir'] + "/" + 'data_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat'
    savemat(file_name, data)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize = (8,12))
    axes[0].plot(agent.averages)
    axes[0].set_ylabel('average_scores', fontsize=15)

    axes[1].plot(agent.scores)
    axes[1].set_ylabel('scores', fontsize=15)
    axes[1].set_xlabel('Episodes', fontsize=15)
    savefig_filename = args['summary_dir'] + "/" + 'SCORES.pdf'
    plt.savefig(savefig_filename, format = 'pdf')
