#!/bin/bash 

#PiPy packages
import os
import argparse
import pprint as pp
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np
import tensorflow as tf
import time
from joblib import dump, load
from scipy.io import savemat
import datetime
import json
import matplotlib.pyplot as plt

#Local Packages
from env.SEIR_V0_3 import SEIR_v0_3
from RL_algo.PPO import AC_model, PPOAgent

#to reduce the tensorflow messages
# tf.get_logger().setLevel('WARNING')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

# /home/kosaraju/anaconda3/envs/tf-gpu/bin/python $DIR/run.py --gamma=

#---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for PPO agent')
    #loading the environment to get it default params
    env = SEIR_v0_3()
    start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    #--------------------------------------------------------------------
    #general params
    parser.add_argument('--summary_dir', help='directory for saving and loading model and other data', default='./results/Senario-2/delete_2/')
    parser.add_argument('--use_gpu', help='weather to use gpu or not', type = bool, default=True)
    parser.add_argument('--save_model', help='Saving model from summary_dir', type = bool, default=True)
    parser.add_argument('--load_model', help='Loading model from summary_dir', type = bool, default=True)
    parser.add_argument('--random_seed', help='seeding the random number generator',type = int, default=1123)
    parser.add_argument('--mpi', help='Use Multi-Processing', type = bool, default=True)
    # parser.add_argument('--start_time', help='simulation start time for book keeping', type = str, default=start_time)

    #PPO agent params
    parser.add_argument('--max_episodes', help='max number of episodes', type = int, default=300)
    parser.add_argument('--exp_name', help='Name of the experiment', default='seir')
    parser.add_argument('--gamma', help='models the long term returns', type =float, default=0.995)
    parser.add_argument('--traj_per_episode', help='trajectories per episode', type = int, default=100)
    parser.add_argument('--EPSILON', help='Clip parameter of PPO algorithm, between 0-1',type =float, default=0.1)
    parser.add_argument('--C', help='Controls the entropy, exploration',type =float, default=5e-1)


    #model/env paramerterss
    parser.add_argument('--sim_length', help='Total number of days', type = int, default=175)
    parser.add_argument('--sampling_time', help='Sampling time (in days) used for the environment', type = int, default=7)
    parser.add_argument('--discretization_time', help='discretization time (in minutes) used for the environment ', type = int, default=5)
    parser.add_argument('--env_weight', help='0-Social cost, 1-economic cost', type = float, default=0.5)
    #-
    parser.add_argument('--training_noise', help='Do we train the agent with noisy state', type = bool, default=False)
    parser.add_argument('--training_noise_percent', help='Percentage of training noise', type = float, default=50.)
    parser.add_argument('--training_theta', help='Percentage of training noise', type = float, default=113.92)
    #-
    parser.add_argument('--Validation_noise', help='Do we train the agent with noisy state', type = bool, default=False)
    parser.add_argument('--Validation_noise_percent', help='Percentage of training noise', type = float, default=15.)
    parser.add_argument('--Validation_theta', help='Percentage of training noise', type = float, default=113.92)
    # - E, I, R inital conditions
    parser.add_argument('--E', help='Number of Exposed people', type = int, default=81)
    parser.add_argument('--I', help='Number of Infectious people', type = int, default=138)
    parser.add_argument('--R', help='Number of Recovered people', type = int, default=115)
    
    #Network parameters
    parser.add_argument('--params', help='Hiden layer parameters', type = int, default=400)
    parser.add_argument('--lr', help='learning rate', type = float, default=5e-4)
    parser.add_argument('--EPOCHS', help='Number of epochs for training',type =int, default=20)
    parser.add_argument('--rnn', help='Use reccurent neural networks?', type = bool, default=True)
    parser.add_argument('--rnn_steps', help='if rnn = True, then how many time steps do we see backwards',type =int, default=1)

    args = vars(parser.parse_args())
    
    # making the summary directory
    try:
        os.mkdir(args['summary_dir'])
    except:
        pass

    pp.pprint(args)

    
    #setting random seed
    np.random.seed(args['random_seed'])
    random.seed(args['random_seed'])
    tf.random.set_seed(args['random_seed'])

    #inital state
    inital_state = [10e5 - args['E']- args['I'] - args['R'], args['E'], args['I'], args['R']]
    env = SEIR_v0_3(
        discretizing_time = args['discretization_time'], 
        sampling_time     = args['sampling_time'], 
        sim_length        = args['sim_length'],
        weight            = args["env_weight"],
        theta             = args['training_theta'],
        inital_state      = inital_state,
        noise             = args['training_noise'],
        noise_percent     = args['training_noise_percent'],
        validation        = False   
        )
    
    test_env = SEIR_v0_3(
        discretizing_time = args['discretization_time'], 
        sampling_time     = args['sampling_time'], 
        sim_length        = args['sim_length'],
        weight            = args["env_weight"],
        theta             = args['Validation_theta'],
        inital_state      = inital_state,
        noise             = args['Validation_noise'],
        noise_percent     = args['Validation_noise_percent'],   
        validation        = True
        )
    
    env_args = {
        'discretizing_time' : args['discretization_time'], 
        'sampling_time'     : args['sampling_time'], 
        'sim_length'        : args['sim_length'],
        'weight'            : args["env_weight"],
        'theta'             : args['training_theta'],
        'inital_state'      : inital_state,
        'noise'             : args['training_noise'],
        'noise_percent'     : args['training_noise_percent'],
        'validation'        : False   }

    # n_threads = 5
    env_list = [SEIR_v0_3(**env_args) for _ in range(args['traj_per_episode'])]

    # env.weight, test_env.weight= args['env_weight'], args['env_weight']
    env.seed(args['random_seed'])
    test_env.seed(args['random_seed'])
    [env_from_list.seed(args['random_seed']) for env_from_list in env_list]

    agent = PPOAgent(
        env              =   env, 
        test_env         =   test_env, 
        env_list         =   env_list,
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
        rnn_steps        =   args['rnn_steps'],
        use_mpi          =   args['mpi']
        )       
    

    t0= time.perf_counter()
    if not args['mpi']:
        agent.run()
    else:
        agent.run_mpi()
    t1 = time.perf_counter()
    print("Time taken to run the experiment: ",t1-t0)
    # testing
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


    #saving the arguments to a text file
    end_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    args['start_time'] = start_time
    args['end_time'] = end_time
    args['Total _time'] = (t1-t0)/(60*60)
    try:
        args_path = args['summary_dir']+'/args.txt'
        with open(args_path, 'w') as file:
            file.write(json.dumps(args)) # use `json.loads` to do the reverse
    except:
        pass 