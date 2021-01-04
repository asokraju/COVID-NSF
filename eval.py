#!/bin/bash 

#PiPy packages
import os
from os import walk, listdir

import argparse
import pprint as pp
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
from seaborn import palettes
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
import pandas as pd
import seaborn as sns

#Local Packages
from env.SEIR_v0_2 import SEIR_v0_2
from RL_algo.PPO import AC_model_new, PPOAgent


def data_per_exp(sub_dir, noise = 0):

    # loading args dict
    args_file = sub_dir + 'args.txt'
    with open(args_file, 'r') as file:
        args = json.load(file)
    args['summary_dir'] = sub_dir

    #rng
    np.random.seed(args['random_seed'])
    random.seed(args['random_seed'])
    tf.random.set_seed(args['random_seed'])

    #model_name
    Model_name  = 'PPO_' + args['exp_name'] +'_'

    #loading the standard scalar model
    ss_path = sub_dir + Model_name + 'std_scaler.bin' 
    std_scalar = load(ss_path)

    #loading the environment
    test_env = SEIR_v0_2(
        discretizing_time = args['discretization_time'], 
        sampling_time     = args['sampling_time'], 
        sim_length        = args['sim_length'])
    test_env.weight = args['env_weight']
    test_env.set_state(np.array([98588, 208, 449, 755]))
    # test_env.theta = np.full(shape=1, fill_value=3.37, dtype=float)

    # setting the same initial state for all senarios
    init_I = 200 # np.random.randint(200, high=500)
    init_E = 1200 # np.random.randint(900, high=1200)
    init_S = test_env.popu - init_I - init_E
    test_env.set_state( np.array([init_S, init_E, init_I, 0], dtype=float))
    #loading the actor and critic and their weights
    Actor_name  = sub_dir + "/" + Model_name + '_Actor.h5'
    Critic_name = sub_dir + "/" + Model_name + '_Critic.h5'
    Actor_Critic_name = sub_dir + "/" + Model_name + '_Actor_Critic.h5' 
    Actor, Critic, Actor_Critic = AC_model_new(
            input_shape    =    test_env.observation_space.shape[0], 
            action_dim     =    3, 
            lr             =    args['lr'], 
            EPSILON        =    args['EPSILON'], 
            C              =    args['C'],
            rnn            =    args['rnn'],
            rnn_steps      =    args['rnn_steps']
            )
    if os.path.isfile(Actor_name):
        # print('loading Actor weights')
        #Actor_name = self.path + "/" + self.Model_name + '_Actor.h5'
        Actor = load_model(Actor_name, compile=False)
    if os.path.isfile(Critic_name):
        # print('loading Critic weights')
        #Actor_name = self.path + "/" + self.Model_name + '_Actor.h5'
        Critic = load_model(Critic_name, compile=False)
    if os.path.isfile(Actor_Critic_name):
        # print('loading Actor_Critic weights')
        #Actor_Critic_name = self.path + "/" + self.Model_name + '_Actor_Critic_name.h5'
        Critic = load_model(Actor_Critic_name, compile=False)
    
    #testing the env
    states = []
    state = test_env.reset()
    if args['rnn']:
        for _ in range(args['rnn_steps'] - 1):
            state = std_scalar.transform(np.reshape(state, (1,-1)))
            states.append(state.tolist())
            state, _, _, _ = test_env.step(test_env.action_space.sample())
    done = False

    while not done:
        state = std_scalar.transform(np.reshape(state, (1,-1)))
        # #adding noise
        S_true, E_true, I_true, R_true = state[0][0], state[0][1], state[0][2], state[0][3]
        I_noisy = I_true * (1-noise)
        S_noisy = S_true + (I_true * noise / 2)
        E_noisy = E_true + (I_true * noise / 2)
        state  = np.array([[S_noisy, E_noisy, I_noisy, R_true]])

        if args['rnn']:
            states.append(state.tolist())
            state = states[-args['rnn_steps']:]
            state = np.reshape(state, (1, args['rnn_steps'], -1))
        #### Predicting the action
        if args['rnn']:
            prediction = Actor.predict(state)[0]
        else:
            prediction = Actor.predict(np.reshape(state, (1,-1)))[0]
        action = np.random.choice(3, p=prediction)
        ####
        state, _, done, _ = test_env.step(action)
    #savefig_filename = sub_dir  + 'eval.pdf'
    #test_env.plot(savefig_filename = savefig_filename)
    return test_env.state_trajectory, test_env.action_trajectory

def data(exp_dir, noise = 0):
    exp_w = os.listdir(exp_dir)
    print(exp_w)
    exp_w = list(filter(lambda x: x[:4] != 'eval', exp_w))
    print(exp_w)
    S, E, I, R, Act = [], [], [], [], []
    for exp in iter(exp_w):
        print(exp_dir + exp +'/')
        states, actions = data_per_exp(exp_dir + exp +'/', noise = noise)
        states, actions = np.array(states), np.array(actions)
        S.append(states[:,0])
        E.append(states[:,1])
        I.append(states[:,2])
        R.append(states[:,3])
        Act.append(actions)
    S_pd = pd.DataFrame(np.array(S).T, columns = exp_w)
    E_pd = pd.DataFrame(np.array(E).T, columns = exp_w)
    I_pd = pd.DataFrame(np.array(I).T, columns = exp_w)
    R_pd = pd.DataFrame(np.array(R).T, columns = exp_w)
    Act_pd = pd.DataFrame(np.array(Act).T, columns = exp_w)
    
    return S_pd, E_pd, I_pd, R_pd, Act_pd


def plot(path, to_load_sub_dir, savefig_filename=None, noise = 0, format = 'pdf'):
    name = '_eval' + '_' + str(int(np.ceil(noise*100)))
    S = pd.read_csv(path + to_load_sub_dir + '/S' + name + '.csv', index_col=0)
    E = pd.read_csv(path + to_load_sub_dir + '/E' + name + '.csv', index_col=0)
    I = pd.read_csv(path + to_load_sub_dir + '/I' + name + '.csv', index_col=0)
    R = pd.read_csv(path + to_load_sub_dir +'/R' + name + '.csv', index_col=0)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,15))
    sns.barplot(data=S, ax=axes[0, 0])
    sns.barplot(data=E, ax=axes[0, 1])
    sns.barplot(data=I, ax=axes[1, 0])
    sns.barplot(data=R, ax=axes[1, 1])
    axes[0, 0].set_ylabel('Susceptible', fontsize=15)
    axes[0, 1].set_ylabel('Exposed', fontsize=15)
    axes[1, 0].set_ylabel('Infectious', fontsize=15)
    axes[1, 1].set_ylabel('Removed', fontsize=15)
    fig.suptitle('Agerage people over 140 days, with ' + str(int(np.ceil(noise*100))) + '% unreported Infectious people' ) # or plt.suptitle('Main title')

    if savefig_filename is not None:
        assert isinstance(savefig_filename + '.' + format, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename + '.' + format, format = format)
    else:
        plt.show()
    plt.close(fig)


def plot_act(path, to_load_sub_dir, savefig_filename=None, noise = 0, format = 'pdf'):
    exp_w = os.listdir(path)
    exp_w = list(filter(lambda x: x[:4] != 'eval', exp_w))
    name = '_eval' + '_' + str(int(np.ceil(noise*100)))
    A = pd.read_csv(path + to_load_sub_dir + '/act' + name + '.csv', index_col=0)
    A = A.astype('category')
    # melting
    melt_cols = exp_w.copy()
    melt_cols.insert(0, 'time')
    A_1 = A.reset_index()
    A_1.columns = melt_cols
    Act_melt = pd.melt(A_1, id_vars=['time'], var_name='weight',value_name= 'action')
    dict_act = {0: 'Lock-down', 1: 'Social distancing', 2:'Open-economy'}
    Act_melt['action'] = Act_melt.action.map(dict_act)
    Act_melt['time'] = Act_melt['time'] * (5 / (60 * 24 * 7))
    #sns.set()
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))
    pal = {'Lock-down':"Red", 'Open-economy':"Green",'Social distancing':'Yellow'}
    sns.stripplot(x='time', y='weight',hue='action', data=Act_melt, size =3, jitter =True, palette = pal)
    ax.set_title('actions vs time (weeks) with ' + str(int(np.ceil(noise*100))) + '% unreported Infectious people')
    # sns.swarmplot(x='weight', y='time',hue='action', data=Act_melt, size =3)
    if savefig_filename is not None:
        assert isinstance(savefig_filename + '.' + format, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename + '.' + format, format = format)
    else:
        plt.show()
    plt.close(fig)
    return A_1, Act_melt 

def plot_infected(path, to_load_sub_dir, savefig_filename=None, noise = 0, format = 'pdf'):
    name = '_eval' + '_' + str(int(np.ceil(noise*100)))
    S = pd.read_csv(path + to_load_sub_dir + '/S' + name + '.csv', index_col=0)
    E = pd.read_csv(path + to_load_sub_dir + '/E' + name + '.csv', index_col=0)
    I = pd.read_csv(path + to_load_sub_dir + '/I' + name + '.csv', index_col=0)
    R = pd.read_csv(path + to_load_sub_dir +'/R' + name + '.csv', index_col=0)
    Infected = E + I + R
    Infected.reset_index(inplace=True)
    Infected['index'] = Infected['index'] * (5/(60 * 24 * 7))
    # Infected_melt = Infected.melt('index', var_name='Weights',  value_name='Infected')
    # print(Infected_melt.head())

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (15,15))
    # g = sns.catplot(x="index", y="Infected", hue='Weights', data=Infected_melt)
    Infected.plot.line(x = 'index', ax = axes, linewidth=7.0)
    print("done")
    axes.set_ylabel('Infected', fontsize=15)
    fig.suptitle('Infected people over 140 days, with ' + str(int(np.ceil(noise*100))) + '% unreported Infectious people' ) # or plt.suptitle('Main title')

    if savefig_filename is not None:
        assert isinstance(savefig_filename + '.' + format, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename + '.' + format, format = format)
    else:
        plt.show()
    plt.close(fig)

def plot_infected_weight(path, to_load_sub_dir, savefig_filename=None, weight = 0.5, format = 'pdf'):
    Overall_infected = []
    NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1.0]
    for noise in iter(NOISE):
        name = '_eval' + '_' + str(int(np.ceil(noise*100)))
        S = pd.read_csv(path + to_load_sub_dir + '/S' + name + '.csv', index_col=0)
        E = pd.read_csv(path + to_load_sub_dir + '/E' + name + '.csv', index_col=0)
        I = pd.read_csv(path + to_load_sub_dir + '/I' + name + '.csv', index_col=0)
        R = pd.read_csv(path + to_load_sub_dir +'/R' + name + '.csv', index_col=0)
        Infected = E + I + R
        Overall_infected.append(Infected[str(weight)].values.tolist())
    df = pd.DataFrame(np.array(Overall_infected).T, columns=NOISE)
    # print(np.shape(Overall_infected))
    # print(df.head())
    df.reset_index(inplace=True)
    df['index'] = df['index'] * (5/(60 * 24 * 7))
    # # Infected_melt = Infected.melt('index', var_name='Weights',  value_name='Infected')
    # # print(Infected_melt.head())

    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (15,15))
    # # # g = sns.catplot(x="index", y="Infected", hue='Weights', data=Infected_melt)
    # df.plot.line(x = 'index', ax = axes, linewidth=7.0)
    # # print("done")
    # axes.set_ylabel('Infected', fontsize=15)
    # fig.suptitle('Infected people over 140 days, for w=' + str(int(np.ceil(weight))) ) # or plt.suptitle('Main title')
    ax = df.plot(x = 'index', lw=2, colormap='jet', marker='.', markersize=10, title='Infected people over 140 days, for w=' + str(weight))
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")

    if savefig_filename is not None:
        assert isinstance(savefig_filename + '.' + format, str), "filename for saving the figure must be a string"
        ax.get_figure().savefig(savefig_filename + '.' + format, format = format)
    else:
        ax.show()
    # plt.close(fig)

dir = './results/exp-7-7days/'
# dir = './results/exp-6-7days/'
# dir = './results/experiment-5-rnn/'
# dir = './results/experiment-4-rnn/'

to_save_sub_dir = 'eval_yang_no_change_theta_new'

try:
    os.mkdir(dir + to_save_sub_dir)
except:
    pass

# NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1.0]
# for noise in iter(NOISE):
#     print("RUNNING FOR NOISE: {}".format(noise))

#     name = '_eval' + '_' + str(int(np.ceil(noise*100)))
#     #-------
#     # #uncomment this for the first time
#     # S, E, I, R, A = data(dir, noise = noise)  
#     # S.to_csv(dir + to_save_sub_dir + '/S' + name + '.csv')
#     # E.to_csv(dir + to_save_sub_dir + '/E' + name+ '.csv')
#     # I.to_csv(dir + to_save_sub_dir + '/I' + name+ '.csv')
#     # R.to_csv(dir + to_save_sub_dir + '/R' + name+ '.csv')
#     # A.to_csv(dir + to_save_sub_dir + '/act' + name+ '.csv')
#     # #--------

#     # plot(dir, to_save_sub_dir,  dir + to_save_sub_dir + '/states' + name , noise = noise, format = 'jpg')
#     # plot_act(dir, to_save_sub_dir,  dir + to_save_sub_dir + '/actions'+ name , noise = noise, format = 'jpg')
    
#     # plot_infected(dir, to_save_sub_dir, dir + to_save_sub_dir + '/Infected'+ name, noise = noise, format = 'jpg')


# WEIGHTS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
# for w in iter(WEIGHTS):
#     print("RUNNING FOR WEIGHTS: {}".format(w))

#     name = '_eval' + '_' + str(w)
#     #-------
#     # #uncomment this for the first time
#     # S, E, I, R, A = data(dir, noise = noise)  
#     # S.to_csv(dir + to_save_sub_dir + '/S' + name + '.csv')
#     # E.to_csv(dir + to_save_sub_dir + '/E' + name+ '.csv')
#     # I.to_csv(dir + to_save_sub_dir + '/I' + name+ '.csv')
#     # R.to_csv(dir + to_save_sub_dir + '/R' + name+ '.csv')
#     # A.to_csv(dir + to_save_sub_dir + '/act' + name+ '.csv')
#     # #--------

#     # plot(dir, to_save_sub_dir,  dir + to_save_sub_dir + '/states' + name , noise = noise, format = 'jpg')
#     # plot_act(dir, to_save_sub_dir,  dir + to_save_sub_dir + '/actions'+ name , noise = noise, format = 'jpg')
    
#     # plot_infected(dir, to_save_sub_dir, dir + to_save_sub_dir + '/Infected'+ name, noise = noise, format = 'jpg')
#     plot_infected_weight(dir, to_save_sub_dir, savefig_filename =  dir + to_save_sub_dir + '/Infected_weight'+ name, weight = w, format = 'jpg')


def plot_IE(dir, to_load_sub_dir, noise = 0, weight = None, format = 'jpg' ):
    """
    given a noise value it will generate the figure for every weight 
    """
    # NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1.0]
    name = '_eval' + '_' + str(int(np.ceil(noise*100)))
    path = dir + to_load_sub_dir 
    print("RUNNING FOR NOISE: {}".format(noise))
    name = '_eval' + '_' + str(int(np.ceil(noise*100)))
    #-------
    # loading the date of every weight for a given noise
    # 
    S = pd.read_csv(path + '/S' + name + '.csv', index_col=0)
    E = pd.read_csv(path + '/E' + name + '.csv', index_col=0)
    I = pd.read_csv(path + '/I' + name + '.csv', index_col=0)
    R = pd.read_csv(path +'/R' + name + '.csv', index_col=0) 
    Act = pd.read_csv(path +'/act' + name + '.csv', index_col=0) 
    EI = I+E
    labels = ['Exposed', 'Infecetious', 'Exposed and Infecetious']
    for i, df in enumerate([E, I, EI]):
        df.reset_index(inplace = True)
        df['index'] = df['index'] * (5 / (60 * 24 * 7))
        ax = df.plot(x = 'index', lw=2, colormap='jet', markersize=5, title= labels[i] + ' people over 20 weeks, for Noise =' + str(noise*100) + '%')
        ax.set_xlabel("Time (weeks)")
        ax.set_ylabel('# of '+ labels[i] + ' people')
        ax.set_ylim([0, 25000])
        savefig_filename = path + '/' + labels[i] + '.' + format
        if savefig_filename is not None:
            assert isinstance(savefig_filename + '.' + format, str), "filename for saving the figure must be a string"
            ax.get_figure().savefig(savefig_filename + '.' + format, format = format)
        else:
            ax.show()
    #
    #--------

# plot_IE(dir='./results/exp-7-7days/', to_load_sub_dir='eval_yang_no_change_theta_new')

def plot_rewards_noise(dir, to_load_sub_dir, noise = 0, weight = None, format = 'jpg' ):
    """
    given a noise value it will generate the figure of rewards for every weight 
    """
    # NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1.0]
    name = '_eval' + '_' + str(int(np.ceil(noise*100)))
    path = dir + to_load_sub_dir 
    print("RUNNING FOR NOISE: {}".format(noise))
    name = '_eval' + '_' + str(int(np.ceil(noise*100)))
    eco_costs    = np.array([1, 0.25, 0], dtype=float)
    eco_costs_dict ={0:eco_costs[0], 1:eco_costs[1], 2: eco_costs[2]}
    Ts = 7 
    #-------
    # loading the date of every weight for a given noise
    # 
    S = pd.read_csv(path + '/S' + name + '.csv', index_col=0)
    E = pd.read_csv(path + '/E' + name + '.csv', index_col=0)
    I = pd.read_csv(path + '/I' + name + '.csv', index_col=0)
    R = pd.read_csv(path +'/R' + name + '.csv', index_col=0) 
    Act = pd.read_csv(path +'/act' + name + '.csv', index_col=0) 

        # Public health Cost increases with increase in Infected people.
    publichealthCost   =  (1e-4 * I) * Ts
        # Rewards
    weights = S.columns
    rewards = []
    for w in iter(weights):
        economicCost = Act[w].map(eco_costs_dict) * Ts
        temp = - float(w) * economicCost - (1 - float(w)) * publichealthCost[w]
        rewards.append(temp)
    Rewards = pd.DataFrame(np.array(rewards).T, columns= weights)
    Rewards.reset_index(inplace = True)
    Rewards['index'] = Rewards['index'] * (5 / (60 * 24 * 7))
    ax = Rewards.plot(x = 'index', lw=2, colormap='jet', markersize=5, title= 'Rewards over 20 weeks, for Noise =' + str(noise*100) + '%')
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel('Rewards')
    ax.set_ylim([-10, 0])
    savefig_filename = path + '/' + 'Rewards' +  name
    if savefig_filename is not None:
        assert isinstance(savefig_filename + '.' + format, str), "filename for saving the figure must be a string"
        ax.get_figure().savefig(savefig_filename + '.' + format, format = format)
    else:
        ax.show()
    #     
    #
#     #--------
# NOISE = np.arange(0,11) / 10.0
# for n in iter(NOISE):
#     plot_rewards_noise(dir='./results/exp-7-7days/', to_load_sub_dir='eval_yang_no_change_theta_new', noise = n)



def plot_rewards_weight(dir, to_load_sub_dir, weight = None, format = 'jpg' ):
    """
    given a noise value it will generate the figure of rewards for every weight 
    """
    
    eco_costs    = np.array([1, 0.25, 0], dtype=float)
    eco_costs_dict ={0:eco_costs[0], 1:eco_costs[1], 2: eco_costs[2]}
    Ts = 7
    NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1.0]
    rewards = []
    path = dir + to_load_sub_dir 
    for noise in iter(NOISE):
        # print("RUNNING FOR NOISE: {}".format(noise))
        name = '_eval' + '_' + str(int(np.ceil(noise*100)))
        # S = pd.read_csv(path + '/S' + name + '.csv', index_col=0)
        # E = pd.read_csv(path + '/E' + name + '.csv', index_col=0)
        I = pd.read_csv(path + '/I' + name + '.csv', index_col=0)
        # R = pd.read_csv(path +'/R' + name + '.csv', index_col=0) 
        Act = pd.read_csv(path +'/act' + name + '.csv', index_col=0) 

        publichealthCost   =  (1e-4 * I[weight]) * Ts
        economicCost = Act[weight].map(eco_costs_dict) * Ts
        temp = - float(weight) * economicCost - (1 - float(weight)) * publichealthCost
        rewards.append(temp)

    NOISE_str = [str(n*100)+'%' for n in NOISE]
    Rewards = pd.DataFrame(np.array(rewards).T, columns= NOISE_str)
    Rewards.reset_index(inplace = True)
    Rewards['index'] = Rewards['index'] * (5 / (60 * 24 * 7))
    ax = Rewards.plot(x = 'index', lw=2, colormap='jet', markersize=5, title= 'Rewards over 20 weeks, for weight =' + weight)
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel('Rewards')
    ax.set_ylim([-10, 0.1])
    savefig_filename = path + '/' + 'Rewards' +  '_eval' + '_weight=' + weight
    print(savefig_filename)
    if savefig_filename is not None:
        assert isinstance(savefig_filename + '.' + format, str), "filename for saving the figure must be a string"
        ax.get_figure().savefig(savefig_filename + '.' + format, format = format)
    else:
        ax.show()

# path = './results/exp-7-7days/' + 'eval_yang_no_change_theta_new'
# name = '_eval' + '_' + str(int(np.ceil(0*100)))
# S = pd.read_csv(path + '/S' + name + '.csv', index_col=0)
# weights = S.columns
# for w in iter(weights):
#     plot_rewards_weight(dir='./results/exp-7-7days/', to_load_sub_dir='eval_yang_no_change_theta_new', weight = w)



def plot_eco_soc_rewards(dir, to_load_sub_dir, noise = 0, weight = None, format = 'jpg' ):
    """
    given a noise value it will generate the figure of rewards for every weight 
    """
    # NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1.0]
    name = '_eval' + '_' + str(int(np.ceil(noise*100)))
    path = dir + to_load_sub_dir 
    print("RUNNING FOR NOISE: {}".format(noise))
    name = '_eval' + '_' + str(int(np.ceil(noise*100)))
    eco_costs    = np.array([1, 0.25, 0], dtype=float)
    eco_costs_dict ={0:eco_costs[0], 1:eco_costs[1], 2: eco_costs[2]}
    Ts = 7 
    #-------
    # loading the date of every weight for a given noise
    # 
    S = pd.read_csv(path + '/S' + name + '.csv', index_col=0)
    E = pd.read_csv(path + '/E' + name + '.csv', index_col=0)
    I = pd.read_csv(path + '/I' + name + '.csv', index_col=0)
    R = pd.read_csv(path +'/R' + name + '.csv', index_col=0) 
    Act = pd.read_csv(path +'/act' + name + '.csv', index_col=0) 

        # Public health Cost increases with increase in Infected people.
    publichealthCost   =  (1e-4 * I) * Ts
        # Rewards
    weights = S.columns
    eco_cost = []

    for w in iter(weights):
        economicCost = Act[w].map(eco_costs_dict) * Ts
        #temp = - float(w) * economicCost - (1 - float(w)) * publichealthCost[w]
        eco_cost.append(economicCost.values)
    eco_cost = pd.DataFrame(np.array(eco_cost).T, columns= weights)

    total_cost = []
    for w in iter(weights):
        temp = float(w) * publichealthCost[w] + (1-float(w))*eco_cost[w]
        total_cost.append(temp)
    Total_Cost = pd.DataFrame(np.array(total_cost).T, columns=weights)
    # print(publichealthCost.head())
    eco_cost.reset_index(inplace = True)
    publichealthCost.reset_index(inplace = True)
    Total_Cost.reset_index(inplace = True)
    publichealthCost['index'] = publichealthCost['index'] * (5 / (60 * 24 * 7))
    eco_cost['index'] = eco_cost['index'] * (5 / (60 * 24 * 7))
    Total_Cost['index'] = Total_Cost['index'] * (5 / (60 * 24 * 7))
    # print(publichealthCost.head()) 
    # print(eco_cost.head())
    # fig = plt.figure()

    # for frame in [eco_cost, publichealthCost]:
    #     plt.plot(frame['index'], frame[weights])

    # # plt.xlim(0,18000)
    # # plt.ylim(0,30)
    # plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (16,8))
    eco_cost.plot(x = 'index', ax=ax[0])
    publichealthCost.plot(x = 'index', ax=ax[1])
    Total_Cost.plot(x = 'index', ax=ax[2])
    ax[0].set_title('Economic Costs')
    ax[1].set_title('Public Health Costs')
    ax[2].set_title('Total Costs')

    ax[0].set_xlabel('Time (weeks')
    ax[1].set_xlabel('Time (weeks')
    ax[2].set_xlabel('Time (weeks')

    ax[0].set_ylim([-1, 15])
    ax[1].set_ylim([-1, 15])
    ax[2].set_ylim([-1, 15])


    savefig_filename = path + '/' + 'Costs_total' +  name 
    if savefig_filename is not None:
        assert isinstance(savefig_filename + '.' + format, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename + '.' + format, format = format)
    else:
        plt.show()
    #     
    #
#     #--------
# NOISE = np.arange(0,11) / 10.0
# for n in iter(NOISE):
plot_eco_soc_rewards(dir='./results/exp-7-7days/', to_load_sub_dir='eval_yang_no_change_theta_new', noise = 0.0)

