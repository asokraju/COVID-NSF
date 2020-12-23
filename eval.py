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


def data_per_exp(sub_dir):

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
        print('loading Actor weights')
        #Actor_name = self.path + "/" + self.Model_name + '_Actor.h5'
        Actor = load_model(Actor_name, compile=False)
    if os.path.isfile(Critic_name):
        print('loading Critic weights')
        #Actor_name = self.path + "/" + self.Model_name + '_Actor.h5'
        Critic = load_model(Critic_name, compile=False)
    if os.path.isfile(Actor_Critic_name):
        print('loading Actor_Critic weights')
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
    savefig_filename = sub_dir  + 'eval.pdf'
    test_env.plot(savefig_filename = savefig_filename)
    return test_env.state_trajectory, test_env.action_trajectory

def data(exp_dir):
    exp_w = os.listdir(exp_dir)
    print(exp_w)
    exp_w = list(filter(lambda x: x != 'eval', exp_w))
    print(exp_w)
    S, E, I, R, Act = [], [], [], [], []
    for exp in iter(exp_w):
        print(exp_dir + exp +'/')
        states, actions = data_per_exp(exp_dir + exp +'/')
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


def plot(path, savefig_filename=None):
    S = pd.read_csv(path + 'eval/S_eval.csv', index_col=0)
    E = pd.read_csv(path + 'eval/E_eval.csv', index_col=0)
    I = pd.read_csv(path + 'eval/I_eval.csv', index_col=0)
    R = pd.read_csv(path + 'eval/R_eval.csv', index_col=0)
    _, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,15))
    sns.barplot(data=S, ax=axes[0, 0])
    sns.barplot(data=E, ax=axes[0, 1])
    sns.barplot(data=I, ax=axes[1, 0])
    sns.barplot(data=R, ax=axes[1, 1])
    axes[0, 0].set_ylabel('Susceptible', fontsize=15)
    axes[0, 1].set_ylabel('Exposed', fontsize=15)
    axes[1, 0].set_ylabel('Infectious', fontsize=15)
    axes[1, 1].set_ylabel('Removed', fontsize=15)
    if savefig_filename is not None:
        assert isinstance(savefig_filename, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = 'pdf')
    else:
        plt.show()


def plot_act(path, savefig_filename=None):
    exp_w = os.listdir(path)
    exp_w = list(filter(lambda x: x != 'eval', exp_w))
    A = pd.read_csv(path + 'eval/act_eval.csv', index_col=0)
    A = A.astype('category')
    print(A.head())
    print(A.info())
    print(A.columns)
    percent_a = []
    for key in A.columns:
        temp = A[key].value_counts(normalize=False,ascending=False,dropna=False)/A.shape[0]
        percent_a.append(temp.values)
    Act_pd = pd.DataFrame(np.array(percent_a).T, columns = exp_w)
    
    # print(Act_pd)
    # sns.barplot(
    #     data= Act_pd,
    #     hue = '0.3'
    # )
    # plt.show()

    data = Act_pd.values
    X = np.arange(5)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
    ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
    plt.show()
    return Act_pd

dir = './results/exp-6-7days/'
#dir = './results/exp-7-7days/'
# dir = './results/experiment-5-rnn/'
#dir = './results/experiment-4-rnn/'

try:
    os.mkdir(dir + 'eval')
except:
    pass

# S, E, I, R, A = data(dir)
# S.to_csv(dir + 'eval/S_eval.csv')
# E.to_csv(dir + 'eval/E_eval.csv')
# I.to_csv(dir + 'eval/I_eval.csv')
# R.to_csv(dir + 'eval/R_eval.csv')
# A.to_csv(dir + 'eval/act_eval.csv')

plot_act(dir, dir + 'eval/eval.pdf')
