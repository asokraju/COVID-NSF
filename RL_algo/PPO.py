import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import gym
from gym import spaces
from gym.utils import seeding

import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K

import threading
from threading import Thread, Lock
import time

from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from scipy.io import savemat

def AC_model(input_shape, action_dim, lr):
    """
    defines Actor and Critic models that shares the input and the hidden layers
    """
    input_layer = Input(input_shape)
    layer_1 = Dense(units=128, activation='elu', kernel_initializer='he_uniform')(input_layer)
    actions = Dense(units=action_dim, activation='softmax', kernel_initializer='he_uniform')(layer_1)
    values = Dense(units=1, activation='linear', kernel_initializer='he_uniform')(layer_1)

    def ppo_loss(y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, predictions_old, actions_onehot = y_true[:, :1], y_true[:, 1:1+action_dim], y_true[:, 1+action_dim:]
        EPSILON = 0.1
        C = 5e-3
        prob = y_pred * actions_onehot
        old_prob = predictions_old * actions_onehot
        r = prob / (old_prob +1e-10)
        
        p1 = r * advantages
        p2 = K.clip(r, min_value=1-EPSILON, max_value=1+EPSILON) * advantages
        entropy_loss = -prob * K.log(prob + 1e-10)

        loss = -K.mean(K.minimum(p1, p2) + C * entropy_loss)
        return loss

    Actor = Model(inputs = input_layer, outputs = actions)
    Actor.compile(loss = ppo_loss, optimizer=RMSprop(lr=lr))
    #print(Actor.summary())
    Critic = Model(inputs = input_layer, outputs = values)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))
    #print(Critic.summary())
    return Actor, Critic




################################################################################################################
class PPOAgent:
    def __init__(
        self,  env, test_env, 
        exp_name = 'seir', 
        EPSIODES = 10000, 
        lr = 0.0001, 
        EPOCHS = 10,
        path = None,
        gamma = 0.95
        ):
        self.exp_name = exp_name
        self.env = env # SEIR_v0_2(discretizing_time = 5, sampling_time = 1, sim_length = 100)
        self.test_env = test_env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 3# specific to SEIR model self.env.action_space.n
        self.lr = lr
        self.gamma = gamma
        self.path = path

        self.EPSIODES, self.episode, self.max_avg_reward = EPSIODES, 0, -100 #max_avg_reward depends on the environment
        self.scores, self.averages, self.episodes = [], [], []
        self.Actor, self.Critic = AC_model(input_shape=self.state_dim, action_dim=self.action_dim, lr=self.lr)

        self.Model_name  = 'PPO_' + exp_name +'_'
        self.Actor_name  = self.path + "/" + self.Model_name + '_Actor.h5'
        self.Critic_name = self.path + "/" + self.Model_name + '_Critic.h5'

        self.EPOCHS = EPOCHS
        self.std_scalar = StandardScaler()
        self._standardizing_state()

    def _standardizing_state(self):
        path = self.path + "/" + self.Model_name + 'std_scaler.bin'
        try:
            self.std_scalar=load(path)
        except:
            print("fitting a StandardScaler() to scale the states")
            states = []
            for _ in range(100):  
                self.env.reset()  
                d = False
                while not d:
                    s, r, d, _ =  self.env.step(self.env.action_space.sample())
                    states.append(list(s))
            X = np.array(states)
            self.std_scalar = StandardScaler()
            self.std_scalar.fit(X)
            print("done!")
            dump(self.std_scalar, path, compress=True)

    def predict_actions(self, state):
        """
        predicts the actions propabilities from actor model. 
        And samples an action using the actions propabilities
        """
        np.reshape(state, (1,-1))
        prediction = self.Actor.predict(np.reshape(state, (1,-1)))[0]
        action = np.random.choice(self.action_dim, p=prediction)
        return prediction, action
    
    def save(self):
        self.Actor.save(self.Actor_name)
        self.Critic.save(self.Critic_name)
    
    def discount_rewards(self, rewards):
        reward_sum = 0
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards -= np.mean(discounted_rewards) # normalizing the result
        discounted_rewards /= np.std(discounted_rewards) # divide by standard deviation
        return discounted_rewards

    def replay(self, states, actions, rewards, predictions):
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)
        discounted_r = np.vstack(self.discount_rewards(rewards))
        values = self.Critic.predict(states)

        advantages = discounted_r - values
        
        y_true = np.hstack([advantages, predictions, actions])

        self.Actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))
        self.Critic.fit(states, discounted_r, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))

    def run(self):
        for e in range(self.EPSIODES):
            state = self.env.reset()
            done, score, save_model = False, 0, ""
            #memory buffers of states, actions, rewards, predicted action propabilities
            states, actions, rewards, predictions = [], [], [], []
            while not done:
                state = self.std_scalar.transform(np.reshape(state, (1,-1)))
                prediction, action  = self.predict_actions(state)
                next_state, reward, done, _ = self.env.step(action)
                actions_onehot = tf.keras.utils.to_categorical(action, self.action_dim)
                states.append(state)
                actions.append(actions_onehot)
                predictions.append(prediction)
                rewards.append(reward)
                state = next_state
                score += reward 
                if done:
                    self.scores.append(score)
                    average = sum(self.scores[-50:]) / len(self.scores[-50:])
                    self.averages.append(average)
                    self.episodes.append(self.episode)
                    if average >= self.max_avg_reward :
                        self.max_avg_reward = average
                        self.save()
                        save_model = 'SAVING'
                    else:
                        save_model = ""
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPSIODES, score, average, save_model))
                    self.replay(states, actions, rewards, predictions)
    
    def load(self):
        if os.path.isfile(self.Actor_name):
            print('loading Actor weights')
            #Actor_name = self.path + "/" + self.Model_name + '_Actor.h5'
            self.Actor = load_model(self.Actor_name, compile=False)
        if os.path.isfile(self.Critic_name):
            print('loading Critic weights')
            #Actor_name = self.path + "/" + self.Model_name + '_Actor.h5'
            self.Critic = load_model(self.Critic_name, compile=False)

    def test(self, savefig_filename = None):
        self.load()
        test_env = self.test_env #SEIR_v0_2(discretizing_time = 5, sampling_time = 1, sim_length = 100)
        state = test_env.reset()
        done, score, save_model = False, 0, ""
        # memory buffers of states, actions, rewards, predicted action propabilities
        states, actions, rewards, predictions = [], [], [], []
        while not done:
            state = self.std_scalar.transform(np.reshape(state, (1,-1)))
            _, action  = self.predict_actions(state)
            state, _, done, _ = test_env.step(action)
        savefig_filename = self.path + "/" + savefig_filename
        test_env.plot(savefig_filename = savefig_filename)
