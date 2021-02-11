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
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Flatten, GRU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K

import concurrent.futures 
import time
import copy
from functools import partial

from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from scipy.io import savemat

def AC_model(input_shape, action_dim, lr):
    """
    defines Actor and Critic models that shares the input and the hidden layers
    Updates Actor and Critic individually
    """
    input_layer = Input(input_shape)
    layer_1 = Dense(units=128, activation='elu', kernel_initializer='he_uniform')(input_layer)

    layer_a = Dense(units=32, activation='elu', kernel_initializer='he_uniform')(input_layer)
    actions = Dense(units=action_dim, activation='softmax', kernel_initializer='he_uniform')(layer_a)

    layer_v = Dense(units=32, activation='elu', kernel_initializer='he_uniform')(layer_1)
    values = Dense(units=1, activation='linear', kernel_initializer='he_uniform')(layer_v)

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
    Actor.compile(loss = ppo_loss, optimizer=Adam(lr=lr))
    #print(Actor.summary())
    Critic = Model(inputs = input_layer, outputs = values)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))
    #print(Critic.summary())
    return Actor, Critic

def AC_model_new(input_shape, action_dim, lr, EPSILON, C, rnn, rnn_steps):
    """
    defines Actor and Critic models that shares the input and the hidden layers
    updated Actor and Critic together
    """
    if rnn:
        input_layer = Input((rnn_steps, input_shape))
        layer_1 = GRU(units=32, kernel_initializer='he_uniform')(input_layer)
    else:
        input_layer = Input(input_shape)
        layer_1 = Dense(units=128, activation='elu', kernel_initializer='he_uniform')(input_layer)

    layer_a = Dense(units=32, activation='elu', kernel_initializer='he_uniform')(layer_1) #input_layer
    actions = Dense(units=action_dim, activation='softmax', kernel_initializer='he_uniform', name='actions')(layer_a)

    layer_v = Dense(units=32, activation='elu', kernel_initializer='he_uniform')(layer_1)
    values = Dense(units=1, activation='linear', kernel_initializer='he_uniform', name='values')(layer_v)

    def ppo_loss(y_true, y_pred, EPSILON, C):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, predictions_old, actions_onehot = y_true[:, :1], y_true[:, 1:1+action_dim], y_true[:, 1+action_dim:]
        # EPSILON = 0.1
        # C = 5e-2
        prob = y_pred * actions_onehot
        old_prob = predictions_old * actions_onehot
        r = prob / (old_prob +1e-10)
        
        p1 = r * advantages
        p2 = K.clip(r, min_value=1-EPSILON, max_value=1+EPSILON) * advantages
        entropy_loss = -prob * K.log(prob + 1e-10)

        loss = -K.mean(K.minimum(p1, p2) + C * entropy_loss)
        return loss
    
    def ppo_parametrized(EPSILON, C):
        def ppo_loss_fun(y_true, y_pred):
            return ppo_loss(y_true, y_pred, EPSILON, C)
        return ppo_loss_fun

    
    lossWeights  = {'actions':0.5, 'values':0.5}
    lossFuns     = {'actions':ppo_parametrized(EPSILON, C), 'values':'mse'}

    Actor = Model(inputs = input_layer, outputs = actions)
    Critic = Model(inputs = input_layer, outputs = values)
    Actor_Critic = Model(inputs = input_layer, outputs = [actions, values])
    Actor_Critic.compile(optimizer=RMSprop(lr=lr), loss=lossFuns, loss_weights=lossWeights)
    # Actor_Critic.compile(optimizer=RMSprop(lr=lr), loss=[ppo_parametrized(EPSILON, C), 'mse'], loss_weights=[0.5, 0.5])
    return Actor, Critic, Actor_Critic


def sampler(env, rnn_steps, episode_num, gamma, std_scalar_path, actor_path, critic_path):
        states, actions, rewards, predictions = [], [], [], []
        # if os.path.isfile(actor_path):
        # Loading the models for actor, critic and standard scalar
        try:
            std_scalar = load(std_scalar_path)
        except:
            assert os.path.isfile(std_scalar_path), "Standard Scalar path is empty"

        try:
            Actor = load_model(actor_path, compile=False)
        except:
            assert os.path.isfile(actor_path), "Actor path is empty"

        try:
            Critic = load_model(critic_path, compile=False)
        except:
            assert  os.path.isfile(critic_path), "Critic path is empty"

        state = env.reset()
        for _ in range(rnn_steps - 1):
            state = std_scalar.transform(np.reshape(state, (1,-1)))
            states.append(state.tolist())
            state, _, _, _ = env.step(env.action_space.sample())

        done, score = False, 0
        while not done:
            state = std_scalar.transform(np.reshape(state, (1,-1)))
            states.append(state.tolist())
            state = np.reshape(states[-rnn_steps:], (1, rnn_steps, -1))
            # print(np.reshape(state, (1,-1)), np.shape(np.reshape(state, (1,-1))))
            #prediction, action  = self.predict_actions(states[-self.rnn_steps:])
            if episode_num<=1:
                action = np.random.choice(3)
                prediction = np.full((3,), 1/3)
            else:
                # prediction = Actor.predict(np.reshape(state, (1,-1)))[0]
                prediction = Actor.predict(state)[0]
                action = np.random.choice(env.action_space.n, p=prediction)
            next_state, reward, done, _ = env.step(action)
            actions_onehot = tf.keras.utils.to_categorical(action, env.action_space.n)
            actions.append(actions_onehot)
            predictions.append(prediction)
            rewards.append(reward)
            state = next_state
            score += reward
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        #computing the discounted rewards
        reward_sum = 0
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards -= np.mean(discounted_rewards) # normalizing the result
        discounted_rewards /= (np.std(discounted_rewards) + 1e-10) # divide by standard deviation, added 1e-10 for numerical stability
        discounted_r = np.vstack(discounted_rewards)


        states_rnn = [states[i:i+rnn_steps].tolist() for i in range(len(states) - rnn_steps+1)]
        states_rnn = np.reshape(states_rnn, (-1, rnn_steps, env.observation_space.shape[0]))
        values = Critic.predict(states_rnn)
        # print("values", values)
        # print("discounted_r", np.shape(discounted_r))
        advantages = discounted_r - values
        
        y_true = np.hstack([advantages, predictions, actions])

        #Y = [y_true, discounted_r]
        return states_rnn, y_true, discounted_r , score     


def _std_scalar_work(env):
    """
    this generates the samples to train std scalar function
    this is used when mpi=true
    """
    states = []
    env.reset()  
    d = False
    while not d:
        s, _, d, _ =  env.step(env.action_space.sample())
        states.append(list(s))
    return states

################################################################################################################
class PPOAgent:
    def __init__(
        self,  env, test_env, env_list,
        exp_name = 'seir', 
        EPSIODES = 10000, 
        lr = 0.0001, 
        EPOCHS = 10,
        path = None,
        gamma = 0.95,
        traj_per_episode = 4,
        EPSILON = 0.1,
        C = 1e-2,
        rnn = False,
        rnn_steps = 2, 
        use_mpi = False
        ):
        self.exp_name = exp_name
        self.env = env # SEIR_v0_2(discretizing_time = 5, sampling_time = 1, sim_length = 100)
        self.test_env = test_env
        self.env_list = env_list
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 3# specific to SEIR model self.env.action_space.n
        self.lr = lr
        self.EPSILON = EPSILON
        self.C = C

        self.rnn = rnn
        self.rnn_steps = rnn_steps
        self.use_mpi = use_mpi
        self.gamma = gamma
        self.path = path
        self.traj_per_episode =traj_per_episode
        self.EPSIODES, self.episode, self.max_avg_reward = EPSIODES, 0, -200 #max_avg_reward depends on the environment
        self.scores, self.averages, self.episodes = [], [], []
        # self.Actor, self.Critic = AC_model(input_shape=self.state_dim, action_dim=self.action_dim, lr=self.lr)
        self.Actor, self.Critic, self.Actor_Critic = AC_model_new(
            input_shape    =    self.state_dim, 
            action_dim     =    self.action_dim, 
            lr             =    self.lr, 
            EPSILON        =    self.EPSILON, 
            C              =    self.C,
            rnn            =    self.rnn,
            rnn_steps      =    self.rnn_steps
            )

        self.Model_name  = 'PPO_' + exp_name +'_'
        self.Actor_name  = self.path + "/" + self.Model_name + '_Actor.h5'
        self.Critic_name = self.path + "/" + self.Model_name + '_Critic.h5'
        self.Actor_Critic_name = self.path + "/" + self.Model_name + '_Actor_Critic.h5' 
        self.std_path = self.path + "/" + self.Model_name + 'std_scaler.bin'
        self.EPOCHS = EPOCHS
        self.std_scalar = StandardScaler()
        self._standardizing_state()

        # try:
        #     self.load()
        #     print("Successfully loaded the weight of the Actor-Critic neural network")
        # except:
        #     pass

    def _standardizing_state(self):
        n_envs = 100
        # self.std_path = self.path + "/" + self.Model_name + 'std_scaler.bin'
        if os.path.isfile(self.std_path):
            self.std_scalar=load(self.std_path)
        else:
            print("fitting a StandardScaler() to scale the states")
            states = []
            if self.use_mpi:
                """
                if multi processing is enabled, we will use it to obtain the samples

                """
                print("Using concurent futures to obtain the samples")
                list_envs = [copy.deepcopy(self.env) for _ in range(n_envs)]
                with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:        
                    future_to_samples = {executor.submit(_std_scalar_work, env): env for env in list_envs}
                    kwargs = {
                        'total': len(future_to_samples),
                        'unit': 'it',
                        'unit_scale': True,
                        'leave': True
                    }
                for future in concurrent.futures.as_completed(future_to_samples):
                    states_new = future.result()
                    states.append(states_new)
                states = np.vstack(states)
            else:
                for _ in range(n_envs):  
                    self.env.reset()  
                    d = False
                    while not d:
                        s, _, d, _ =  self.env.step(self.env.action_space.sample())
                        states.append(list(s))
            X = np.array(states)
            self.std_scalar = StandardScaler()
            self.std_scalar.fit(X)
            print("done!")
            dump(self.std_scalar, self.std_path, compress=True)

    def predict_actions(self, state):
        """
        predicts the actions propabilities from actor model. 
        And samples an action using the actions propabilities
        """
        if self.rnn:
            prediction = self.Actor.predict(state)[0]
        else:
            prediction = self.Actor.predict(np.reshape(state, (1,-1)))[0]
        action = np.random.choice(self.action_dim, p=prediction)
        return prediction, action
    
    def save(self):
        # print("Saving Actor Critic Models")
        self.Actor.save(self.Actor_name)
        self.Critic.save(self.Critic_name)
        self.Actor_Critic.save(self.Actor_Critic_name)
    
    def discount_rewards(self, rewards):
        reward_sum = 0
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards -= np.mean(discounted_rewards) # normalizing the result
        discounted_rewards /= (np.std(discounted_rewards) + 1e-10) # divide by standard deviation, added 1e-10 for numerical stability
        return discounted_rewards

    def replay(self, states, actions, rewards, predictions):
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)
        discounted_r = np.vstack(self.discount_rewards(rewards))
        values = self.Critic.predict(states)
  
        advantages = discounted_r - values
        
        y_true = np.hstack([advantages, predictions, actions])

        # self.Actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))
        # self.Critic.fit(states, discounted_r, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))
        Y = [y_true, discounted_r]
        self.Actor_Critic.fit(states, Y, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))


    def samples(self):
        states, actions, rewards, predictions = [], [], [], []
        state = self.env.reset()
        done, score = False, 0
        while not done:
            state = self.std_scalar.transform(np.reshape(state, (1,-1)))
            if self.e==0:
                action = np.random.choice(3)
                prediction = np.full((3,), 1/3)
            else:
                prediction, action  = self.predict_actions(state)
            next_state, reward, done, _ = self.env.step(action)
            actions_onehot = tf.keras.utils.to_categorical(action, self.action_dim)
            states.append(state)
            actions.append(actions_onehot)
            predictions.append(prediction)
            rewards.append(reward)
            state = next_state
            score += reward
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)
        discounted_r = np.vstack(self.discount_rewards(rewards))
        values = self.Critic.predict(states)

        advantages = discounted_r - values
        
        y_true = np.hstack([advantages, predictions, actions])

        #Y = [y_true, discounted_r]

        return states, y_true, values, discounted_r , score

    def samples_rnn(self):
        states, actions, rewards, predictions = [], [], [], []
        state = self.env.reset()
        for _ in range(self.rnn_steps - 1):
            state = self.std_scalar.transform(np.reshape(state, (1,-1)))
            states.append(state.tolist())
            state, _, _, _ = self.env.step(self.env.action_space.sample())

        done, score = False, 0
        while not done:
            state = self.std_scalar.transform(np.reshape(state, (1,-1)))
            states.append(state.tolist())
            state = np.reshape(states[-self.rnn_steps:], (1, self.rnn_steps, -1))
            #prediction, action  = self.predict_actions(states[-self.rnn_steps:])
            if self.e==0:
                action = np.random.choice(3)
                prediction = np.full((3,), 1/3)
            else:
                prediction, action  = self.predict_actions(state)
            next_state, reward, done, _ = self.env.step(action)
            actions_onehot = tf.keras.utils.to_categorical(action, self.action_dim)
            actions.append(actions_onehot)
            predictions.append(prediction)
            rewards.append(reward)
            state = next_state
            score += reward
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)
        discounted_r = np.vstack(self.discount_rewards(rewards))
        states_rnn = [states[i:i+self.rnn_steps].tolist() for i in range(len(states) - self.rnn_steps+1)]
        states_rnn = np.reshape(states_rnn, (-1, self.rnn_steps, self.state_dim))
        values = self.Critic.predict(states_rnn)
        # print("values", values)
        # print("discounted_r", np.shape(discounted_r))
        advantages = discounted_r - values
        
        y_true = np.hstack([advantages, predictions, actions])

        #Y = [y_true, discounted_r]
        return states_rnn, y_true, discounted_r , score     

    def run(self):
        for self.e in range(self.EPSIODES):
            self.Actor.save(self.Actor_name)
            self.Critic.save(self.Critic_name)
            save_model = ''
            #memory buffers of states, actions, rewards, predicted action propabilities
            states, y_true, discounted_r, score = [], [], [], []
            t0 = time.perf_counter()
            for _ in range(self.traj_per_episode):
                if not self.rnn:
                    states_new, y_true_new, discounted_r_new, score_new = self.samples()
                else:
                    states_new, y_true_new, discounted_r_new, score_new = self.samples_rnn()
                
                
                states.append(states_new)
                y_true.append(y_true_new)
                discounted_r.append(discounted_r_new)
                score.append(score_new)
            
            states          = np.vstack(states)
            y_actions       = np.vstack(y_true)
            y_values        = np.vstack(discounted_r)
            # discounted_r    = np.vstack(discounted_r)
            t1 = time.perf_counter()
            print("Time taken to collect all the samples: ",t1-t0)
            mean_score = np.mean(score)
            self.scores.append(mean_score)
            average = sum(self.scores[-50:]) / len(self.scores[-50:])
            self.averages.append(average)
            self.episodes.append(self.episode)
            if average >= self.max_avg_reward :
                self.max_avg_reward = average
                self.save()
                save_model = 'SAVING'
            else:
                save_model = ""
            print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.e, self.EPSIODES, mean_score, average, save_model))
            outputs     = {'actions':y_actions, 'values': y_values}
            self.Actor_Critic.fit(x = states, y = outputs, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(states))
            # self.Actor_Critic.fit(x = states, y = [y_actions, y_values], epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(states))    

    def run_mpi(self):
        current_actor  = self.path + "/" + self.Model_name + '_Current_Actor.h5'
        current_critic  = self.path + "/" + self.Model_name + '_Current_Critic.h5'
        for self.e in range(self.EPSIODES):
            self.Actor.save(current_actor)
            self.Critic.save(current_critic)
            env_list = self.env_list #[copy.deepcopy(self.env) for _ in range(self.traj_per_episode)]
            save_model = ''
            #memory buffers of states, actions, rewards, predicted action propabilities
            states, y_true, discounted_r, score = [], [], [], []
            t0 = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:        
                partial_sampler = partial(
                    sampler,
                    rnn_steps       = self.rnn_steps, 
                    episode_num     = self.e, 
                    gamma           = self.gamma, 
                    std_scalar_path = self.std_path, 
                    actor_path      = current_actor, 
                    critic_path     = current_critic
                    )
                future_to_samples = {executor.submit(partial_sampler, env): env for env in env_list}

            for future in concurrent.futures.as_completed(future_to_samples):
                states_new, y_true_new, discounted_r_new, score_new = future.result()
                states.append(states_new)
                y_true.append(y_true_new)
                discounted_r.append(discounted_r_new)
                score.append(score_new)
            t1 = time.perf_counter()
            print("Time taken to collect all the samples: ",t1-t0)
            states          = np.vstack(states)
            y_actions       = np.vstack(y_true)
            y_values        = np.vstack(discounted_r)
            # discounted_r    = np.vstack(discounted_r)

            mean_score = np.mean(score)
            self.scores.append(mean_score)
            average = sum(self.scores[-50:]) / len(self.scores[-50:])
            self.averages.append(average)
            self.episodes.append(self.episode)
            if average >= self.max_avg_reward :
                self.max_avg_reward = average
                self.save()
                save_model = 'SAVING'
            else:
                save_model = ""
            print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.e, self.EPSIODES, mean_score, average, save_model))
            outputs     = {'actions':y_actions, 'values': y_values}
            self.Actor_Critic.fit(x = states, y = outputs, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(states))
            # self.Actor_Critic.fit(x = states, y = [y_actions, y_values], epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(states))    


    def load(self):
        if os.path.isfile(self.Actor_name):
            print('loading Actor weights')
            #Actor_name = self.path + "/" + self.Model_name + '_Actor.h5'
            self.Actor = load_model(self.Actor_name, compile=False)
        if os.path.isfile(self.Critic_name):
            print('loading Critic weights')
            #Actor_name = self.path + "/" + self.Model_name + '_Actor.h5'
            self.Critic = load_model(self.Critic_name, compile=False)
        if os.path.isfile(self.Actor_Critic_name):
            print('loading Actor_Critic weights')
            #Actor_Critic_name = self.path + "/" + self.Model_name + '_Actor_Critic_name.h5'
            self.Critic = load_model(self.Actor_Critic_name, compile=False)

    def test(self, savefig_filename = None):
        self.load()
        states = []
        test_env = self.test_env #SEIR_v0_2(discretizing_time = 5, sampling_time = 1, sim_length = 100)
        state = test_env.reset()
        if self.rnn:
            for _ in range(self.rnn_steps - 1):
                state = self.std_scalar.transform(np.reshape(state, (1,-1)))
                states.append(state.tolist())
                state, _, _, _ = self.test_env.step(self.test_env.action_space.sample())
        done = False
        # memory buffers of states, actions, rewards, predicted action propabilities
        # states, actions, rewards, predictions = [], [], [], []
        while not done:
            state = self.std_scalar.transform(np.reshape(state, (1,-1)))
            if self.rnn:
                states.append(state.tolist())
                state = states[-self.rnn_steps:]
                state = np.reshape(state, (1, self.rnn_steps, -1))
            _, action  = self.predict_actions(state)
            state, _, done, _ = test_env.step(action)
        savefig_filename = self.path + "/" + savefig_filename
        test_env.plot(savefig_filename = savefig_filename)

