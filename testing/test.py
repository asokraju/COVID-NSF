import numpy as np
from env.SEIR_V0_3 import SEIR_v0_3


import concurrent.futures 
import time
import threading
from threading import Thread, Lock

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def samples(env):
    # t0 = time.perf_counter()
    # print("started")
    States = []
    Actions = []
    Rewards = []
    s = env.reset()
    done = False
    while not done:
        States.append(s)
        a = np.random.randint(0, 3)
        Actions.append(a)
        s,r,done, _ = env.step(a)
        Rewards.append(r)
    # print("done")
    # t1 = time.perf_counter()
    # print(t1-t0)
    States = np.vstack(States)
    Actions = np.vstack(Actions)
    Rewards = np.vstack(Rewards)
    return States, Actions, Rewards

def predict_actions(state, actor):
    """
    predicts the actions propabilities from actor model. 
    And samples an action using the actions propabilities
    """
    prediction = actor.predict(state)[0]
    action = np.random.choice(3, p=prediction)
    return prediction, action

def worker(n=16):
    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        
    t0 = time.perf_counter()
    #     env_list = [SEIR_v0_3() for _ in range(16)]
    #     var =[16]
    #     results = executor.map(samples, env_list)
    pool = ThreadPool(4)
    pool = Pool()
    env_list = [SEIR_v0_3() for _ in range(n)]
    results = pool.map(samples, env_list)
    pool.close()
    pool.join()
    States, Actions, Rewards = [], [], []
    for S, A, R in results:
        States.append(S)
        Actions.append(A)
        Rewards.append(R)
        
    States = np.vstack(States)
    Actions = np.vstack(Actions)
    Rewards = np.vstack(Rewards)
    t1 = time.perf_counter()
    print(t1-t0)
    print(170==np.shape(States)[0]//n)
    return States, Actions, Rewards


if __name__ == '__main__':
    worker(n=102)





















































#Working
# import numpy as np
# from env.SEIR_V0_3 import SEIR_v0_3


# import concurrent.futures 
# import time

# def samples(env):
#     # t0 = time.perf_counter()
#     # print("started")
#     States = []
#     Actions = []
#     Rewards = []
#     s = env.reset()
#     done = False
#     while not done:
#         States.append(s)
#         a = np.random.randint(0, 3)
#         Actions.append(a)
#         s,r,done, _ = env.step(a)
#         Rewards.append(r)
#     # print("done")
#     # t1 = time.perf_counter()
#     # print(t1-t0)
#     States = np.vstack(States)
#     Actions = np.vstack(Actions)
#     Rewards = np.vstack(Rewards)
#     return States, Actions, Rewards

# def predict_actions(state, actor):
#     """
#     predicts the actions propabilities from actor model. 
#     And samples an action using the actions propabilities
#     """
#     prediction = actor.predict(state)[0]
#     action = np.random.choice(3, p=prediction)
#     return prediction, action

# def worker():
#     with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        
#         t0 = time.perf_counter()
#         env_list = [SEIR_v0_3() for _ in range(16)]
#         var =[16]
#         results = executor.map(samples, env_list)
        
#     States, Actions, Rewards = [], [], []
#     for S, A, R in results:
#         States.append(S)
#         Actions.append(A)
#         Rewards.append(R)
        
#     States = np.vstack(States)
#     Actions = np.vstack(Actions)
#     Rewards = np.vstack(Rewards)
#     t1 = time.perf_counter()
#     print(t1-t0)
#     print(170==np.shape(States)[0]//8)
#     return States, Actions, Rewards
# if __name__ == '__main__':
#     worker()



# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.compat.v1.keras.backend import set_session
# from tensorflow.keras import backend as K

# import numpy as np
# import concurrent.futures 
# import time
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 0:
#     print(f'GPUs {gpus}')
#     try: tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError: pass

# def simple_model():
#     model = keras.models.Sequential([
#         keras.layers.Dense(units = 10, input_shape = [1]),
#         keras.layers.Dense(units = 1, activation = 'sigmoid')
#     ])
#     model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
#     return model

# def clone_model(model):
#     model_clone = tf.keras.models.clone_model(model)
#     model_clone.set_weights(model.get_weights())
#     return model_clone

# def work(model, seq):
#     # model = clone_model(model)# model_list[model_id]
#     # print(model)
#     # import tensorflow as tf
#     model_clone = tf.keras.models.clone_model(model)
#     model_clone.set_weights(model.get_weights())
#     return model_clone.predict(seq)

# def worker(model, num_of_seq = 4):
#     seqences = np.arange(0,100).reshape(num_of_seq, -1)
#     with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:        
#         t0 = time.perf_counter()
#         # model_list = [clone_model(model) for _ in range(num_of_seq)]
#         index_list = np.arange(1, num_of_seq)# [clone_model(model) for _ in range(num_of_seq)]
#         # print(model_list)
#         future_to_samples = {executor.submit(work, model, seq): seq for seq in zip(seqences)}
#     Seq_out = []
#     for future in concurrent.futures.as_completed(future_to_samples):
#         out = future.result()
#         Seq_out.append(out)
#     t1 = time.perf_counter()
#     print(t1-t0)
#     return np.reshape(Seq_out, (-1, )), t1-t0



# if __name__ == '__main__':
#     model = simple_model()
#     num_of_seq = 4
#     # model_list = [clone_model(model) for _ in range(4)]
#     out = worker(model, num_of_seq=num_of_seq)
#     print(out)




# import tensorflow as tf
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Dense, Lambda, Add, Flatten, GRU
# from tensorflow.keras.optimizers import Adam, RMSprop
# from tensorflow.keras import backend as K
# import copy
# import numpy as np
# from env.SEIR_V0_3 import SEIR_v0_3
# import matplotlib.pyplot as plt

# import concurrent.futures 
# import time


# def AC_model_new(input_shape, action_dim, lr, EPSILON, C, rnn, rnn_steps):
#     """
#     defines Actor and Critic models that shares the input and the hidden layers
#     updated Actor and Critic together
#     """
#     if rnn:
#         input_layer = Input((rnn_steps, input_shape))
#         layer_1 = GRU(units=32, kernel_initializer='he_uniform')(input_layer)
#     else:
#         input_layer = Input(input_shape)
#         layer_1 = Dense(units=128, activation='elu', kernel_initializer='he_uniform')(input_layer)

#     layer_a = Dense(units=32, activation='elu', kernel_initializer='he_uniform')(layer_1) #input_layer
#     actions = Dense(units=action_dim, activation='softmax', kernel_initializer='he_uniform', name='actions')(layer_a)

#     layer_v = Dense(units=32, activation='elu', kernel_initializer='he_uniform')(layer_1)
#     values = Dense(units=1, activation='linear', kernel_initializer='he_uniform', name='values')(layer_v)

#     def ppo_loss(y_true, y_pred, EPSILON, C):
#         # Defined in https://arxiv.org/abs/1707.06347
#         advantages, predictions_old, actions_onehot = y_true[:, :1], y_true[:, 1:1+action_dim], y_true[:, 1+action_dim:]
#         # EPSILON = 0.1
#         # C = 5e-2
#         prob = y_pred * actions_onehot
#         old_prob = predictions_old * actions_onehot
#         r = prob / (old_prob +1e-10)
        
#         p1 = r * advantages
#         p2 = K.clip(r, min_value=1-EPSILON, max_value=1+EPSILON) * advantages
#         entropy_loss = -prob * K.log(prob + 1e-10)

#         loss = -K.mean(K.minimum(p1, p2) + C * entropy_loss)
#         return loss
    
#     def ppo_parametrized(EPSILON, C):
#         def ppo_loss_fun(y_true, y_pred):
#             return ppo_loss(y_true, y_pred, EPSILON, C)
#         return ppo_loss_fun

    
#     lossWeights  = {'actions':0.5, 'values':0.5}
#     lossFuns     = {'actions':ppo_parametrized(EPSILON, C), 'values':'mse'}

#     Actor = Model(inputs = input_layer, outputs = actions)
#     Critic = Model(inputs = input_layer, outputs = values)
#     Actor_Critic = Model(inputs = input_layer, outputs = [actions, values])
#     Actor_Critic.compile(optimizer=RMSprop(lr=lr), loss=lossFuns, loss_weights=lossWeights)
#     # Actor_Critic.compile(optimizer=RMSprop(lr=lr), loss=[ppo_parametrized(EPSILON, C), 'mse'], loss_weights=[0.5, 0.5])
#     return Actor, Critic, Actor_Critic

# def networks(
#     exp_name = 'seir',
#     EPSIODES = 10000 ,
#     lr = 0.0001, 
#     EPOCHS = 10,
#     path = None,
#     gamma = 0.95,
#     traj_per_episode = 4,
#     EPSILON = 0.1,
#     C = 1e-2,
#     rnn = True,
#     rnn_steps = 1,
#     state_dim = 4,
#     action_dim = 3):
#     Actor, Critic, Actor_Critic = AC_model_new(
#             input_shape    =    state_dim, 
#             action_dim     =    action_dim, 
#             lr             =    lr, 
#             EPSILON        =    EPSILON, 
#             C              =    C,
#             rnn            =    rnn,
#             rnn_steps      =    rnn_steps
#             )
#     return Actor, Critic, Actor_Critic

# def discount_rewards(rewards, gamma = 0.99):
#     reward_sum = 0
#     discounted_rewards = []
#     for reward in rewards[::-1]:
#         reward_sum = reward + gamma * reward_sum
#         discounted_rewards.append(reward_sum)
#     discounted_rewards.reverse()
#     discounted_rewards = np.array(discounted_rewards)
#     discounted_rewards -= np.mean(discounted_rewards) # normalizing the result
#     discounted_rewards /= (np.std(discounted_rewards) + 1e-10) # divide by standard deviation, added 1e-10 for numerical stability
#     return discounted_rewards

# def predict_actions(state, actor):
#     """
#     predicts the actions propabilities from actor model. 
#     And samples an action using the actions propabilities
#     """
#     prediction = actor.predict(state)[0]
#     action = np.random.choice(3, p=prediction)
#     return prediction, action

# def samples(env, var):
#     # t0 = time.perf_counter()
#     # print("started")
#     States = []
#     Actions = []
#     Rewards = []
#     s = env.reset()
#     done = False
#     while not done:
#         States.append(s)
#         a = np.random.randint(0, 3)
#         Actions.append(a)
#         s,r,done, _ = env.step(a)
#         Rewards.append(r)
#     # print("done")
#     # t1 = time.perf_counter()
#     # print(t1-t0)
#     discounted_r = np.vstack(discount_rewards(Rewards))
#     States = np.vstack(States)
#     Actions = np.vstack(Actions)
#     # Rewards = np.vstack(Rewards)
#     # print(var)
#     return States, Actions, discounted_r

# def samples_rnn(env, actor, critic):
#     rnn_steps = 1
#     action_dim = 3
#     state_dim = 4
#     print("running rnn samples")
#     states, actions, rewards, predictions = [], [], [], []
#     state = env.reset()
#     for _ in range(rnn_steps - 1):
#         state = np.reshape(state, (1,-1))
#         states.append(state.tolist())
#         state, _, _, _ = env.step(env.action_space.sample())

#     done, score = False, 0
#     while not done:
#         state = np.reshape(state, (1,-1))
#         states.append(state.tolist())
#         state = np.reshape(states[-rnn_steps:], (1, rnn_steps, -1))
#         # time_0 = time.perf_counter()
#         prediction, action  = predict_actions(states[-rnn_steps:], actor)
#         # print(time.perf_counter()-time_0)
#         # prediction, action  = np.full((3,), 1/3), np.random.choice(3)
#         # predict_actions(state, actor)
#         next_state, reward, done, _ = env.step(action)
#         actions_onehot = tf.keras.utils.to_categorical(action, action_dim)
#         actions.append(actions_onehot)
#         predictions.append(prediction)
#         rewards.append(reward)
#         state = next_state
#         score += reward
#     states = np.vstack(states)
#     actions = np.vstack(actions)
#     predictions = np.vstack(predictions)
#     discounted_r = np.vstack(discount_rewards(rewards))
#     states_rnn = [states[i:i+rnn_steps].tolist() for i in range(len(states) - rnn_steps+1)]
#     states_rnn = np.reshape(states_rnn, (-1, rnn_steps, state_dim))
#     values = critic.predict(states_rnn)
#     # print("values", values)
#     # print("discounted_r", np.shape(discounted_r))
#     advantages = discounted_r - values
    
#     y_true = np.hstack([advantages, predictions, actions])

#     #Y = [y_true, discounted_r]
#     return states_rnn, y_true, discounted_r , score  
# def clone_model(model):
#     model_clone = tf.keras.models.clone_model(model)
#     model_clone.set_weights(model.get_weights())
#     return model_clone
# def worker(actor, critic, num_of_envs = 4):
#     with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        
#         t0 = time.perf_counter()
#         var = 0
#         env_list = [SEIR_v0_3() for _ in range(num_of_envs)]
#         actor_list = [clone_model(actor) for _ in range(num_of_envs)]
#         # actor_list = [1 for _ in range(num_of_envs)]
#         # print(actor_list)
#         critic_list = [clone_model(critic) for _ in range(num_of_envs)]
#         # critic_list = [1 for _ in range(num_of_envs)]
#         # print(critic_list)
#         # results = executor.map(samples, env_list)
#         # samples_rnn(env, actor, critic)
#         future_to_samples = {executor.submit(samples_rnn, env, actor, critic): env for env, actor, critic in zip(env_list, actor_list, critic_list)}
#         # future_to_samples = {executor.submit(samples_rnn, env, clone_model(actor), clone_model(critic)): env for env in env_list}
#     States, Actions, Rewards = [], [], []
#     for future in concurrent.futures.as_completed(future_to_samples):
#         S, A, R, score = future.result()
#         States.append(S)
#         Actions.append(A)
#         Rewards.append(R)
        
#     States = np.vstack(States)
#     Actions = np.vstack(Actions)
#     Rewards = np.vstack(Rewards)
#     t1 = time.perf_counter()
#     print(t1-t0)
#     print(170==np.shape(States)[0]//num_of_envs)
#     return States, Actions, Rewards, t1-t0

# if __name__ == '__main__':
#     # Time = []
#     # steps = np.arange(1, 201, 5)
#     # for t in steps:
#     #     _, _, _, tt = worker(num_of_envs = t)
#     #     Time.append(tt)
#     # print(Time, steps)
#     # plt.plot(steps, Time)
#     # # plt.ylim()
#     # plt.show()
#     actor, critic, actor_critic = networks()
#     worker(actor, critic, num_of_envs=8)