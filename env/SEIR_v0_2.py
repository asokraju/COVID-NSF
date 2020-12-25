import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import matplotlib.pyplot as plt

class SEIR_v0_2(gym.Env):
    """
    Description:
            Each city's population is broken down into four compartments --
            Susceptible, Exposed, Infectious, and Removed -- to model the spread of
            COVID-19.

    Source:
            Code modeled after cartpole.py from
            github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    
    Time:
            discretizing_time: time in minutes used to discretizing the model
            sampling_time: time in days that we sample from the system
            sim_length: time in days
            
            
    Observation*:
            Type: Box(4,)
            Num     Observation       Min     Max
            0       Susceptible       0       Total Population
            1       Exposed           0       Total Population
            2       Infected          0       Total Population
            3       Recovered         0       Total Population
            
    
    Actions*:
            Type: Box(4,), min=0 max=2
            Num     Action                                   Change in model                Crowd density
            0       Lockdown                                 affect transmission rate       0
            1       Social distancing                        affect transmission rate       0.5-1 = 0.75     
            2       No Social distancing                     affect transmission rate       1-5   = 1.5
            

    Reward:
            reward = lambda * economic cost + (1-lambda) * public health cost
            
            Economic cost:
            Num       Action                                    Crowd density               cost
            0         Lockdown                                  0                           1.0
            1         Social distancing                         0.5-1 = 0.75                0.25
            2         No Social distancing (regular day)        1-5   = 1.5                 0.0

            Health cost:                                min                     max
                1.0 - 0.00001* number of infected      0.0                      1.0
            lambda:
                a user defined weight. Default 0.5

    Episode Termination:
            Episode length (time) reaches specified maximum (end time)
            The end of analysis period is 100 days
    """


    metadata = {'render.modes': ['human']}

    def __init__(self, discretizing_time = 5, sampling_time = 1, sim_length = 100):
        super(SEIR_v0_2, self).__init__()

        self.dt           = discretizing_time/(24*60)
        self.Ts           = sampling_time
        self.time_steps   = int((self.Ts) / self.dt)
       
        self.popu         = 1e5 #100000
        self.trainNoise   = False
        self.weight       = 0.5 #reward weighting

        #model paramenters
        self.theta        = np.full(shape=1, fill_value=2, dtype=float)#np.array([2, 2, 2, 2], dtype = float) #choose a random around 1
        self.d            = np.full(shape=1, fill_value=1/24, dtype=float)#np.array([1/24, 1/24, 1/24, 1/24], dtype = float) # 1 hour or 1/24 days

        #crowd density = np.full(shape=1, fill_value=6, dtype=float)
        #self.beta         = self.theta * self.d * np.full(shape=1, fill_value=1, dtype=float) #needs to be changed
        self.sigma        = 1.0/5   # needds to be changed
        self.gamma        = 0.05    # needs to be changed

        self.n_actions    = 3                                               # total number of actions 
        self.rho          = np.array([0, 0.75, 1.5], dtype=float)      # possible actions (crowd_densities)

        #Economic costs 
        self.eco_costs    = np.array([1, 0.25, 0], dtype=float) 

        #gym action space and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, np.inf, shape=(4,), dtype=np.float64)

        #Total number of simulation days
        self.sim_length   = sim_length
        self.daynum       = 0

        #seeding
        self.seed()

        # initialize state
        self.get_state()

        #memory to save the trajectories
        self.state_trajectory  = []
        self.action_trajectory = []
        self.rewards            = []
        self.count             = 0
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_state(self):
        init_I = np.random.randint(200, high=500) 
        init_E = np.random.randint(900, high=1200)  
        init_S = self.popu - init_I - init_E
        self.state = np.array([init_S, init_E, init_I, 0], dtype=float)

    def set_state(self, state):
        err_msg = "%s is Invalid. S+E+I+R not equal to %s"  % (state, self.popu)
        assert self.popu==sum(state), err_msg
        self.state = state
    
    def mini_step(self, rho):

        # action should be with in 0 - 2
        # 
        self.beta = self.theta * self.d * rho
        S, E, I, R = self.state

        dS = - (self.beta) * I * S / self.popu
        dE = - dS - (self.sigma * E)
        dI = (self.sigma * E) - (self.gamma * I)
        dR = (self.gamma * I)

        new_S = S + self.dt * dS
        new_E = E + self.dt * dE
        new_I = I + self.dt * dI
        new_R = R + self.dt * dR

        return np.array([new_S, new_E, new_I, new_R], dtype =float)

    def step(self, action):

        self.daynum += self.Ts

        for _ in range(self.time_steps):
            self.state = self.mini_step(self.rho[action])

            # saving the states and actions in the memory buffers
            self.state_trajectory.append(list(self.state))
            self.action_trajectory.append(action)
            self.count += 1
        # Costs
        # action represent the crowd density, so decrease in crowd density increases the economic cost
        economicCost = self.eco_costs[action] * self.Ts

        # Public health Cost increases with increase in Infected people.
        publichealthCost   =  (1e-4 * self.state[2]) * self.Ts
        
        # Rewards
        reward = - self.weight * economicCost - (1 - self.weight) * publichealthCost

        # Check if episode is over
        done = bool(self.state[2] < 0.5 or self.daynum >= self.sim_length)

        # saving the states and actions in the memory buffers
        #self.state_trajectory.append(list(self.state))
        #self.action_trajectory.append(action)
        for _ in range(self.time_steps):
            self.rewards.append(reward)
        return self.state, reward, done, {}
        
    def reset(self):

        # reset to initial conditions
        self.daynum = 0
        self.get_state()

        #memory reset
        self.state_trajectory = []
        self.action_trajectory = []
        self.rewards            = []
        self.count = 0

        return self.state

    def dataframe(self):
        title_states = ['Susceptible', 'Exposed', 'Infected', 'Removed']
        time = np.array(range(self.count), dtype=np.float32)*self.dt
        test_obs_reshape = np.concatenate(self.state_trajectory).reshape((self.count ,self.observation_space.shape[0]))
        self.df = pd.DataFrame(data=test_obs_reshape, index = time, columns=title_states)
        cat_type = pd.CategoricalDtype(categories=[0, 1, 2],ordered=True)
        self.df['actions'] = self.action_trajectory
        self.df['actions_cat'] = self.df.actions.astype(cat_type)
        self.df['rewards'] = self.rewards
        return self.df

    def plot(self, savefig_filename=None):
        df = self.dataframe()
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize = (10,15))

        df.iloc[:,:4].plot.area(ax=axes[0], stacked=True)
        axes[0].set_ylim(0, self.popu)

        df['actions'].plot(ax=axes[1], label = 'actions')
        axes[1].set_ylabel('actions', fontsize=15)
        axes[1].set_ylim(-0.5, 2.5)

        df['rewards'].plot(ax=axes[2], label = 'rewards')
        axes[2].set_ylabel('Rewards', fontsize=15)
        axes[2].set_xlabel('Time (days)', fontsize=15)
        if savefig_filename is not None:
            assert isinstance(savefig_filename, str), "filename for saving the figure must be a string"
            plt.savefig(savefig_filename, format = 'pdf')
            #plt.savefig('savefig_filename', format='eps')
        else:
            plt.show()

