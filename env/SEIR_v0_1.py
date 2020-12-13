import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt


class SEIR_v0_1(gym.Env):
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
            
            
    Observation*:
            Type: Box(4,)
            Num     Observation       Min     Max
            0       Susceptible       0       Total Population
            1       Exposed           0       Total Population
            2       Infected          0       Total Population
            3       Recovered         0       Total Population
            
    
    Actions*:
            Type: Box(4,), min=0 max=2
            Num     Action                                   Change in model
            0       Lockdown                                 affect transmission rate
            1       Social distancing                        affect transmission rate     
            2       No Social distancing                     affect transmission rate
            

    Reward:
            reward = lambda * economic cost + (1-lambda) * public health cost
            
            Economic cost:
                - sum(action)
            Health cost:
                -0.00002* number of infected
            lambda:
                a user defined weight. Default 0.5

    Episode Termination:
            Episode length (time) reaches specified maximum (end time)
            The end of analysis period is 100 days
    """


    metadata = {'render.modes': ['human']}

    def __init__(self, discretizing_time = 5, sampling_time = 1, action_max = 2, sim_length = 100):
        super(SEIR_v0_1, self).__init__()

        self.dt           = discretizing_time/(24*60)
        self.Ts           = sampling_time
        self.time_steps   = int((self.Ts) / self.dt)
        self.n_agents     = 1

        self.popu         = 100000
        self.current      = 1
        self.pred         = 1
        self.trainNoise   = False
        self.weight       = 0.5 #reward weighting

        #model paramenters
        self.theta        = np.full(shape=self.n_agents, fill_value=2, dtype=float)#np.array([2, 2, 2, 2], dtype = float) #choose a random around 1
        self.d            = np.full(shape=self.n_agents, fill_value=1/24, dtype=float)#np.array([1/24, 1/24, 1/24, 1/24], dtype = float) # 1 hour or 1/24 days

        #crowd density = np.full(shape=4, fill_value=6, dtype=float)
        self.beta         = self.theta * self.d * np.full(shape=self.n_agents, fill_value=1, dtype=float) #needs to be changed
        self.sigma        = 1.0/5  # needds to be changed
        self.gamma        = 0.05 #needs to be changed

        self.action_max   = action_max     # Maximum value of action
        self.n_actions    = action_max + 1 #total number of actions 

        #gym action space and observation space
        self.action_space = spaces.MultiDiscrete([self.n_actions for _ in range(self.n_agents)])
        #spaces.Box(low = 0, high = self.action_max, shape = (4,), dtype = np.float64)
        self.observation_space = spaces.Box(0, np.inf, shape=(4,), dtype=np.float64)

        #Total number of simulation days
        self.sim_length   = sim_length
        self.daynum       = 0

        #seeding
        self.seed()

        # initialize state
        self.get_state()

        #memory to save the trajectories
        self.state_trajectory = []
        self.action_trajectory = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_state(self):
        self.state = np.array([self.popu-200-1000, 1000, 200, 0], dtype=float)

    def set_state(self, state):
        tot_state = sum(state)
        err_msg = "%s is Invalid. S+E+I+R not equal to %s"  % (state, self.popu)
        assert self.popu==sum(state), err_msg
        self.state = state
    
    def mini_step(self, action):

        # action should be with in 0 - 2
        # 
        self.beta = self.theta * self.d * action
        S, E, I, R = self.state

        dS = - sum(self.beta) * I * S / self.popu
        dE = - dS - (self.sigma * E)
        dI = (self.sigma * E) - (self.gamma * I)
        dR = (self.gamma * I)

        new_S = S + self.dt * dS
        new_E = E + self.dt * dE
        new_I = I + self.dt * dI
        new_R = R + self.dt * dR

        return np.array([new_S, new_E, new_I, new_R], dtype =float)

    def step(self, action):

        self.daynum += 1

        for _ in range(self.time_steps):
            self.state = self.mini_step(action)

        # Costs
        # action represent the crowd density, so decrease in crowd density increases the economic cost
        economicCost = - (2 / (self.action_max*self.n_agents*1.5)) * np.sum(action)

        # Public health Cost increases with increase in Infected people.
        publichealthCost   =  0.00002*abs(self.state[2])
        
        #Rewards
        reward = - self.weight * economicCost - (1 - self.weight) * publichealthCost

        # Check if episode is over
        done = bool(self.state[2] < 0.5 or self.daynum == self.sim_length)

        # saving the states and actions in the memory buffers
        self.state_trajectory.append(list(self.state))
        self.action_trajectory.append(action)
        return self.state, reward, done, {}
        
    def reset(self):

        # reset to initial conditions
        self.daynum = 0
        self.get_state()

        #memory reset
        self.state_trajectory = []
        self.action_trajectory = []

        return self.state
    
    def plot(self, savefig_filename=None):
        title_states = ['Susceptible', 'Exposed', 'Infected', 'Removed']
        test_steps = self.daynum
        time = np.array(range(test_steps), dtype=np.float32)*self.Ts
        test_obs_reshape = np.concatenate(self.state_trajectory).reshape((test_steps ,self.observation_space.shape[0]))
        #test_act_reshape = self.action_trajectory#np.concatenate(self.action_trajectory).reshape((test_steps ,self.action_space.shape[0]))
        state_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        #total_dim = self.observation_space.shape[0] + self.action_space.shape[0]

        fig, ax = plt.subplots(nrows=1, ncols=state_dim, figsize = (24,4))
        for i in range(state_dim):
            ax[i].plot(time, test_obs_reshape[:, i], label=title_states[i])
            #ax[i].set_ylim(des[i]-50, des[i]+50)
            ax[i].set_title(title_states[i], fontsize=15)
            ax[i].set_xlabel('Time', fontsize=10)
            ax[i].set_ylim(0, self.popu)
            ax[i].set_label('Label via method')
            ax[i].legend()
        plt.show()
