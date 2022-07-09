import gym
import numpy as np
import time, math

class constants:

    # SIMULATION SETTINGS
    EPISODES = 25000 # Total Number of Episodes
    SHOW = 5000 # Show the simulation after every SHOW episodes
    obsevation = [25, 30, 45, 50] # Number of steps per observation
    observation_step_size = np.array([0.25, 0.2, 0.01, 0.15]) # Step size per obsevarion
    adjustment = np.array([15,10,1,10]) # Required to make the discrete state effective

    # LEARNING SETTINGS
    alpha = 0.1 # LEARNING RATE
    gamma = 0.95 # DISCOUNT RATE
    
    
    # EXPLORATION SETTINGS
    epsilon = 1  # 
    epsilon_decay_value = 0.999555
    
    
    # METRICS
    total_time = 0 # Total Time 
    episode_time = 0 # Time Per Episode
    total_reward = 0 # Total Reward
    prior_reward = 0 # Reward of Previous Episode

class utilities:

    def __init__(self):
        self.nudge = np.array([0, 0, 0, 0])

    # UTILITY FUNCTION
    def get_discrete_state(self, state):
        '''
            Converts the continuous state into a discrete state
        '''
        self.nudge = state / constants.observation_step_size
        discrete_state = self.nudge + constants.adjustment
        return tuple(discrete_state.astype(int))