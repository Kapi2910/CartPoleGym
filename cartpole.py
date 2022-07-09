'''
    TITLE: Q-Learning on a Cart Pole 
    MADE BY: Aaditya Prakash Kattekola
'''
from constants import constants, utilities, gym, np, time, math

utils = utilities()
env = gym.make('CartPole-v1')
q_table = np.random.uniform(low=0, high=1, size=(constants.obsevation + [env.action_space.n]))

#q_table = np.load(f"QTable/{i}-q_table.npy")

for episode in range(constants.EPISODES + 1):
    t0 = time.time()
    discrete_state = utils.get_discrete_state(env.reset())
    done = False
    episode_reward = 0

    if episode % constants.SHOW == 0:
        render = True
        print("Episode: " + str(episode))
    else:
        render = False

    while not done:
        if np.random.random() > constants.epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, constants.reward, done, _ = env.step(action)
        episode_reward += constants.reward
        new_discrete_state = utils.get_discrete_state(new_state)

        if episode % constants.SHOW == 0:
            env.render()
       
        # If simulation did not end yet after last step - update Q table
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - constants.alpha) * current_q + constants.alpha * (constants.reward + constants.gamma * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state


    # Decaying is being done every episode if episode number is within decaying range
    if constants.epsilon >= 0.05:
        if episode_reward > constants.prior_reward and episode > 10000:
            constants.epsilon = math.pow(constants.epsilon_decay_value, episode - 10000)

            if episode % 500 == 0:
                print("Epsilon: " + str(constants.epsilon))


    t1 = time.time()

    constants.episode_time = t1-t0
    constants.total_time = constants.total_time + constants.episode_time

    constants.total_reward += episode_reward 
    constants.prior_reward = episode_reward

    if episode % 1000 == 0: 
        mean = constants.total_time / 1000
        print("Time Average: " + str(mean))
        constants.total = 0

        mean_reward = constants.total_reward / 1000
        print("Mean Reward: " + str(mean_reward))
        constants.total_reward = 0

        #np.save(f"QTable/{episode}-q_table.npy", q_table)


env.close()

