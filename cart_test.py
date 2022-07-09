import gym, time
import numpy as np

env = gym.make('CartPole-v1')
obsevation = [30, 30, 50, 50]
observation_step_size = np.array([0.25, 0.25, 0.01, 0.1])

done = False

def get_discrete_state(state):
    print(type(state))
    discrete_state = state/observation_step_size + np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int))

i = 25000
q_table = np.load(f"QTable/{i}-q_table.npy")

discrete_state = get_discrete_state(env.reset())


for episode in range(i+1):
    action = np.argmax(q_table[discrete_state])

    while not done:
        new_state, reward, done, _ = env.step(action)

        env.render()
        for _ in range(1000000):
            continue
        new_discrete_state = get_discrete_state(new_state)
    discrete_state = new_discrete_state
    if episode % 100 == 0:
        print(f"EPISODE: {episode}")
        print("Epsilon: " + str(discrete_state))

env.close()
