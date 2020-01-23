import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pdb import set_trace as st
import time
env = gym.make("MountainCar-v0")

lr = [0.5, 0.55, 0.6]
gamma = [0.9, 0.95, 1]
q_table_size = [10, 20, 100]

total_episode = 10000
display = 500
epsilons = [0, 0.5, 1]

def epsilon_dacayer(epsilon, ori):
    delta_epsilon = ori/(total_episode//
env.close()2)
    return epsilon-delta_epsilon

def get_integrate_state(state):
    integrate_state = (state - env.observation_space.low) // stride
    return tuple(integrate_state.astype(int))

def epsilon_greedy_action(state, epsilon):
    integrate_state = get_integrate_state(state)
    if np.random.random() < epsilon:
        action = np.random.randint(0,env.action_space.n)
    else:
        action = np.argmax(q_table[integrate_state])
    return action

env.reset()
# 创建Q表

epsilon_rewards = []
displayer = {'ep':[],'reward0':[],'reward1':[],'reward2':[]}
for i in range(3):
    env.reset()
    num_state = len(env.observation_space.low)
    stride = (env.observation_space.high - env.observation_space.low) / q_table_size[1]
    epsilon = epsilons[i]
    q_table = np.zeros([q_table_size[1]]*num_state+[env.action_space.n])
    print(q_table.shape)
    time_start = time.time()
    total_succ = 0
    for episode in range(total_episode):
        # initiate reward every episode
        epsilon_reward = 0
        # total_succ = 0
        if episode % display == 0:
            print("episode: {}".format(episode))

        state = env.reset()
        done = False
        while not done:
            # ACTION
            action = epsilon_greedy_action(state, epsilon)
            # receive info from environment
            next_state, reward, done, _ = env.step(action)
            epsilon_reward += reward

            if not done:
                # update q_table
                td_target = reward + gamma[1] * np.max(q_table[get_integrate_state(next_state)])
                q_table[get_integrate_state(state)][action] += lr[1] * (td_target - q_table[get_integrate_state(state)][action])

            elif next_state[0] >= 0.5:#成功
                q_table[get_integrate_state(state)][action] = 0
                total_succ = total_succ + 1

            state = next_state

        # epsilon decay
        if episode < total_episode//2:
            epsilon = epsilon_dacayer(epsilon, epsilons[i])
        if(episode % display == 0):
            print("epsilon: {} in {} episode".format(epsilon, episode))

        # record aggrated rewards on each epsoide
        epsilon_rewards.append(epsilon_reward)

        # every display calculate average rewards
        if episode % display == 0 and episode!=0:
            average_reward = sum(epsilon_rewards[-display:]) / len(epsilon_rewards[-display:])
            print("{} Success in {} episodes, rate: {} %".format(total_succ, display, 100*total_succ/display))
            print("Average Reward: {}".format(average_reward))
            total_succ = 0
            if i==0:
                displayer['ep'].append(episode)
            displayer['reward'+str(i)].append(average_reward)

    time_end = time.time() 

print("Training Q Learning Done in " + str(time_end-time_start) + " seconds.")

######## DISPLAY #################
plt.plot(displayer['ep'], displayer['reward0'], label = 'epsilon:0')
plt.plot(displayer['ep'], displayer['reward1'], label = 'epsilon:0.5')
plt.plot(displayer['ep'], displayer['reward2'], label = 'epsilon:1.0')
plt.legend()
plt.xlabel('total_episode')
plt.ylabel('Rewards')
plt.show()
env.close()
