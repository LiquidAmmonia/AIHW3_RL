##### 实现了Q_Learning SARSA E-SARSA 三种方法
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pdb import set_trace as st
import time
env = gym.make("MountainCar-v0")

lr = 0.54
gamma = 0.95
q_table_size = 20

total_episode = 10000
display = 500

env.reset()
# 创建Q表
num_state = len(env.observation_space.low)
stride = (env.observation_space.high - env.observation_space.low) / q_table_size

epsilon = 1
index = 1 # 1---QLearning  2---SARSA  3---E-SARSA

def epsilon_dacayer(epsilon):
    delta_epsilon = 1/(total_episode//2)
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

epsilon_rewards = []
aggr_epsilon_rewards = {'ep':[],'Q-Learning':[],'SARSA':[],'E-SARSA':[]}

################# Q_LEARNING ################
print("Start training using Q-Learning...")
q_table = np.zeros([q_table_size]*num_state+[env.action_space.n])
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
            td_target = reward + gamma * np.max(q_table[get_integrate_state(next_state)])
            q_table[get_integrate_state(state)][action] += lr * (td_target - q_table[get_integrate_state(state)][action])

        elif next_state[0] >= 0.5:
            q_table[get_integrate_state(state)][action] = 0
            total_succ = total_succ + 1

        state = next_state

    # epsilon decay
    if episode < total_episode//2:
        epsilon = epsilon_dacayer(epsilon)
    if(episode % display == 0):
        print("epsilon: {} in {} episode".format(epsilon, episode))

    # record aggrated rewards on each epsoide
    epsilon_rewards.append(epsilon_reward)

    # every display calculate average rewards
    if episode % display == 0:
        average_reward = sum(epsilon_rewards[-display:]) / len(epsilon_rewards[-display:])
        print("{} Success in {} episodes, rate: {} %".format(total_succ, display, 100*total_succ/display))
        print("Average Reward: {}".format(average_reward))
        total_succ = 0
        aggr_epsilon_rewards['ep'].append(episode)
        aggr_epsilon_rewards['Q-Learning'].append(average_reward)

time_end = time.time() 

print("Training Q Learning Done in " + str(time_end-time_start) + " seconds.")

done = False
state = env.reset()
print("Testing Q Learning...")
while not done:
    action = np.argmax(q_table[get_integrate_state(state)])
    next_state, _, done, _ = env.step(action)
    state = next_state
    # env.render()
print("Testing Q Learning done.")

############# SARSA ################
print("Start training using SARSA...")
q_table = np.zeros([q_table_size]*num_state+[env.action_space.n])
print(q_table.shape)
env.reset()
epsilon = 1

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
        next_action = epsilon_greedy_action(next_state, epsilon)

        if not done:
            # update q_table
            td_target = reward + gamma * q_table[get_integrate_state(next_state)][next_action]
            q_table[get_integrate_state(state)][action] += lr * (td_target - q_table[get_integrate_state(state)][action])

        elif next_state[0] >= 0.5:
            q_table[get_integrate_state(state)][action] = 0
            total_succ = total_succ + 1

        action = next_action
        state = next_state

    # epsilon decay
    if episode < total_episode//2:
        epsilon = epsilon_dacayer(epsilon)
    if(episode % display == 0):
        print("epsilon: {} in {} episode".format(epsilon, episode))

    # record aggrated rewards on each epsoide
    epsilon_rewards.append(epsilon_reward)

    # every display calculate average rewards
    if episode % display == 0:
        average_reward = sum(epsilon_rewards[-display:]) / len(epsilon_rewards[-display:])
        print("{} Success in {} episodes, rate: {} %".format(total_succ, display, 100*total_succ/display))
        print("Average Reward: {}".format(average_reward))
        total_succ = 0
        if episode==0:
            average_reward=-200
        # aggr_epsilon_rewards['ep'].append(episode)
        aggr_epsilon_rewards['SARSA'].append(average_reward)

time_end = time.time() 

print("Training SARSA Done in " + str(time_end-time_start) + " seconds.")

done = False
state = env.reset()
print("Testing SARSA...")
while not done:
    action = np.argmax(q_table[get_integrate_state(state)])
    next_state, _, done, _ = env.step(action)
    state = next_state
    # env.render()
print("Testing SARSA done.")


############## E-SARSA ##############
print("Start training using E-SARSA...")
q_table = np.zeros([q_table_size]*num_state+[env.action_space.n])
print(q_table.shape)
env.reset()
epsilon = 1

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
            action_expect = 0
            for i in range(env.action_space.n):
                weight = 0
                if(q_table[get_integrate_state(next_state)][i] == np.max(q_table[get_integrate_state(next_state)])):
                    weight = 1-epsilon+epsilon/env.action_space.n
                else:
                    weight = epsilon/env.action_space.n

                action_expect +=  weight * q_table[get_integrate_state(next_state)][i]
            td_target = reward + gamma * action_expect
            q_table[get_integrate_state(state)][action] += lr * (td_target - q_table[get_integrate_state(state)][action])

        elif next_state[0] >= 0.5:
            q_table[get_integrate_state(state)][action] = 0
            total_succ = total_succ + 1

        state = next_state

    # epsilon decay
    if episode < total_episode//2:
        epsilon = epsilon_dacayer(epsilon)
    if(episode % display == 0):
        print("epsilon: {} in {} episode".format(epsilon, episode))

    # record aggrated rewards on each epsoide
    epsilon_rewards.append(epsilon_reward)

    # every display calculate average rewards
    if episode % display == 0:
        
        average_reward = sum(epsilon_rewards[-display:]) / len(epsilon_rewards[-display:])
        print("{} Success in {} episodes, rate: {} %".format(total_succ, display, 100*total_succ/display))
        print("Average Reward: {}".format(average_reward))
        total_succ = 0
        if episode==0:
            average_reward=-200
        # aggr_epsilon_rewards['ep'].append(episode)
        aggr_epsilon_rewards['E-SARSA'].append(average_reward)

time_end = time.time() 

print("Training E-SARSA Done in " + str(time_end-time_start) + " seconds.")

done = False
state = env.reset()
print("Testing E-SARSA...")
while not done:
    action = np.argmax(q_table[get_integrate_state(state)])
    next_state, _, done, _ = env.step(action)
    state = next_state
    # env.render()
print("Testing E-SARSA done.")



######## DISPLAY #################

plt.plot(aggr_epsilon_rewards['ep'], aggr_epsilon_rewards['Q-Learning'], label = 'Q-Learning')
plt.plot(aggr_epsilon_rewards['ep'], aggr_epsilon_rewards['SARSA'], label = 'SARSA')
plt.plot(aggr_epsilon_rewards['ep'], aggr_epsilon_rewards['E-SARSA'], label = 'E-SARSA')
plt.legend()
plt.xlabel('total_episode')
plt.ylabel('Rewards')
plt.show()
env.close()
