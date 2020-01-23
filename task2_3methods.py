##### 实现了Q_Learning SARSA E-SARSA 三种方法
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pdb import set_trace as st
import time
env = gym.make("MountainCarContinuous-v0")
env.reset()

lr = 0.55
gamma = 1
total_episode = 7000
display = 500
q_table_size = 20
epsilon = 1

index = 2 # 1---QLearning  2---SARSA  3---E-SARSA
adjusted_reward = True # 更改环境所给的Reward

env.reset()
# 创建Q表
action_num = 4
action_stride = 2/action_num
action_space = np.zeros((action_num, 1))
for i in range(action_num):
    action_space[i] = -1.0+i*action_stride+0.5*action_stride

num_state = len(env.observation_space.low)
stride = (env.observation_space.high - env.observation_space.low) / q_table_size
q_table = np.zeros([q_table_size]*num_state+[action_num])

print(q_table.shape)

def epsilon_dacayer(epsilon):
    delta_epsilon = 1/(total_episode//2)
    return epsilon-delta_epsilon

def get_integrate_state(state):
    integrate_state = (state - env.observation_space.low) // stride
    return tuple(integrate_state.astype(int))

def epsilon_greedy_action(state, epsilon):
    integrate_state = get_integrate_state(state)
    if np.random.random() < epsilon:
        action_idx = np.random.randint(0,len(action_space))
    else:
        action_idx = np.argmax(q_table[integrate_state])
    return action_idx

epsilon_rewards = []
displayer = {'ep':[],'Q-Learning1':[],'SARSA1':[],'E-SARSA1':[], 'Q-Learning2':[],'SARSA2':[],'E-SARSA2':[]}

if index==1:
    ################# Q_LEARNING ################
    print("Start training using Q-Learning...")
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
            action_idx = epsilon_greedy_action(state, epsilon)
            # receive info from environment
            next_state, reward, done, _ = env.step(action_space[action_idx])
            epsilon_reward += reward

            if adjusted_reward:
                if state[1] > 0:
                    reward += (next_state[0])*0.5
                
            if not done:
                # update q_table
                td_target = reward + gamma * np.max(q_table[get_integrate_state(next_state)])
                q_table[get_integrate_state(state)][action_idx] += lr * (td_target - q_table[get_integrate_state(state)][action_idx])

            elif next_state[0] >= 0.5:
                q_table[get_integrate_state(state)][action_idx] = 0
            if next_state[0] >= 0.45:
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

            displayer['ep'].append(episode)
            displayer['Q-Learning1'].append(average_reward)
            displayer['Q-Learning2'].append(total_succ/display)
            total_succ = 0
elif index==2:
    ############# SARSA ################
    print("Start training using SARSA...")
    time_start = time.time()
    q_table = np.zeros([q_table_size]*num_state+[action_num])

    print(q_table.shape)
    env.reset()
    epsilon = 1
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
            action_idx = epsilon_greedy_action(state, epsilon)
            # receive info from environment
            next_state, reward, done, _ = env.step(action_space[action_idx])
            epsilon_reward += reward
            next_action_idx = epsilon_greedy_action(next_state, epsilon)

            if adjusted_reward:
                if state[1] > 0:
                    reward += (next_state[0])*0.5
                
            if not done:
                # update q_table
                td_target = reward + gamma * np.max(q_table[get_integrate_state(next_state)])
                q_table[get_integrate_state(state)][action_idx] += lr * (td_target - q_table[get_integrate_state(state)][action_idx])

            elif next_state[0] >= 0.5:
                q_table[get_integrate_state(state)][action_idx] = 0
            if next_state[0] >= 0.45:
                total_succ = total_succ + 1

            state = next_state
            # action_space[action_idx] = action_space[next_action_idx]
            action_idx = next_action_idx

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

            displayer['ep'].append(episode)
            displayer['SARSA1'].append(average_reward)
            displayer['SARSA2'].append(total_succ/display)
            total_succ = 0
    
    time_end = time.time()

    print("Training SARSA in " + str(time_end-time_start) + " seconds.")

    done = False
    state = env.reset()
    print("Testing SARSA...")
    acc_reward = 0
    while not done:
        action_idx = np.argmax(q_table[get_integrate_state(state)])
        next_state, reward, done, _ = env.step(action_space[action_idx])
        acc_reward+=reward
        state = next_state
        env.render()
    print("Final Score: "+ str(acc_reward))
    print("Testing SARSA done.")
    

    print("Training SARSA Done in " + str(time_end-time_start) + " seconds.")

    done = False
    state = env.reset()
    print("Testing SARSA...")
    acc_reward = 0
    while not done:
        action_idx = np.argmax(q_table[get_integrate_state(state)])
        next_state, reward, done, _ = env.step(action_space[action_idx])
        acc_reward+=reward
        state = next_state
        env.render()
    print("Final Score: "+ str(acc_reward))
    print("Testing SARSA done.")
elif index==3:
    ############# E-SARSA ################
    print("Start training using E-SARSA...")
    time_start = time.time()
    q_table = np.zeros([q_table_size]*num_state+[action_num])

    print(q_table.shape)
    env.reset()
    epsilon = 1
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
            action_idx = epsilon_greedy_action(state, epsilon)
            # receive info from environment
            next_state, reward, done, _ = env.step(action_space[action_idx])
            epsilon_reward += reward

            if adjusted_reward:
                if state[1] > 0:
                    reward += (next_state[0])*0.5
                
            if not done:
                # update q_table
                action_expect = 0
                for i in range(action_num):
                    weight = 0
                    if(q_table[get_integrate_state(next_state)][i] == np.max(q_table[get_integrate_state(next_state)])):
                        weight = 1-epsilon+epsilon/action_num
                    else:
                        weight = epsilon/action_num

                    action_expect +=  weight * q_table[get_integrate_state(next_state)][i]
                
                td_target = reward + gamma * np.max(q_table[get_integrate_state(next_state)])
                q_table[get_integrate_state(state)][action_idx] += lr * (td_target - q_table[get_integrate_state(state)][action_idx])

            elif next_state[0] >= 0.5:
                q_table[get_integrate_state(state)][action_idx] = 0
            if next_state[0] >= 0.45:
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

            displayer['ep'].append(episode)
            displayer['E-SARSA1'].append(average_reward)
            displayer['E-SARSA2'].append(total_succ/display)
            total_succ = 0
    
    
    time_end = time.time()
    print("Training E-SARSA in " + str(time_end-time_start) + " seconds.")

    done = False
    state = env.reset()
    print("Testing E-SARSA...")
    acc_reward = 0
    while not done:
        action_idx = np.argmax(q_table[get_integrate_state(state)])
        next_state, reward, done, _ = env.step(action_space[action_idx])
        acc_reward+=reward
        state = next_state
        env.render()
    print("Final Score: "+ str(acc_reward))
    print("Testing E-SARSA done.")
    

    print("Training E-SARSA Done in " + str(time_end-time_start) + " seconds.")

    done = False
    state = env.reset()
    print("Testing E-SARSA...")
    acc_reward = 0
    while not done:
        action_idx = np.argmax(q_table[get_integrate_state(state)])
        next_state, reward, done, _ = env.step(action_space[action_idx])
        acc_reward+=reward
        state = next_state
        env.render()
    print("Final Score: "+ str(acc_reward))
    print("Testing E-SARSA done.")

######## DISPLAY #################

time_end = time.time()  
plt.figure()  
if index==1:
    plt.plot(displayer['ep'], displayer['Q-Learning1'], label = 'Q-Learning')
elif index==2:
    plt.plot(displayer['ep'], displayer['SARSA1'], label = 'SARSA')
elif index==3:
    plt.plot(displayer['ep'], displayer['E-SARSA1'], label = 'E-SARSA')
 
# plt.plot(displayer['ep'], displayer['Reward'], label = 'Reward')
plt.legend()
plt.xlabel('total_episode')
plt.ylabel('Rewards')
plt.show()

plt.figure()
if index==1:
    plt.plot(displayer['ep'], displayer['Q-Learning2'], label = 'Q-Learning')
elif index==2:
    plt.plot(displayer['ep'], displayer['SARSA2'], label = 'SARSA')
elif index==3:
    plt.plot(displayer['ep'], displayer['E-SARSA2'], label = 'E-SARSA')
# plt.plot(displayer['ep'], displayer['SuccessRate'], label = 'Success Rate')
plt.legend()
plt.xlabel('total_episode')
plt.ylabel('Success Ratio')
plt.show()

env.close()
