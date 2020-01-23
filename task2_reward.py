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

adjusted_reward = True # 更改环境所给的Reward

env.reset()
# 创建Q表
action_num = 4
action_stride = 2/action_num
action_space = np.zeros((action_num, 1))
# action_space
# action_space = np.array(range(-1.0, 1.0, action_stride))
for i in range(action_num):
    action_space[i] = -1.0+i*action_stride+0.5*action_stride
# action_space = action_space.reshape(len(action_space), 1)


num_state = len(env.observation_space.low)
stride = (env.observation_space.high - env.observation_space.low) / q_table_size
# q_table = np.zeros([q_table_size]*num_state+[len(action_space)])
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
displayer = {'ep':[],'Reward':[],'SuccessRate':[]}


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
        displayer['Reward'].append(average_reward)
        displayer['SuccessRate'].append(total_succ/display)
        total_succ = 0
   
time_end = time.time()  
plt.figure()   
plt.plot(displayer['ep'], displayer['Reward'], label = 'Reward')
plt.legend()
plt.xlabel('total_episode')
plt.ylabel('Rewards')
plt.show()

plt.figure()
plt.plot(displayer['ep'], displayer['SuccessRate'], label = 'Success Rate')
plt.legend()
plt.xlabel('total_episode')
plt.ylabel('Success Ratio')
plt.show()

print("Training Q Learning Done in " + str(time_end-time_start) + " seconds.")

done = False
state = env.reset()
print("Testing Q Learning...")
acc_reward = 0
while not done:
    action_idx = np.argmax(q_table[get_integrate_state(state)])
    next_state, reward, done, _ = env.step(action_space[action_idx])
    acc_reward+=reward
    state = next_state
    env.render()
print("Final Score: "+ str(acc_reward))
print("Testing Q Learning done.")

env.close()
