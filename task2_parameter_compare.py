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
displayer = {'ep':[],'Reward0':[],'SuccessRate0':[],'Reward1':[],'SuccessRate1':[],'Reward2':[],'SuccessRate2':[]}

for i in range(3):
    env.reset()
    # 创建Q表
    action_nums = [4, 8, 12]
    action_num = action_nums[0]
    alphas = [0.2, 0.5, 0.8]
    action_stride = 2/action_num
    action_space = np.zeros((action_num, 1))
    epsilon=1
    for j in range(action_num):
        action_space[j] = -1.0+j*action_stride+0.5*action_stride

    num_state = len(env.observation_space.low)
    stride = (env.observation_space.high - env.observation_space.low) / q_table_size

    q_table = np.zeros([q_table_size]*num_state+[action_num])

    # if i==0:
    #     q_table = np.zeros([q_table_size]*num_state+[action_num])
    # else:
    #     q_table = np.empty([q_table_size]*num_state+[action_num])

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
            action_idx = epsilon_greedy_action(state, epsilon)
            # receive info from environment
            next_state, reward, done, _ = env.step(action_space[action_idx])
            epsilon_reward += reward

            if adjusted_reward:
                if state[1] > 0:
                    reward += (next_state[0])*alphas[i]
                
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
            if episode ==0:
                average_reward=-30
            print("{} Success in {} episodes, rate: {} %".format(total_succ, display, 100*total_succ/display))
            print("Average Reward: {}".format(average_reward))
            if(i==0):
                displayer['ep'].append(episode)
            displayer['Reward'+str(i)].append(average_reward)
            displayer['SuccessRate'+str(i)].append(total_succ/display)
            total_succ = 0
    
    time_end = time.time() 

print("Training Q Learning Done in " + str(time_end-time_start) + " seconds.")

plt.figure()   
plt.plot(displayer['ep'], displayer['Reward0'], label = 'alpha:0.2')
plt.plot(displayer['ep'], displayer['Reward1'], label = 'alpha:0.5')
plt.plot(displayer['ep'], displayer['Reward2'], label = 'alpha:0.8')
plt.legend()
plt.xlabel('total_episode')
plt.ylabel('Rewards')
plt.show()

plt.figure()
plt.plot(displayer['ep'], displayer['SuccessRate0'], label = 'alpha:0.2')
plt.plot(displayer['ep'], displayer['SuccessRate1'], label = 'alpha:0.5')
plt.plot(displayer['ep'], displayer['SuccessRate2'], label = 'alpha:0.8')
plt.legend()
plt.xlabel('total_episode')
plt.ylabel('Success Ratio')
plt.show()



env.close()
# done = False
# state = env.reset()
# print("Testing Q Learning...")
# acc_reward = 0
# while not done:
#     action_idx = np.argmax(q_table[get_integrate_state(state)])
#     next_state, reward, done, _ = env.step(action_space[action_idx])
#     acc_reward+=reward
#     state = next_state
#     env.render()
# print("Final Score: "+ str(acc_reward))
# print("Testing Q Learning done.")

