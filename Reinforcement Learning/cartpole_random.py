import gym
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make('CartPole-v0')
environment.reset()

def determine_action(observation, weights):
    """
        Returns 0 (go left) or 1 (go right)
        depending on whether the weighted sum of weights * observations > 0
    """
    weighted_sum = np.dot(observation, weights)
    action = 0 if weighted_sum < 0 else 1
    return action


def run_episode(environment, weights):
    """
    	This function runs one episode with a given set of weights
    	and returns the total reward with those weights
    """
    observation = environment.reset()
    total_reward = 0
    for step in range(200):
        action = determine_action(observation, weights)
        observation, reward, done, info = environment.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def find_best_weights(num_episodes):
    """
        This function runs a number of episodes and picks random weights for each one and evaluates
        the reward given by the weights.
        It returns the weights for the best episode.
    """
    best_weights = None
    best_reward = 0
    observation = environment.reset()
    for episode in range(num_episodes):
        weights = np.random.rand(4) * 2 - 1
        reward = run_episode(environment, weights)
        if reward > best_reward:
            best_weights = weights
            best_reward = reward
            #print("Current Best Weights at episode #{} are {}".format(episode, best_weights))
    return best_weights, best_reward

#Same as run_episode but renders as well for debugging purposes
def render_episode(environment, weights):
    observation = environment.reset()
    total_reward = 0
    for step in range(200):
        environment.render()
        action = determine_action(observation, weights)
        #print(type(observation))
        print(str(observation[0]) + ", " + str(observation[1]) + ", " + str(observation[2]) + ", " + str(observation[3]))
        observation, reward, done, info = environment.step(action)
        total_reward += reward
        if done:
            break
    print("steps: ",step)
    return total_reward

#Runs a certain number of episodes with the given environment and weights and returns the mean reward of all episodes
def meanReward(environment,weights):
    total_reward = 0
    episodes = 5
    for i in range(episodes):
        total_reward += run_episode(environment,weights)

    return total_reward/episodes

#number of times hill climbing is tested
numTrials = 1

plotClimbs = np.zeros(numTrials)#Stores number of hill climbs needed to reach the maximum reward

for t in range(numTrials):
    best_weights = find_best_weights(4)[0]#Does a random search to find a good starting weight with 4 episodes
    #print("starting reward is: {}".format(run_episode(environment,best_weights)))
    for i in range(1): #Runs up to 300 hill climbs
        noise = np.random.randn(4)/10#Randomly creates a 4 length noise array with a standard deviation of 10%

        reward = run_episode(environment,best_weights)#Calculates reward of current weights
        if reward < run_episode(environment,best_weights*noise):#Compares reward of current weights and reward of noisy weights
            best_weights = best_weights*noise#adjusts best_weights if the noisy weights produce a better reward


        #Stops hill climbing if it gets a reward of 200 5 times in a row with the current weights.
        #Only used to create histogram. Doesn't actually use this to train the weights.
        #If training a more complex program this would be taken out because the meanReward method runs 5 episodes and would probably slow down the program.
        if(meanReward(environment,best_weights)) == 200:
            break
    plotClimbs[t] = i#Stores number of hill climbs needed to reach a reward of 200 for this trial
    print("trial: {}, hill climbs: {}".format(t,i))
print("average number of hill climbs to reach a reward of 200: {}".format(np.mean(plotClimbs)))

#Creates a histogram of number of hill climbs needed to reach a reward of 200
#In the majority of cases this number is less than 50 meaning that it usually will only require 100 episodes to arrive at satisfactory weights
#2 episodes per hill climb. One for noisy weights, and one for current weights.
#plt.hist(plotClimbs,bins = 100)
#plt.show()
render_episode(environment,best_weights)
