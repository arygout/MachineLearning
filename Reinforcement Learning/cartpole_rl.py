import numpy as np
import matplotlib.pyplot as plt
import gym

h = 1

def determine_action(observation, weights):
	weighted_sum = np.dot(observation, weights)
	action = 0 if weighted_sum < 0 else 1
	return action

def run_episode(environment, weights):  
	observation = environment.reset()
	total_reward = 0
	for step in range(200):
		action = determine_action(observation, weights)
		observation, reward, done, info = environment.step(action)
		total_reward += reward
		if done:
			break
	return total_reward

def render_episode(environment, weights):  
	observation = environment.reset()
	total_reward = 0
	for step in range(200):
		environment.render()
		action = determine_action(observation, weights)
		observation, reward, done, info = environment.step(action)
		total_reward += reward
		if done:
			break
	print("steps: ",step)
	return total_reward

def meanReward(environment,weights):
	total_reward = 0
	episodes = 100
	for i in range(episodes):
		total_reward += run_episode(environment,weights)
	total_reward = total_reward/episodes
	return total_reward

def rewardDerivative(environment,weights):
	normalReward = meanReward(environment,weights)
	weightDerivatives = np.zeros(4)
	for i in range(weights.shape[0]):
		weights[i] += h
		newReward = run_episode(environment,weights)
		weightDerivatives[i] = (newReward-normalReward)/h
		weights[i] -= h
	return weightDerivatives


environment = gym.make('CartPole-v0')
weights = np.linspace(-0.5,0.5,4) 
weightGain = 0.1


"""total_reward1 = 0
for i in range(200):
	total_reward1 += run_episode(environment,weights)

total_reward2 = 0
for i in range(200):
	total_reward2 += run_episode(environment,weights)

print("t1: {}, t2: {}, t1/t2: {}".format(total_reward1,total_reward2,total_reward1/total_reward2))"""

epochs = 100

plotRewards = np.zeros(epochs)
plotX = np.linspace(0,99,100)

print(weights)
for i in range(30):
	weightDerivatives = rewardDerivative(environment,weights)
	weights -= weightDerivatives*weightGain
	plotRewards[i] = meanReward(environment,weights)
	print("weights: ", weights, plotRewards[i])
	print("derivatives: ", weightDerivatives)
	#print()


render_episode(environment,weights)
#lt.plot(plotX,plotRewards,"g-")
#plt.show()

