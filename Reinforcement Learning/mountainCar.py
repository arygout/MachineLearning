import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
bestWeights = np.zeros(2)
bestReward = 0;
for i in range(1000):
    observation = env.reset()
    weights = np.random.normal(0,5,2)
    print(weights)
    for t in range(100):
        #env.render()
        action = int(np.round((np.tanh(observation[0]*weights[0]+observation[1]*weights[1])+1)/2))
        print(np.abs(observation[0]))
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
        if(np.abs(observation[0]) > np.abs(bestReward)):
            bestReward = observation[0]
            bestWeights = weights

print("bestWeights", bestWeights)
print("bestReward", bestReward)
observation = env.reset()
for t in range(100):
    env.render()
    action = int(np.round((np.tanh(observation[0]*bestWeights[0]+observation[1]*bestWeights[1])+1)/2))
    print(observation,action)
    observation, reward, done, info = env.step(action)
