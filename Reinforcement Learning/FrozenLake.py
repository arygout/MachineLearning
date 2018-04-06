import gym
import numpy as np
import matplotlib.pyplot as plt

def convertObservation(observation):
    curRow = int(observation/4)
    curCol = observation%4
    return (curRow,curCol)

def determineAction(observation, state):
    """
        Returns 0 (go left) or 1 (go right)
        depending on whether the weighted sum of weights * observations > 0
    """
    #make sure previous observation gets set even if function returns early
    c = convertObservation(observation)
    p = convertObservation(state["prevStep"][0])

    cs = np.cumsum(state[c,:])
    v = np.random.uniform(0, cs[cs.shape[0]-1]) #random value from 0 to sum of all action weights for a given row and column
    for i in range(cs.shape[0]):
        if v < cs[i]:
            state["prevStep"] = (observation,i)
            return i

def updateQ(observation,reward,done):
    discount = 0.99
    learningRate = 0.1
    c = convertObservation(observation)
    p = convertObservation(state["prevStep"][0])
    oldQ = state["q"][p,state["prevStep"][1]]
    maxQ = np.max(state["q"][c,:])
    if done:
        state["q"][p,state["prevStep"][1]] = oldQ * (1 - learningRate) + learningRate * reward
        return
    state["q"][p,state["prevStep"][1]] = oldQ * (1 - learningRate) + learningRate * (reward + discount * maxQ)

def initEpisode(observation):
    state["prevStep"][0] = observation

def initState(state):
    state["q"] = np.ones((4,4,4))*0.1
    state["prevStep"] = ""

def runEpisode(env, state):
    """
    	This function runs one episode with a given set of state
    	and returns the total reward with those weights
    """
    observation = env.reset()
    initialization(observation)
    for step in range(200):
        environment.render()
        action = determineAction(observation, state)
        observation, reward, done, info = env.step(action)
        updateQ(observation,reward,done)
        if done:
            break

def runEpisode(env, state):
    """
    	This function runs one episode with a given set of state
    	and returns the total reward with those weights
    """
    observation = env.reset()
    initialization(observation)
    for step in range(200):
        action = determineAction(observation, state)
        observation, reward, done, info = env.step(action)
        updateQ(observation,reward,done)
        if done:
            break



def main():
    state = {}
    initState(state)
    env = gym.make('FrozenLake-v0')
    env.reset()
    for i in range(200):
        runEpisode(env,state)
    render_episode()

if __name__ == '__main__':
    main()
