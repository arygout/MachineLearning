import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.reset()

def determineAction(observation, Q, bounds, shouldRender):
    qI = np.array([])
    for i in range(len(bounds)):
        qI.push(np.digitize([observation[i]],bounds[i]))

    curQ = Q[qI[0]][qI[1]][qI[2]][qI[3]]


    if shouldRender:
        return np.argmax(curQ)
    else:
        cs = np.cumsum(curQ)
        rVal = np.random.uniform(0,cs[-1])
        return np.digitize([rVal],cs)[0]

def runEpisode(observation, Q, bounds, shouldRender, env):

    observation = env.reset()
    prevObservation = observation
    total_reward = 0
    for step in range(200):
        action = determine_action(observation, Q, bounds, shouldRender)
        prevObservation = observation
        observation, reward, done, info = env.step(action)
        total_reward += reward
                if shouldRender:
                    env.render()
        if done:
            break
    return total_reward

def updateQ(observation, prevObservation, Q, action, reward, done):





def main():

if __name__ == '__main__':
    main()
