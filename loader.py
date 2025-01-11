import sys
import numpy as np
import gym
import csv
from random import random
import ast



if __name__=="__main__":

    with open('policy.csv', 'r') as f:
        header=f.readline()
        rows=[row.split('"')for row in f.readlines()]
        policy={ast.literal_eval(row[1]):int(row[2].strip(',')) for row in rows}
    
    env = gym.make("MsPacmanDeterministic-v4", obs_type="grayscale", frameskip=8, render_mode="human")
    obs=env.reset()

    s=tuple(obs.flatten()) 
    
    total_reward=0
    done=False
    
    while not done:
        if s in policy:
            action = policy[s]    
            newObs, reward, done, _,  = env.step(action)
            newObs= tuple(newObs.flatten())
        else:
            done=True
        s=newObs
        total_reward += reward
        if done:
            break
