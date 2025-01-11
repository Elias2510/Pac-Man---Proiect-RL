import sys
import numpy as np
import gym
from random import random

sys.stderr = open("log", "w", buffering=1)


env = gym.make("MsPacmanDeterministic-v4", obs_type="grayscale", frameskip=8, render_mode="human")
initial_state=current_state = tuple(env.reset().flatten())
gamma=0.99

known_paths={current_state:0}
known_paths[current_state]={}
known_paths[current_state][1]={"reward":0, "next_state":0}

decay=0.99

def policy_improvement (states , nA , value_function, gamma=0.9):
    policy={}
    for s in states:
        value_per_action = np.zeros(shape=(nA,))
        for action in known_paths[s]:
                
            action_val =  known_paths[s][action]['reward'] + gamma * value_function[known_paths[s][action]["next_state"]]
            value_per_action[action] = action_val
        
        best_action = np.argmax(value_per_action)
        policy[s] = best_action
    
    return policy


def value_iteration(states, nA, epsilon, gamma=0.9, tol=15):
    
    numIters = 0

    maxChange = np.inf
    
    done=False
    value_function={}

    total_reward=0
    
    rand=0
    think=0
    obs=env.reset()
    s=tuple(obs.flatten())
    while not done:

        if s not in best_policy or random()<epsilon:
            a = env.action_space.sample()
            rand+=1
            #print("A mutat random")
            while not a:
                a = env.action_space.sample()
        else:
            #print("A gandit")
            think+=1
            a=best_policy[s]

        next_state, reward, done, _  = env.step(a)
        next_state=tuple(next_state.flatten())
        total_reward+=reward
        known_paths[s][a]={"reward":reward,"next_state":next_state}

        if next_state not in known_paths:
            known_paths[next_state]={}
            known_paths[next_state][0]={"reward":0,"next_state":0}
        
        s=next_state
        if done:      
            break

    print(f"Scorul dupa explorare {total_reward}")
    print(f'Miscari gandite {think}\nMiscari aleatoare {rand}')

    while maxChange > tol:
        numIters += 1
        maxChange = 0.0
        for s in states:
            bestActionValue = -np.inf
            for action in known_paths[s]:
                 
                if known_paths[s][action]['next_state'] not in value_function:
                    value_function[known_paths[s][action]['next_state']]=0
                
                value_for_thisAction = 0.0

                if known_paths[s][action]['next_state'] ==0:      
                    value_for_thisAction =  known_paths[s][action]['reward'] 
                else:
                    value_for_thisAction = known_paths[s][action]['reward'] + gamma * value_function[known_paths[s][action]['next_state']]
                    
                if value_for_thisAction > bestActionValue:
                    bestActionValue = value_for_thisAction
            
            if s in value_function:
                maxChange = max(maxChange, abs(value_function[s] - bestActionValue))

            value_function[s] = bestActionValue

        
    print(f"Value iteration converged after {numIters} steps\n")
    policy = policy_improvement(states, nA, value_function, gamma)
    return value_function, policy


def runEpisode(env, policy):
     
    total_reward = 0
    print("Joaca ce a invatat")
    done=False
    obs=env.reset()
    s=tuple(obs.flatten()) 

    for _ in range (100):
        
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
    
        print(f"Episode reward: {total_reward}")
        
        s=env.reset()
        s=tuple(s.flatten())
        done=False

    total_reward/=100
    print(f"Scorul mediu al politicii {total_reward}")    
   
   

if __name__=="__main__":

    epsilon=1
    best_policy={}
    for i in range(10):
        print(f"##############Episode: {i}###############\nEpsilon: {epsilon}\n")
        states=list(known_paths.keys())
        best_value, best_policy = value_iteration(states, env.action_space.n, epsilon, gamma=gamma, tol=10e-3)
        epsilon =min(0.10,epsilon*decay)
        print(f"Stie {len(best_policy.keys())} mutari\n")

    runEpisode(env, policy=best_policy)