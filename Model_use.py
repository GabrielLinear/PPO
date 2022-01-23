import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment
import collections
import torch.optim as optim
import time

class Policy(nn.Module):

    def __init__(self,input_size,nb_action):
        super(Policy, self).__init__()
        self.nb_action = nb_action
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(input_size,200)
        self.fc2 = nn.Linear(200,75)
        self.fc3 = nn.Linear(75,nb_action)
        self.fc3bis = nn.Linear(75,nb_action)
        
        
    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.fc3(x)) # Tanh because action_values between -1 and 1.
        #sigma = F.softplus(self.fc3bis(x))# Activation to stay always >= 0
        #sigma = torch.clamp(sigma,0.001) # Activation to stay always > 0
        #writer.add_histogram("Moyenne",mu,e)
        #writer.add_histogram("Variance",sigma,e)
        sigma = torch.ones(self.nb_action,requires_grad=False).to(self.device)/2 
        m = torch.distributions.normal.Normal(mu,sigma,False)# False, whereas constraint on mu = 0
        return m
    

def collect_trajectories(env,env_info,policy,device):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    state = env_info.vector_observations # get the current state (for each agent)
    states_tab , action_tab, reward_tab, prob_tab = [],[],[], []
    while True:
        state = torch.from_numpy(state).to(device)
        policy.eval()
        with torch.no_grad(): # Everything with torch no grad.
            m = policy(state) 

        
            # Sample maybe on gradient as to check that
            sample = m.sample()
            action_tab.append(sample) # No clip and store

            # Proba not on clip and detach from Gradient.
            proba = m.log_prob(sample)
            #proba = torch.exp(proba) #Proba on CUDA no detach
            
            # Interact with the environment 
            sample = torch.clip(sample.detach().cpu(), -1, 1) # CLIP BEFORE TAKING THE PROBA OR AFTER?
            sample = sample.numpy()


            # Step the environment
            env_info = env.step(sample)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            # Store values
            prob_tab.append(proba)
            reward_tab.append(np.asarray(rewards))
            states_tab.append(state)

            # BREAK IF END OF THE EPISODE
            if np.any(dones):                                  # exit loop if episode finished
                break
            state = next_states
            time.sleep(0.1)
    return states_tab, action_tab, reward_tab,prob_tab


######################## MAIN USE #################################
env = UnityEnvironment(file_name="C:/Users/gabyc/Desktop/Reinforcment_TP/deep-reinforcement-learning/p2_continuous-control/Multi_agent/Reacher_Windows_x86_64/Reacher.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]  
states = env_info.vector_observations # get the current state (for each agent
num_agents = len(states)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
nb_states = len(states[0])
action_size = brain.vector_action_space_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy = Policy(nb_states,action_size).to(device) # Policy network
optimizer = optim.Adam(policy.parameters(), lr=2e-4)
policy.load_state_dict(torch.load("Model_checkpoint/PPO_actor_stable.pth"))

episode = 2
for e in range(episode):
        states, actions, rewards,prob = collect_trajectories(env,env_info, policy,device)
    total_rewards = np.mean(np.sum(rewards,axis=0))
    print(total_rewards)
