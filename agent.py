import gfootball.env as football_env
import collections
import cv2
from gfootball.env import football_action_set
from gfootball.env import observation_preprocessing
import gym
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random

class DQN(torch.nn.Module):
    def __init__(self, input , hidden, output) -> None:
        super(DQN, self).__init__()

        self.linear1 = torch.nn.Linear(input, hidden)
        self.linear2 = torch.nn.Linear(hidden, hidden)
        self.linear3 = torch.nn.Linear(hidden, hidden*2)
        self.linear4 = torch.nn.Linear(hidden*2, hidden*2)
        self.linear5 = torch.nn.Linear(hidden*2, hidden*2)
        self.linear6 = torch.nn.Linear(hidden*2, hidden)
        self.linear7 = torch.nn.Linear(hidden, hidden)
        self.final = torch.nn.Linear(hidden, output)

    def forward(self, state):
        # state of dim (batch, input_dim)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        x = F.relu(self.linear7(x))
        x = self.final(x)

        return x   # (batch, output)


class Agent():
    def __init__(self,DEVICE, pretrained=None) -> None:
        self.pretrained = pretrained
        self.DEVICE =  DEVICE
        if pretrained:
            self.dqn = torch.load('/home/darthbaba/code/football/agent/saved_model/agent'+self.pretrained+'.pth').to(DEVICE)
            self.target = torch.load('/home/darthbaba/code/football/agent/saved_model/target'+self.pretrained+'.pth').to(DEVICE)
            self.dqn.eval()
            self.target.eval()

        self.dqn = DQN(115, 256, 19).to(DEVICE)
        self.target = DQN(115,256, 19).to(DEVICE)
        self.target.load_state_dict(self.dqn.state_dict())
        self.gamma = 0.99
        self.batch_size = 64
        self.eps = 1.0
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=0.001)
        self.replay_memory_buffer = deque(maxlen=10000)
        
    def select_action(self, state):
        p = np.random.random()
        if np.random.uniform() < self.eps:
            action = np.random.choice(19)
        else:
            temp = torch.from_numpy(state).float().unsqueeze(0)
            action_temp = self.dqn(temp.to(self.DEVICE))
            max_q, action = torch.max(action_temp, 1)
            action = int(action.cpu().numpy())
        return action

    def append_replay_buffer(self, s1, a1, r, s2, done):
        self.replay_memory_buffer.append((s1, a1, r, s2, done))

    def update_epsilon(self):
        # Decay epsilon
        if self.eps >= 0.01:
            self.eps *= 0.95

    def save_agent_and_target(self,num):
        torch.save(self.dqn, '/home/darthbaba/code/football/agent/saved_model/agent'+num+'.pth')
        torch.save(self.target, '/home/darthbaba/code/football/agent/saved_model/target'+num+'.pth')

    # TRAIN LOOP
    def train(self, s1, a1, r, s2, done, n):
        self.append_replay_buffer(s1, a1, r, s2, done)

        if n % 10 == 0:
            self.update_epsilon()
            self.target.load_state_dict(self.dqn.state_dict())

        if len(self.replay_memory_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_memory_buffer, self.batch_size)
        state_batch = torch.from_numpy(np.vstack([i[0] for i in minibatch])).float()
        action_batch = torch.from_numpy(np.vstack([i[1] for i in minibatch])).int()
        reward_batch = torch.from_numpy(np.vstack([i[2] for i in minibatch])).float()
        next_state_batch = torch.from_numpy(np.vstack([i[3] for i in minibatch])).float()
        done_list = torch.from_numpy(np.vstack([i[4] for i in minibatch]).astype(np.uint8)).float()

        next_state_values = self.target(next_state_batch.to(self.DEVICE)).max(1)[0]
        t = self.gamma * next_state_values.detach().cpu().unsqueeze(1) * (1-done_list)
        target_values = torch.add(reward_batch,t).to(self.DEVICE)
        Q_values = self.dqn(state_batch.to(self.DEVICE))
        state_action_values = Q_values.gather(1, action_batch.type(torch.int64).to(self.DEVICE))
        loss = torch.nn.MSELoss()(state_action_values, target_values)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

agent = Agent(DEVICE,'1')
env = football_env.create_environment(env_name="11_vs_11_easy_stochastic", stacked=False, representation='simple115v2',rewards="scoring,checkpoints", logdir='/tmp/football', write_goal_dumps=False, 
                                    write_full_episode_dumps=False, render=False)

for i in range(20):
    s = env.reset()
    steps = 0
    r =  0
    while True:
        a = agent.select_action(s)
        obs, rew, done, info = env.step(a)
        steps += 1
        r += rew
        agent.train(s,a,rew,obs, done, steps)
        s = obs


        if steps % 100 == 0:
            print("Step %d Reward: %f" % (steps, r))

        if done:
            # agent.save_agent_and_target('1')
            break

    print("GAME: %d Steps: %d Reward: %.2f" % (i+1, steps, r))
agent.save_agent_and_target('1')