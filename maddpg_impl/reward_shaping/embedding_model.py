from tensorflow.python.ops.gen_math_ops import sqrt
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim, Tensor
from reward_shaping.config import Config
import heapq
from collections import deque
import math


class EmbeddingModel(nn.Module):
    def __init__(self, obs_size, num_outputs):
        super(EmbeddingModel, self).__init__()

        if(len(obs_size)==0):
            return 

        self.obs_size = 0
        for m in obs_size:
            self.obs_size += m[0]
        self.num_outputs = 0
        for m in num_outputs:
            self.num_outputs += m

        # episode instric reward network
        self.fc1 = nn.Linear(self.obs_size, Config.embed_hidden_size)
        self.fc2 = nn.Linear(Config.embed_hidden_size, Config.embed_hidden_size)
        self.last = nn.Linear(Config.embed_hidden_size * 2, self.num_outputs)

        # long term reward network
        self.long_fc = nn.Linear(self.obs_size,Config.embed_hidden_size)
        self.long_fc2 = nn.Linear(Config.embed_hidden_size,self.num_outputs)
        self.long_stable_fc = nn.Linear(self.obs_size,Config.embed_hidden_size)
        self.long_stable_fc2 = nn.Linear(Config.embed_hidden_size,self.num_outputs)

        # ave and err
        self.stats = RunningStats()
        self.k_th_mean = RunningStats()
        self.optimizer = optim.Adam(self.parameters(), lr=Config.embed_lr)


    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.last(x)
        return nn.Softmax(dim=1)(x)

    def embedding(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def long_embedding(self,x):
        x = F.relu(self.long_fc(x))
        x = F.relu(self.long_fc2(x))
        return x

    def long_stable_embedding(self,x):
        x = F.relu(self.long_stable_fc(x))
        x = F.relu(self.long_stable_fc2(x))
        return x

    def train_model(self, obs_n, obs_next_n, act_n):

        length = len(obs_n[0])

        states = [np.zeros(0) for i in range(length)]
        for i in range(length):
            # states[i] = np.append(np.empty(0),[arrays[i] for arrays in obs_n])
            for arrays in obs_n:
                states[i] = np.append(states[i],arrays[i])

        states = torch.from_numpy(np.stack(states,axis=0)).float()

        next_states = [np.zeros(0) for i in range(length)]
        for i in range(length):
            # next_states[i]=np.append(np.empty(0),[arrays[i] for arrays in obs_next_n])
            for arrays in obs_next_n:
                next_states[i]=np.append(next_states[i],arrays[i])

        next_states = torch.from_numpy(np.stack(next_states,axis=0)).float()

        actions = [ np.zeros(0) for i in range(length)]
        for i in range(length):
            # actions[i]=np.append(np.zeros(0),[arrays[i] for arrays in act_n])
            for arrays in act_n:
                actions[i]=np.append(actions[i],arrays[i])
        actions = torch.from_numpy(np.stack(actions,axis=0)).float()

        # batch_size = torch.stack(batch.state).size()[0]
        # # last 5 in sequence
        # states = torch.stack(batch.state).view(batch_size,
        #                                        Config.sequence_length,
        #                                        self.obs_size)[:, -5:, :]
        # next_states = torch.stack(batch.next_state).view(
        #     batch_size, Config.sequence_length, self.obs_size)[:, -5:, :]
        # actions = torch.stack(batch.action).view(batch_size,
        #                                          Config.sequence_length,
        #                                          -1).long()[:, -5:, :]

        self.optimizer.zero_grad()
        net_out = self.forward(states, next_states)
        # mask = (actions == actions.max(dim=1,keepdim=True)[0]).to(dtype=torch.float32)
        # actions_one_hot = torch.ones_like(actions)
        # actions_one_hot = torch.mul(mask,actions_one_hot)
        # actions_one_hot = torch.squeeze(F.one_hot((actions * 1000).to(torch.int64))).float()
        loss = nn.MSELoss()(net_out, actions)
        # loss.backward()

        long_loss = nn.MSELoss()(self.long_embedding(states),self.long_stable_embedding(states).detach())

        total_loss = loss+long_loss
        total_loss.backward()

        self.optimizer.step()
        return total_loss.item()


    def compute_intrinsic_reward(self,
                                episodic_memory,
                                current_c_state,
                                new_obs_tensor,
                                k=10,
                                kernel_cluster_distance=0.008,
                                kernel_epsilon=0.0001,
                                c=0.001,
                                sm=8):
        state_dist = [(torch.dist(c_state, current_c_state).item()) for c_state in episodic_memory]
        state_dist = heapq.nsmallest(k, state_dist)
        # heapq.heapify(state_dist)
        # state_dist = heapq.nlargest(k,state_dist)
        # state_dist = state_dist[:k]
        dist = np.array(state_dist)

        self.k_th_mean.push(np.max(dist))
        dist = dist / self.k_th_mean.mean()

        dist = np.max(dist - kernel_cluster_distance, 0)
        kernel = kernel_epsilon / (dist + kernel_epsilon)
        s = np.sqrt(np.sum(kernel)) + c

        if np.isnan(s) or s>sm :
            return 0

        # if np.isnan(s) or s > sm:
        #     s = sm

        long_loss = nn.MSELoss()(self.long_embedding(new_obs_tensor),self.long_stable_embedding(new_obs_tensor))

        self.stats.push(long_loss.item())

        alpha = 1+ (long_loss.item()-self.stats.mean())/self.stats.deviation()
        intrisic_reward = (1/s) * min(max(alpha,1),5)
        # self.lastReward = (1/s) - self.lastReward
    
        return intrisic_reward 



class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
    
    def clear(self):
        self.n = 0
        
    def push(self, x):
        self.n += 1
        
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0
    
    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 1

    def deviation(self):
        return math.sqrt(self.variance())