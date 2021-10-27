import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim, Tensor
from reward_shaping.config import Config
import heapq
from collections import deque


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
        self.fc2 = nn.Linear(Config.embed_hidden_size, Config.embed_hidden_size//2)
        self.last = nn.Linear(Config.embed_hidden_size, self.num_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=Config.embed_lr)

        self.lastReward = 0

        # long term reward network
        self.long_fc = nn.Linear(self.obs_size,Config.embed_hidden_size)
        self.long_fc2 = nn.Linear(Config.embed_hidden_size,Config.embed_hidden_size//2)
        self.long_stable_fc = nn.Linear(self.obs_size,Config.embed_hidden_size)
        self.long_stable_fc2 = nn.Linear(Config.embed_hidden_size,Config.embed_hidden_size//2)
        self.history_apha = deque(maxlen=int(1e6))

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
        mask = (actions == actions.max(dim=1,keepdim=True)[0]).to(dtype=torch.float32)
        actions_one_hot = torch.ones_like(actions)
        actions_one_hot = torch.mul(mask,actions_one_hot)
        # actions_one_hot = torch.squeeze(F.one_hot((actions * 1000).to(torch.int64))).float()
        loss = nn.MSELoss()(net_out, actions_one_hot)
        loss.backward()

        long_loss = nn.MSELoss()(self.long_embedding(states),self.long_stable_embedding(states).detach())
        long_loss.backward()

        self.optimizer.step()
        return loss.item()+long_loss.item()


    def compute_intrinsic_reward(self,
                                episodic_memory,
                                current_c_state,
                                new_obs_tensor,
                                k=50,
                                kernel_cluster_distance=0.008,
                                kernel_epsilon=0.0001,
                                c=0.001,
                                sm=8):
        state_dist = [(torch.dist(c_state, current_c_state).item())
                    for c_state in episodic_memory]
        # heapq.heapify(state_dist)
        state_dist = heapq.nlargest(k,state_dist)
        # state_dist = state_dist[:k]
        # dist = [d[1].item() for d in state_dist]
        dist = np.array(state_dist)

        dist = dist**2 / np.mean(dist)**2

        dist = np.max(dist - kernel_cluster_distance, 0)
        kernel = kernel_epsilon / (dist + kernel_epsilon)
        s = np.sqrt(np.sum(kernel)) + c

        # if(np.sum(kernel)<=0):
        #     print(kernel)

        if np.isnan(s) or s > sm:
            return 0

        long_loss = nn.MSELoss()(self.long_embedding(new_obs_tensor),self.long_stable_embedding(new_obs_tensor))
        self.history_apha.append(long_loss.item())
        apha = 1+ (long_loss.item()-np.mean(self.history_apha))/np.std(self.history_apha,ddof=1)
        intrisic_reward = (1/s) * min(max(apha,1),5)
        # self.lastReward = (1/s) - self.lastReward
        if np.isnan(intrisic_reward):
            return 0

        return intrisic_reward