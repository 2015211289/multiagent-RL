import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim, Tensor
from maddpg_impl.reward_shaping.config import Config


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

        self.fc1 = nn.Linear(self.obs_size, Config.embed_hidden_size)
        self.last = nn.Linear(Config.embed_hidden_size * 2, self.num_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=Config.embed_lr)

    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.last(x)
        return nn.Softmax(dim=1)(x)

    def embedding(self, x):
        # TODO 匹配train model数据格式
        # input = x[0]
        # for i in range(1, len(x)):
        #     input = np.append(input, x[i])
        x = F.relu(self.fc1(x))
        return x

    def train_model(self, obs_n, obs_next_n, act_n):

        length = len(obs_n[0])

        states = [np.empty(0) for _ in range(length)]
        for i in range(length):
            for arrays in obs_n:
                states[i] = np.append(states[i],arrays[i])

        states = torch.from_numpy(np.stack(states,axis=0)).float()

        next_states = [np.empty(0) for _ in range(length)]
        for i in range(length):
            for arrays in obs_next_n:
                next_states[i]=np.append(next_states[i],arrays[i])

        next_states = torch.from_numpy(np.stack(next_states,axis=0)).float()

        actions = [np.empty(0) for _ in range(length)]
        for i in range(length):
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
        # actions_one_hot = torch.squeeze(F.one_hot(actions,
        #                                           self.num_outputs)).float()
        loss = nn.MSELoss()(net_out, actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def compute_intrinsic_reward(episodic_memory,
                             current_c_state,
                             k=10,
                             kernel_cluster_distance=0.008,
                             kernel_epsilon=0.0001,
                             c=0.001,
                             sm=8):
    state_dist = [(c_state, torch.dist(c_state, current_c_state))
                  for c_state in episodic_memory]
    state_dist.sort(key=lambda x: x[1])
    state_dist = state_dist[:k]
    dist = [d[1].item() for d in state_dist]
    dist = np.array(dist)

    dist = np.max(dist - kernel_cluster_distance, 0)
    kernel = kernel_epsilon / (dist + kernel_epsilon)
    s = np.sqrt(np.sum(kernel)) + c

    # if(np.sum(kernel)<=0):
    #     print(kernel)

    if np.isnan(s) or s > sm:
        return 0
    return 1 / s