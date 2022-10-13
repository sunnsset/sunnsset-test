import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from collections import deque
import random
import math
import utils
import numpy as np
import threat
import time
import matplotlib.pyplot as plt

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 迭代次数(10000)
M = 100
# 迷你批的大小(64)
N = 64
# 更新网络的次数(50)
nb_train_steps = 50
# 折扣因子(0.99)
gamma = 0.99
# 更新目标网络的系数(0.001)
tau = 0.001


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):  # s_dim是联合状态空间6维，a_dim是联合动作空间4维
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 40)
        self.fc2 = nn.Linear(40 + a_dim, 30)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, s, a):
        s = F.relu(self.fc1(s))
        q = F.relu(self.fc2(torch.cat([s, a], dim=1)))
        q = self.fc3(q)
        return q


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):  # s_dim是联合状态空间6维，a_dim是联合动作空间4维
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, a_dim)

    def forward(self, s):
        a = F.relu(self.fc1(s))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a))
        return a


class Buffer:
    def __init__(self, size_max):
        self.buffer = deque(maxlen=size_max)

    def store(self, transition):  # transition是8维
        self.buffer.append(transition)

    def sample(self):
        minibatch = random.sample(self.buffer, N)
        s_lst1 = []
        a_lst1 = []
        r_lst1 = []
        s_prime_lst1 = []
        s_lst2 = []
        a_lst2 = []
        r_lst2 = []
        s_prime_lst2 = []
        for transition in minibatch:
            s1, a1, r1, s_prime1, s2, a2, r2, s_prime2 = transition
            s_lst1.append(s1)
            a_lst1.append(a1)
            r_lst1.append([r1])
            s_prime_lst1.append(s_prime1)
            s_lst2.append(s2)
            a_lst2.append(a2)
            r_lst2.append([r2])
            s_prime_lst2.append(s_prime2)
        return torch.Tensor(s_lst1).to(device), torch.Tensor(a_lst1).to(device), torch.Tensor(r_lst1).to(device), torch.Tensor(s_prime_lst1).to(device), \
               torch.Tensor(s_lst2).to(device), torch.Tensor(a_lst2).to(device), torch.Tensor(r_lst2).to(device), torch.Tensor(s_prime_lst2).to(device)


    def size(self):
        return len(self.buffer)


def select_action(s, actor):  # s是6维，actor是4维
    a = [actor(torch.Tensor(s).to(device))[0].item(), actor(torch.Tensor(s).to(device))[1].item(), actor(torch.Tensor(s).to(device))[2].item(), actor(torch.Tensor(s).to(device))[3].item()]
    a += np.random.normal(loc=0, scale=0.2, size=4)
    a = np.clip(a, -1, 1)
    return a


def soft_update(net, net_target):
    for param, param_target in zip(net.parameters(), net_target.parameters()):
        param_target.data.copy_(tau * param.data + (1 - tau) * param_target.data)


if __name__ == '__main__':

    critic = Critic(s_dim=6, a_dim=4).to(device)
    actor = Actor(s_dim=6, a_dim=4).to(device)

    critic_optim = optim.Adam(critic.parameters())
    actor_optim = optim.Adam(actor.parameters())

    critic_target = copy.deepcopy(critic)
    actor_target = copy.deepcopy(actor)

    buffer = Buffer(size_max=int(1e6))

    loss = [[], []]

    a_r1 = [0, 0]
    a_r2 = [0, 0]

    for i in range(M):

        # 第i次迭代的开始时间
        start = time.time()

        s_r1, s_r2, s_b = utils.reset()
        s1 = utils.observe(s_r1, s_b)
        s2 = utils.observe(s_r2, s_b)

        utils.normalize(s1)
        utils.normalize(s2)

        for j in range(utils.N):
            s = s_r1 + s_r2

            a_r1[0], a_r1[1], a_r2[0], a_r2[1] = select_action(s, actor)
            a_b = threat.select_action(s_r1, s_r2, s_b)

            s_r_prime1 = utils.step(s_r1, a_r1)
            s_r_prime2 = utils.step(s_r2, a_r2)
            if s_r_prime1[0] < 0 or s_r_prime1[0] > utils.MAX_X or s_r_prime1[1] < 0 or s_r_prime1[1] > utils.MAX_Y or\
                    s_r_prime2[0] < 0 or s_r_prime2[0] > utils.MAX_X or s_r_prime2[1] < 0 or s_r_prime2[1] > utils.MAX_Y:
                break
            s_b_prime = utils.step(s_b, a_b)
            if s_b_prime[0] < 0 or s_b_prime[0] > utils.MAX_X or s_b_prime[1] < 0 or s_b_prime[1] > utils.MAX_Y:
                break
            s_prime1 = utils.observe(s_r_prime1, s_b_prime)
            s_prime2 = utils.observe(s_r_prime2, s_b_prime)
            t1 = utils.evaluate(s_prime1)
            t2 = utils.evaluate(s_prime2)
            r1 = utils.reward(s_prime1)
            r2 = utils.reward(s_prime2)

            utils.normalize(s_prime1)
            utils.normalize(s_prime2)

            # 两个全胜利才胜利，一个死就全死
            if r1 + r2 == 2 or r1 == -1 or r2 == -1:
                buffer.store((s1, a_r1, t1, s_prime1, s2, a_r2, t2, s_prime2))
                break
            else:
                buffer.store((s1, a_r1, t1, s_prime1, s2, a_r2, r2, s_prime2))
                s_r1 = s_r_prime1
                s_b = s_b_prime
                s1 = s_prime1
                s_r2 = s_r_prime2
                s2 = s_prime2

        if buffer.size() >= N:

            sum = [0, 0]

            for j in range(nb_train_steps):

                s_lst1, a_lst1, r_lst1, s_prime_lst1, s_lst2, a_lst2, r_lst2, s_prime_lst2 = buffer.sample()

                s_lst = torch.cat((s_lst1, s_lst2), 0)
                r_lst = torch.stack((r_lst1, r_lst2), dim=1)
                a_lst = torch.stack((a_lst1, a_lst2), dim=1)
                s_prime_lst = torch.stack((s_prime_lst1, s_prime_lst2), dim=1)
                # for w in range(len(s_lst2)):
                #     s_lst.append(s_lst1[w]+s_lst2[w])
                #     a_lst.append(a_lst1[w]+a_lst2[w])
                #     r_lst.append(r_lst1[i] + r_lst2[i])
                #     s_prime_lst.append(s_prime_lst1[i]+s_prime_lst2[i])

                print(s_lst, '\n', r_lst)

                target = r_lst + gamma * critic_target(s_prime_lst, actor_target(s_prime_lst))

                critic_loss = F.mse_loss(critic(s_lst, a_lst), target)
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                actor_loss = -critic(s_lst, actor(s_lst)).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                soft_update(critic, critic_target)
                soft_update(actor, actor_target)

                sum[0] += critic_loss
                sum[1] += actor_loss

            loss[0].append(sum[0] / nb_train_steps)
            loss[1].append(sum[1] / nb_train_steps)

        # 第i次迭代的结束时间
        end = time.time()

        print('第', i + 1, '次迭代用时：', end - start, 's')

    plt.figure(1)
    plt.plot(loss[0], label='critic')
    plt.plot(loss[1], label='actor')
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    torch.save(critic.state_dict(), 'critic.pth')
    torch.save(actor.state_dict(), 'actor.pth')
