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
import test_total
import ddpg_independent

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 迭代次数(10000)
M = 300
# 迷你批的大小(64)
N = 64
# 更新网络的次数(50)
nb_train_steps = 50
# 折扣因子(0.99)
gamma = 0.99
# 更新目标网络的系数(0.001)
tau_lose = 0.001  # 输了的学得更快
tau_win = 0.0001

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim * 2, 80)
        self.fc2 = nn.Linear(80 + a_dim, 60)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, s, a):  # s输入就是2个三维的归一化观测
        s = F.relu(self.fc1(s))
        q = F.relu(self.fc2(torch.cat([s, a], dim=1)))
        q = self.fc3(q)
        return q


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim * 2, 80)
        self.fc2 = nn.Linear(80, 60)
        self.fc3 = nn.Linear(60, a_dim)

    def forward(self, s):  # s输入就是2个三维
        a = F.relu(self.fc1(s))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a))
        return a


class Buffer:
    def __init__(self, size_max):
        self.buffer = deque(maxlen=size_max)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self):
        minibatch = random.sample(self.buffer, N)  # 输入时都先存入自己的观测
        s_lst = []  # 联合观测，自己的放前面
        a_lst = []
        r_lst = []
        s_prime_lst = []  # 下一联合观测，自己的放前面
        flag_lst = []
        for transition in minibatch:
            s_r, a, r, s_prime_r, flag = transition
            s_lst.append(s_r)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime_r)
            flag_lst.append(flag)
        return torch.Tensor(s_lst).to(device), torch.Tensor(a_lst).to(device), \
               torch.Tensor(r_lst).to(device), torch.Tensor(s_prime_lst).to(device), torch.Tensor(flag_lst).to(device)

    def size(self):
        return len(self.buffer)


def select_action(s, actor, t):  # 输入的s就是两个三维观测,t是第t代
    a = [actor(torch.Tensor(s).to(device))[0].item(), actor(torch.Tensor(s).to(device))[1].item()]
    delta = 0.2
    # delta = delta * math.exp(-t)
    decay = 0.9999
    delta = delta * math.pow(decay, t)
    a += np.random.normal(loc=0, scale=delta, size=2)
    a[0] = np.clip(a[0], -0.9, 1.1)
    a[1] = np.clip(a[1], -1, 1)
    return a


def soft_update_win(net, net_target):
    for param, param_target in zip(net.parameters(), net_target.parameters()):
        param_target.data.copy_(tau_win * param.data + (1 - tau_win) * param_target.data)

def soft_update_lose(net, net_target):
    for param, param_target in zip(net.parameters(), net_target.parameters()):
        param_target.data.copy_(tau_lose * param.data + (1 - tau_lose) * param_target.data)


if __name__ == '__main__':
    actor1 = Actor(s_dim=3, a_dim=2).to(device)
    # actor1.load_state_dict(torch.load('./net/actor1_1000.pth'))
    actor1.load_state_dict(torch.load('./net/actor1_1000.pth'))
    actor1.eval()

    actor2 = Actor(s_dim=3, a_dim=2).to(device)
    actor2.load_state_dict(torch.load('./net/actor2_1000.pth'))
    actor2.eval()
    actor_optim2 = optim.Adam(actor2.parameters())

    critic2 = Critic(s_dim=3, a_dim=2).to(device)
    critic2.load_state_dict(torch.load('./net/critic2_1000.pth'))
    critic2.eval()
    critic_optim2 = optim.Adam(critic2.parameters())

    critic_target2 = copy.deepcopy(critic2)
    actor_target2 = copy.deepcopy(actor2)

    buffer2 = Buffer(size_max=int(1e6))

    loss1 = [[], []]
    loss2 = [[], []]

    winrate = 0
    winrate_ = 0
    winlist = []

    for i in range(M):
        # 第i次迭代的开始时间
        start = time.time()

        s_r1, s_r2, s_b = utils.reset()
        s1 = utils.observe(s_r1, s_b)
        s2 = utils.observe(s_r2, s_b)

        utils.normalize(s1)
        utils.normalize(s2)

        s_1 = s1 + s2  # 联合观测，自己的放前面
        s_2 = s2 + s1

        result = 0

        for j in range(utils.N):

            # a_r1 = select_action(s_1, actor1, i)  # 每个输入的观测都是自己的放在前面
            a_r1 = [actor1(torch.Tensor(s_1).to(ddpg_independent.device))[0].item(), actor1(torch.Tensor(s_1).to(ddpg_independent.device))[1].item()]
            a_r2 = select_action(s_2, actor2, i)
            a_b, target = threat.select_action(s_r1, s_r2, s_b)

            s_r_prime1 = utils.step(s_r1, a_r1)
            s_r_prime2 = utils.step(s_r2, a_r2)
            # if s_r_prime1[0] < 0 or s_r_prime1[0] > utils.MAX_X or s_r_prime1[1] < 0 or s_r_prime1[1] > utils.MAX_Y:
            #     break
            # if s_r_prime2[0] < 0 or s_r_prime2[0] > utils.MAX_X or s_r_prime2[1] < 0 or s_r_prime2[1] > utils.MAX_Y:
            #     break
            s_b_prime = utils.step(s_b, a_b)
            # if s_b_prime[0] < 0 or s_b_prime[0] > utils.MAX_X or s_b_prime[1] < 0 or s_b_prime[1] > utils.MAX_Y:
            #     break
            s_prime1 = utils.observe(s_r_prime1, s_b_prime)  # 自己的下一观测
            s_prime2 = utils.observe(s_r_prime2, s_b_prime)

            # # t1 = utils.evaluate(s_prime1, s_r_prime1)
            # r1 = utils.reward(s_prime1, s_r_prime1, s_b_prime)
            # t2 = utils.evaluate(s_prime2, s_r_prime2)
            # r2 = utils.reward(s_prime2, s_r_prime2, s_b_prime)
            # # if target == 1:
            # #     if r2 == 1:
            # #         t2 = 100
            # #         print('hello')
            # #     elif r2 == 0 and t2 > -2:
            # #         t2 = 6 * t2
            # #     elif r2 == -1:
            # #         t2 = -10
            # # if t2 <= -2:
            # #     print(t2)
            #
            # if r2 == -1:
            #     print('hello')
            #     t2 = -1
            # else:
            #     t2 = t2/(j+1)
            r1 = utils.evaluate_pt(s_prime1, s_r_prime1, s_b_prime)
            r2 = utils.evaluate_pt(s_prime2, s_r_prime2, s_b_prime)

            utils.normalize(s_prime1)
            utils.normalize(s_prime2)

            r = r1 + r2

            # t2 = r2 * pow(0.999, (j+1))
            t2 = r2 * pow(0.99, (j + 1))
            # print(t2)

            s_prime_1 = s_prime1 + s_prime2  # 下一联合观测，自己的放前面
            s_prime_2 = s_prime2 + s_prime1

            if r2 == -1:
                result = -1

            # 一个赢了就赢了，一个死了就死了
            if r1 == 1 or r2 == 1 or r1 == -1 or r2 == -1:
                buffer2.store((s_2, a_r2, t2, s_prime_2, r2))
                break
            else:
                buffer2.store((s_2, a_r2, t2, s_prime_2, r2))
                s_r1 = s_r_prime1
                s_1 = s_prime_1
                s_r2 = s_r_prime2
                s_2 = s_prime_2
                s_b = s_b_prime

        if buffer2.size() >= N:

            sum2 = [0, 0]

            for j in range(nb_train_steps):

                s_lst2, a_lst2, r_lst2, s_prime_lst2, flag = buffer2.sample()
                # print(flag)

                target2 = r_lst2 + gamma * critic_target2(s_prime_lst2, actor_target2(s_prime_lst2))

                critic_loss2 = F.mse_loss(critic2(s_lst2, a_lst2), target2)
                critic_optim2.zero_grad()
                critic_loss2.backward()
                critic_optim2.step()

                actor_loss2 = -critic2(s_lst2, actor2(s_lst2)).mean()
                actor_optim2.zero_grad()
                actor_loss2.backward()
                actor_optim2.step()

                if result == -1:
                    soft_update_lose(critic2, critic_target2)
                    soft_update_lose(actor2, actor_target2)
                else:
                    soft_update_win(critic2, critic_target2)
                    soft_update_win(actor2, actor_target2)

                sum2[0] += critic_loss2
                sum2[1] += actor_loss2

            loss2[0].append(sum2[0] / nb_train_steps)
            loss2[1].append(sum2[1] / nb_train_steps)

        # 第i次迭代的结束时间
        end = time.time()

        if (i + 1) % 50 == 0:
            winrate, nolossrate = test_total.winrate_test_fixed(actor1, actor2)
            winlist.append(nolossrate)
            print(nolossrate, winrate)
            if nolossrate >= 0.88:
                torch.save(critic2.state_dict(), 'critic2.pth')
                torch.save(actor2.state_dict(), 'actor2.pth')
                break
            else:
                nolossrate = 0

        # if (i + 1) >= 500:
        #     torch.save(critic2.state_dict(), 'critic2.pth')
        #     torch.save(actor2.state_dict(), 'actor2.pth')
        #     break

        print('第', i + 1, '次迭代用时：', end - start, 's')

    plt.figure(1)
    plt.plot(loss2[0], label='critic2')
    plt.plot(loss2[1], label='actor2')
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.figure(2)
    episodes = []
    for i in range(len(winlist)):
        episodes.append(50*(i+1))
    plt.plot(episodes, winlist, label='win-rate in every 50 episodes')
    plt.xlabel('episodes')
    plt.ylabel('win-rate')
    plt.legend()
    plt.show()
