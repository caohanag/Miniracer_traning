#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:29:55 2022

@author: oliver
"""

import numpy as np
import random
from matplotlib import pyplot as plt
from itertools import count
import time



class Minirace:

    def __init__(self, level=1, size=6, normalise=False):
        # level is the dimensionality of state vector (1 or 2)
        self.level = min(2, max(1, level))  # 最小是1 最大是2
        # size is the number of track positions (every 2 pixels, from 1..n-1).
        # this means there are 2 more car positions (0 and n)
        self.size = max(3, size)
        self.xymax = 2 * (self.size + 2)  # 16
        # whether to normalise the state representation
        self.scale = 2 if normalise and level > 1 else 1.0

        # the internal state is s1 = (x, z, d). Previous internal state in s0.
        # x: x-coordinate of the car
        # z[]: x-coordinate of the track, one for each y-coordinate
        # d[]: dx for the track, for each y-coordinate(dx:the relative position of the middle of the track right in front of the car)
        self.reset()

    def observationspace(self):
        """
        Dimensionality of the observation space

        Returns
        -------
        int
            Number of values return as an observation.

        """
        return self.level

    def nexttrack(self, z, d=2):
        """
        Move the next piece of track based on coordinate z, and curvature

        Parameters
        ----------
        z : int
            x-coordinate for the previous track segment.
        d : int, optional
            previous "curvature" (change of coordinate). The default is 2.
            This is to prevent too strong curvature (car can only move 1 step)

        Returns
        -------
        znext : int
            The x-coordinate for the next piece of track (middle of the track).
        dz : int
            The change compared to the previous coordinate (-2..2).

        """
        trackd = random.randint(-2, 2)
        if self.level == 1 or abs(d) > 1:
            trackd = min(1, max(-1, trackd))#-1<=track挪动步数<=1   -----  限制曲率

        znext = max(1, min(self.size, z + trackd))#不出环境边界  1<=track<=6 （self.size：track左端位置限制）
        dz = znext - z
        return znext, dz

    def state(self):
        """
        Returns the (observed) state of the system.

        Depending on level, the observed state is an
        array of 1 to 5 values, or a pixel representation (level 0).

        Returns
        -------
        np.array
            level 1: [dx]
                 dx: relative distance in x-coordinate between car and next
                     piece of track (the one in front of the car).
                     May be normalised to values between -1 and 1,
                     depending on initialisation.
        """
        x, z, d = self.s1
        # print("self.s1:", self.s1)
        ## level 1:
        # return the difference between car x and the next piece of track
        if self.level == 1:
            return np.array([(z[2] - x) / self.scale])
        if self.level == 2:
            return np.array([(z[2] - x) / self.scale, (z[3] - z[2]) / self.scale])

        raise ValueError("level not implemented")

    def transition(self, action=0):
        """
        Apply an action and update the environment.
        0: do nothing
        1: move left
        2: move right

        Parameters
        ----------
        action : int, optional
            The action applied before the update.
            The default is 0 (representing no action).

        Returns
        -------
        np.array
            The new observed state of the environment.

        """
        self.s0 = self.s1

        if self.terminal():
            return self.state()

        x0, z0, d0 = self.s0
        z1 = np.roll(z0, -1) # 沿着给定轴滚动数组元素。超出最后位置的元素将会滚动到第一个位置
        d1 = np.roll(d0, -1)

        x1 = x0
        if action == 1:
            x1 = max(0, x0 - 1)
        elif action == 2:
            x1 = min(self.size - 1, x0 + 1)  #0<= x <=5

        z1[-1], d1[-1] = self.nexttrack(z0[-1], d0[-1])
        self.s1 = (x1, z1, d1)
        return self.state()

    def terminal(self):
        """
        Check if episode is finished.

        Returns
        -------
        bool
            True if episode is finished.

        """
        x, z, _ = self.s1

        return abs(z[1] - x) > 1.0 # 0 +1 -1 分别对应【车在路中央  车在路左端 车在路右端】

    def reward(self, action):
        """
        Calculate immediate reward.
        Positive reward for staying on track.

        Parameters
        ----------
        action : int
            0-2, for the 3 possible actions.

        Returns
        -------
        r : float
            immediate reward.

        """
        r = 1.0 if not self.terminal() else 0.0

        return r

    def step(self, action):
        # return tuple (state, reward, done)
        state = self.transition(action)
        r = self.reward(action)
        done = self.terminal()
        return (state, r, done)

    def reset(self):
        # the internal state is
        # x: x-coordinate of the car   车从0-7（以2为单位）
        # z[]: x-coordinate of the track, one for each y-coordinate  track从1-6，以2为单位
        # d[]: dx for the track, for each y-coordinate   track相比上一级track的偏移
        x = random.randint(0, self.size - 1)  # [0, self.size-1]
        z = np.zeros(self.xymax)
        d = np.zeros(self.xymax)
        z[0] = max(1, min(self.size, x))
        z[1] = z[0] #track最左边的地址 以2为单位
        d[1] = 2  #d[]记录track相比于上一级track挪动了多少
        for i in range(2, self.xymax):
            z[i], d[i] = self.nexttrack(z[i - 1], d[i - 1])

        self.s0 = (x, z, d)
        self.s1 = (x, z, d)
        '''
        print("self.s0:", self.s0)
        print("self.s1:", self.s1)
        self.render()
        '''
        return self.state()

    def sampleaction(self):  # 随机决策
        # return a random action [0,2]
        action = random.randint(0, 2)
        return action

    def render(self, text=True, reward=None, cm=plt.cm.bone_r, f=None): # 输出现状图
        x, z, _ = self.s1
        pix = self.to_pix(x, z, text)
        if text:
            if reward is not None:
                print('{:.3f}'.format(reward))
            print(''.join(np.flip(pix, axis=0).ravel()))
        else:
            fig, ax = plt.subplots()
            if reward is not None:
                plt.title(f'Reward: {reward}', loc='right')
            ax.axis("off")
            plt.imshow(pix, origin='lower', cmap=cm)
            if f is not None:
                plt.savefig(f, dpi=300)
            plt.show()

    def to_pix(self, x, z, text=False): # 生成现状图
        """
        Generate a picture from an internal state representation

        Parameters
        ----------
        x : int
            car x-coordinate
        z : np.array
            array with track coordinates
        text : bool, optional
            flag if generate text represenation

        Raises
        ------
        ValueError
            If x,y,z are outside their range.

        Returns
        -------
        image : np.array
            a square image with pixel values 0, 0.5, and 1.

        """
        if x < 0 or x > self.size + 1:
            raise ValueError('car coordinate value error')
        if np.min(z) < 1 or np.max(z) > self.size:
            raise ValueError('track coordinate value error')

        car = '#' if text else 2

        if text:
            image = np.array(list(':' * (self.xymax + 1)) * (self.xymax)).reshape(self.xymax, -1)
            image[:, -1] = '\n'
        else:
            image = np.ones((self.xymax, self.xymax), dtype=int)

        for i, j in enumerate(z):
            j = int(j * 2)
            image[i, j - 2:j + 4] = ' ' if text else 0

        image[0:2, 2 * x:(2 * x + 2)] = car

        return image

    def state_based_action(self, state):
        if self.level == 1: # (不归一化)
            if state >= 1:
                action = 2 # 右移
            elif state == 0:
                action = 0
            elif state <= -1:
                action = 1 #左移
        return action



def mypolicy_v0(state):
    action = therace.sampleaction()
    return action

def mypolicy(state):
    # selecting actions based on the state information
    action = therace.state_based_action(state)
    return action








if __name__ == "__main__":
    seed = 1
    # torch.manual_seed(seed)

    gamma = 0.99
    render = True
    finalrender = True
    log_interval = 1 # 100
    render_interval = 1 # 1000
    running_reward = 0

    therace = Minirace(level=1, size=6)

    starttime = time.time()

    for i_episode in count(1): # start = 1 无限递归
        state, ep_reward, done = therace.reset(), 0, False

        therace.render(reward=ep_reward)
        print('Episode {}\t t {}\t All reward: {:.2f} Start---------------------------'.format(
            i_episode, 0, ep_reward))


        rendernow = i_episode % render_interval == 0

        for t in range(1, 10000):  # Don't infinite loop while learning

            # select action (randomly)
            action = mypolicy(state)
            print("action: ", action)

            # take the action
            state, reward, done = therace.step(action) # s0 <- s1  s1为执行action之后的状态--------最终therace.render print出来的也是s1的状态
            reward = float(reward)  # strange things happen if reward is an int

            if render and rendernow:
                therace.render(reward=ep_reward)

            ep_reward += reward

            print('Episode {}\t t {}\t Current reward: {:.2f}\t All reward: {:.2f}'.format(
                i_episode, t, reward, ep_reward))
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward # Average reward

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\t Last reward: {:.2f}\t Average reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

        # check if we have solved minirace
        if running_reward > 50:
            secs = time.time() - starttime
            mins = int(secs / 60)
            secs = round(secs - mins * 60.0, 1)
            print("Solved in {}min {}s!".format(mins, secs))

            print("Running reward is now {:.2f} and the last episode "
                  "runs to {} time steps!".format(running_reward, t))

            if finalrender:
                state, ep_reward, done = therace.reset(), 0, False
                for t in range(1, 500):
                    action = mypolicy(state)
                    state, reward, done = therace.step(action)
                    ep_reward += reward
                    therace.render(text=False, reward=ep_reward)
                    if done:
                        break
                print('t {}\t Reward: {:.2f}\t'.format(
                    t, ep_reward))
            break


