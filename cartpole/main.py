import abc
import collections
import gym
import numpy as np
import six
import statistics
import time

import agents
import game_lib
import models

def train_model():
  m = models.DefaultModel()
  m.train_and_save()
  m.load()
  print(m)

def generate_training_data():
  env = gym.make('CartPole-v0')

  agent = agents.RandomAgent(env)
  start = time.time()
  game_lib.play(
    env, agent, episodes=10000, render=False)

  print('Elapsed time: {t} seconds'.format(t=round(time.time() - start, 4)))

  env.close()

def play():
  env = gym.make('CartPole-v0')

  agent = agents.DefaultAgent(env)
  start = time.time()
  game_lib.play(
    env, agent, episodes=10, render=True)

  print('Elapsed time: {t} seconds'.format(t=round(time.time() - start, 4)))

  env.close()

play()
