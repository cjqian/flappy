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
  m = models.train_model()
  print(m)

def generate_training_data():
  env = gym.make('CartPole-v0')

  agent = agents.RandomAgent(env)
  start = time.time()
  game_lib.generate_training_data(
    env, agent, episodes=1000, score_threshold=60, render=False)

  print('Elapsed time: {t} seconds'.format(t=round(time.time() - start, 4)))

  env.close()

train_model()
