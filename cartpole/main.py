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

def train_model(model):
  m = models.DefaultModel()
  m.train_and_save()
  m.load()
  print(m)

def generate_training_data(episodes):
  env = gym.make('CartPole-v0')

  agent = agents.RandomAgent(env)
  start = time.time()
  game_lib.play(
    env, agent, episodes=episodes, score_threshold=80, render=False, save_training_data=True)

  print('Elapsed time: {t} seconds'.format(t=round(time.time() - start, 4)))

  env.close()

# Generate model
generate_training_data(episodes=100000000)
model = models.DefaultModelWrapper(load=False)

def play():
  env = gym.make('CartPole-v0')

  agent = agents.DefaultAgent(env)
  #agent = agents.RandomAgent(env)
  start = time.time()
  game_lib.play(
    env, agent, episodes=10, render=True)

  print('Elapsed time: {t} seconds'.format(t=round(time.time() - start, 4)))

  env.close()

#play()

