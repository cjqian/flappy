import abc
import gym
import numpy as np
import six
import statistics
import time
from typing import List

WINNING_SCORE = 100 # This is actually 195 but let's be gentle for now
TIMESTEPS = 100

# AGENTS
class Agent(six.with_metaclass(abc.ABCMeta, object)):
  def __init__(self, env):
    self._env = env  # environment

  # Returns one-hot encoding; list where ath value is 1
  def encode_discrete_action(self, a):
    l = [0] * self._env.action_space.n
    l[a] = 1
    return l

  @abc.abstractmethod
  def step(self, observation):
    pass

class RandomAgent(Agent):
  def step(self, observation):
    return self._env.action_space.sample()

class GameLog(object):
  def __init__(self):
    self.history = []
    self.score = 0

def play_game(env, agent, render=True): # Returns a game log
  log = GameLog()

  observation = env.reset()
  for t in range(TIMESTEPS): # timesteps
      if render:
        env.render()
      action = agent.step(observation)
      # Observation at t - 1
      log.history.append(
        (observation, agent.encode_discrete_action(action)))
      observation, reward, done, info = env.step(action)
      log.score += reward
      if done:
          print("Episode finished after {ts} timesteps; score {sc}".format(ts=t+1, sc=log.score))
          break

  return log

def generate_training_data(env, agent, episodes=1000, score_threshold = 50, render=False):
  scores = []
  training_data = []
  for _ in range(episodes):
    log = play_game(env, agent, render=render)
    if log.score > score_threshold:
      training_data.extend(log.history)
    scores.append(log.score)

  print('Done. Mean score: {m}'.format(m=statistics.mean(scores)))
  print('Passed: {t}/{s}'.format(t = len([x for x in scores if x > score_threshold]), s=len(scores)))

  # Save training data
  np.save('training.npy', np.array(training_data))
  #print(np.load('training.npy', allow_pickle=True))

def main():
  env = gym.make('CartPole-v0')

  agent = RandomAgent(env)
  generate_training_data(env, agent, episodes=10, score_threshold=50, render=True)
  env.close()

main()
