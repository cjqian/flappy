import abc
import attr
import gym
import six
import time
from typing import List

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

@attr.s
class GameLog(object):
#####################
  history = attr.ib([])
  score = attr.ib(0)

def play_game(env, agent):
    log = GameLog()

    observation = env.reset()
    for t in range(TIMESTEPS): # timesteps
        env.render()
        action = agent.step(observation)
        # Observation at t - 1
        log.history.append(
          (observation, agent.encode_discrete_action(action)))
        print(log.history[-1])
        observation, reward, done, info = env.step(action)
        log.score += reward
        if done:
            print("Episode finished after {ts} timesteps; score {sc}".format(ts=t+1, sc=log.score))
            break

def main():
  env = gym.make('CartPole-v0')

  agent = RandomAgent(env)
  for _ in range(10): # Play 10 games
    play_game(env, agent)

  env.close()

main()
