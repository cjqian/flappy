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
        #print(observation)
        action = agent.step(observation)

        log.history.append((observation, action))
        observation, reward, done, info = env.step(action)
        log.score += reward
        if done:
            print("Episode finished after {ts} timesteps; score {sc}".format(ts=t+1, sc=log.score))
            break

def main():
  env = gym.make('CartPole-v0')
  print('Actions')
  print(env.action_space)
  print('Observations')
  print(env.observation_space)

  agent = RandomAgent(env)
  for _ in range(10): # Play 10 games
    play_game(env, agent)

  env.close()

main()
