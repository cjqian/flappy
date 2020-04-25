import abc
import gym
import six
import time


class Agent(six.with_metaclass(abc.ABCMeta, object)):
  def __init__(self, env):
    self._env = env  # environment

  @abc.abstractmethod
  def step(self, observation):
    pass

class RandomAgent(Agent):
  def step(self, observation):
    return self._env.action_space.sample()

def run_episodes(env, agent, n_episodes = 1):
  for i_episode in range(n_episodes):
      observation = env.reset()
      for t in range(100): # timesteps
          env.render()
          print(observation)
          action = agent.step(observation)
          observation, reward, done, info = env.step(action)
          if done:
              print("Episode finished after {} timesteps".format(t+1))
              break

def main():
  env = gym.make('CartPole-v0')
  print('Actions')
  print(env.action_space)
  print('Observations')
  print(env.observation_space)

  agent = RandomAgent(env)
  run_episodes(env, agent, n_episodes = 5)
  env.close()

main()
