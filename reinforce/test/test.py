import gym
import numpy as np

def policy_function(observation, theta):
  # 根据当前的状态值和策略函数的参数值计算策略函数的输出
  x = np.dot(theta, observation)
  s = 1 / (1 + np.exp(-x))
  # 根据策略函数的输出进行动作选择
  if s > 0.5:
    action = 1
  else:
    action = 0
  return s, action

def test(env, theta):
  # 重置游戏环境
  observation = env.reset()
  total_reward = 0.
  # 智能体最多执行3000个动作(即奖励值达到3000后就结束游戏)
  for i in range(3000):
    # 可视化游戏画面(重绘一帧画面)
    env.render()
    # 使用策略函数选择下一个动作
    s, action = policy_function(theta, observation)
    # 执行动作
    observation, reward, done, info = env.step(action)
    # 计算累积奖励
    total_reward += reward
    if done:
      break
  return total_reward


if __name__ == "__main__":
  # 注册游戏环境
  game_env = gym.make("CartPole-v1")
  # 取消限制
  game_env = game_env.unwrapped
  # print(test(game_env, [0.16347292,0.29266818,0.55101392,0.62101714]))

  print(test(game_env, [0.05558771, 0.56496784, 0.72183366, 0.96786954]))


