import gym
import numpy as np

def policy_function(observation, theta):
    #根据当前的状态值和策略函数的参数值计算策略函数的输出
    x = np.dot(theta, observation)
    s = 1 / (1 + np.exp(-x))
    #根据策略函数的输出进行动作选择
    if s > 0.5:
        action = 1
    else :
        action = 0
    return s, action

def generate_an_episode(env, theta):
    episode = []
    pre_observation = env.reset()
    count_action = 0
    while True:
        s, action = policy_function(theta, pre_observation)
        observation, reward, done, info = env.step(action)
        episode.append([pre_observation, action, s, reward])
        pre_observation = observation
        count_action += 1
        if done or count_action > 5000:
            break
    return episode

# S(x)=1/(1+e-x)
def monte_carlo_policy_gradient(env):
    learning_rate = 0.01
    discount_factor = 0.95

    # 随机初始化策略函数的参数theta
    theta = np.random.rand(4)

    # 让智能体玩2000个回合
    for i in range(1000000):
        # 生成一条完整的游戏情节Episode
        episode = generate_an_episode(env, theta)
        # 使用梯度上升的方法优化策略函数的参数
        for t in range(len(episode)):
            observation, action, s, reward = episode[t]
            # print(observation)
            # 根据蒙特卡罗策略梯度算法中的公式更新参数theta
            # l = -log(1 / (1 + exp(-(theta * observation)))) * reward
            # delta(l|theta) = - (1/s) * s * (1-s) * (-observation) * reward
            # theta += learning_rate * discount_factor ** t * reward * s * (1 - s) * (-observation)
            theta += learning_rate * discount_factor ** t * reward * (1-s) * observation
        if i % 100 == 0:
            # 测试策略的性能
            total_reward = test(env, theta)
            print("iteration %i, Total reward: %i" % (i, total_reward))
            print(theta)

def test(env, theta):
    #重置游戏环境
    observation = env.reset()
    total_reward = 0.
    #智能体最多执行3000个动作(即奖励值达到3000后就结束游戏)
    for i in range(3000):
        #可视化游戏画面(重绘一帧画面)
        # env.render()
        #使用策略函数选择下一个动作
        s, action = policy_function(theta, observation)
        #执行动作
        observation, reward, done, info = env.step(action)
        #计算累积奖励
        total_reward += reward
        if done:
            break
    return total_reward

if __name__ == "__main__":
    #注册游戏环境
    game_env = gym.make("CartPole-v1")
    #取消限制
    game_env = game_env.unwrapped
    #让智能体开始学习玩"CArtPole-vl”游戏
    monte_carlo_policy_gradient(game_env)