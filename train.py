import argparse
import gym
import numpy as np
import time
import parl
from QuadrotorAgent import QuadrotorAgent
from QuadrotorModel import ActorModel, CriticModel, QuadrotorModel
from parl.utils import logger, summary, action_mapping, ReplayMemory
from parl.algorithms import DDPG
from rlschool import make_env

ACTOR_LR = 2e-4  # Actor网络更新的 learning rate
CRITIC_LR = 1e-3  # Critic网络更新的 learning rate

GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward


def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        action_four = action[0] + 0.2 * action[1:]
        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action_four = np.clip(np.random.normal(action_four, 1.0), -1.0, 1.0)

        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action_four = action_mapping(action_four, env.action_space.low[0],
                                     env.action_space.high[0])

        next_obs, reward, done, info = env.step(action_four)
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                    batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)
            action_four = action[0] + 0.2 * action[1:]

            action_four = np.clip(action_four, -1.0, 1.0)
            action_four = action_mapping(action_four, env.action_space.low[0],
                                         env.action_space.high[0])

            next_obs, reward, done, info = env.step(action_four)

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)

def restore_model(num_of_ckptfile):
    ckpt = 'model_dir/steps_{}.ckpt'.format(num_of_ckptfile)  # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
    agent.restore(ckpt)

def main():
    # 创建飞行器环境
    env = make_env("Quadrotor", task="hovering_control")
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_dim = act_dim + 1

    model = QuadrotorModel(act_dim)
    algorithm = DDPG(model,
                     gamma=GAMMA,
                     tau=TAU,
                     actor_lr=ACTOR_LR,
                     critic_lr=CRITIC_LR)
    agent = QuadrotorAgent(algorithm, obs_dim, act_dim)

    # parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
    rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

    # 启动训练
    test_flag = 0
    total_steps = 0
    while total_steps < TRAIN_TOTAL_STEPS:
        train_reward, steps = run_episode(env, agent, rpm)
        total_steps += steps
        logger.info('Steps: {} Reward: {}'.format(total_steps,
                                                  train_reward))  # 打印训练reward

        if total_steps // TEST_EVERY_STEPS >= test_flag:  # 每隔一定step数，评估一次模型
            while total_steps // TEST_EVERY_STEPS >= test_flag:
                test_flag += 1

            evaluate_reward = evaluate(env, agent)
            logger.info('Steps {}, Test reward: {}'.format(
                total_steps, evaluate_reward))  # 打印评估的reward

            # 每评估一次，就保存一次模型，以训练的step数命名
            ckpt = 'model_dir/steps_{}.ckpt'.format(total_steps)
            agent.save(ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        help='environment name',
                        default='HalfCheetah-v2')
    parser.add_argument('--train_total_steps',
                        type=int,
                        default=int(1e6),
                        help='maximum training steps')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='the step interval between two consecutive evaluations')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='Temperature parameter α determines the relative importance of the \
        entropy term against the reward (default: 0.2)')

    args = parser.parse_args()

    main()