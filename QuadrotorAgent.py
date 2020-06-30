import paddle.fluid as fluid
import parl
from parl import layers
import gym
import numpy as np


class QuadrotorAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim=5):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(QuadrotorAgent, self).__init__(algorithm)

        # 注意，在最开始的时候，先完全同步target_model和model的参数
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(name='obs',
                              shape=[self.obs_dim],
                              dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs',
                              shape=[self.obs_dim],
                              dtype='float32')
            act = layers.data(name='act',
                              shape=[self.act_dim],
                              dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(name='next_obs',
                                   shape=[self.obs_dim],
                                   dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(self.pred_program,
                                      feed={'obs': obs},
                                      fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(self.learn_program,
                                              feed=feed,
                                              fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost