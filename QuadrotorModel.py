import paddle.fluid as fluid
import parl
from parl import layers

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid_size = 100
        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=5, act='tanh')

    def policy(self, obs):
        hid = self.fc1(obs)
        logits = self.fc2(hid)
        return logits


class CriticModel(parl.Model):
    def __init__(self):
        hid_size = 100
        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        concat = layers.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = layers.squeeze(Q, axes=[1])
        return Q


class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()