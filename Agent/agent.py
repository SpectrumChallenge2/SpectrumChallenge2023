import torch.nn as nn
import itertools
import copy
from utility.environment_interface import EnvironmentInterface
import torch
import numpy as np
from pathlib import Path
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, agent_id):
        self._agent_id = agent_id
        self._buffer = deque(maxlen=10000)

    def push(self,  state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        self._buffer.append(experience)

    def get_latest_action_experience(self):
        if len(self._buffer) > 0:
            return self._buffer[-1][1]
        else:
            return None

    def sample(self, seq_size):
        start_index = random.randint(0, len(self._buffer) - seq_size)  # Choose a random start index
        batch = [self._buffer[i] for i in range(start_index, start_index + seq_size)]

        state_seq = []
        action_seq = []
        reward_seq = []
        next_state_seq = []

        for experience in batch:
            state, action, reward, next_state = experience

            state_seq.append(state)
            action_seq.append(action)
            reward_seq.append(reward)
            next_state_seq.append(next_state)

        return state_seq, action_seq, reward_seq, next_state_seq

    def __len__(self):
        return len(self._buffer)

class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: list,
                 hidden_act: str = 'ReLU',
                 out_act: str = 'Identity'):
        super(MLP, self).__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._num_neurons = num_neurons
        self._hidden_act = getattr(nn, hidden_act)()
        self._out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self._layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self._layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self._layers.append(self._out_act)
            else:
                self._layers.append(self._hidden_act)

    def forward(self, xs):
        for layer in self._layers:
            xs = layer(xs)
        return xs

class MultiAgent:
    def __init__(self, env):
        self._env = env
        self._run_time = 1000000
        self._num_agents = 2
        self._num_freq_channel = 6
        self._replay_buffer = [ReplayBuffer(i+1) for i in range(self._num_agents)]
        self._agents = [Agent(self._env, i+1) for i in range(self._num_agents)]
        self._target_update_interval = 10
        self._loss1 = []
        self._loss2 = []
        self._reward1 = []
        self._reward2 = []


    def get_actions(self):
        current_network_name = self._env.current_network_name
        network = self.get_network(current_network_name[-1])
        observation_dict = self._agents[network]._observation_dict
        reward = observation_dict['reward']['total_reward']
        observation = self._agents[network].convert_observation_dict_to_arr(observation_dict)
        prev_action = self.get_prev_action_arr(network, observation_dict)
        state = np.concatenate((prev_action, observation), axis=None)
        action_index = self._agents[network].get_action(state)
        action_dict = self._agents[network].convert_action_to_dict(action_index)
        action = self.convert_action_dict_to_arr(action_dict)
        return network, state, action, reward, action_dict

    def get_network(self, current_network_name):
        network = 0
        for i in range(self._num_agents):
            if current_network_name == self._agents[i]._agent_id:
                network = i
                break
        return network

    def get_prev_action_arr(self, network, observation_dict):
        observation_type = observation_dict['observation']['type']
        action_arr = np.zeros(self._num_freq_channel)
        replay_buffer_len = len(self._replay_buffer[network])
        if replay_buffer_len > 0:
            if observation_type == 'tx_data_packet':
                prev_action = self._replay_buffer[network].get_latest_action_experience()
                action_arr = prev_action
        return action_arr

    def convert_action_dict_to_arr(self, action_dict):
        action_arr = np.zeros(self._num_freq_channel)
        if action_dict['selected_freq_channel']:
            selected_channel = action_dict['selected_freq_channel']
            action_arr[selected_channel] = 1
        return action_arr

    def update(self, network, seq_size):
        state_seq, action_seq, reward_seq, next_state_seq = self._replay_buffer[network].sample(seq_size)

        loss, reward = self._agents[network].update(state_seq, action_seq, reward_seq, next_state_seq)

        if network == 1:
            self._loss1.append(loss)
            self._reward1.append(reward)
        else:
            self._loss2.append(loss)
            self._reward2.append(reward)


    def train(self, max_episode, max_steps, seq_size, save_file_name):
        self._env.disable_video_logging()
        self._env.disable_text_logging()
        # self._env.enable_video_logging()
        # self._env.enable_text_logging()
        print("Train start")
        for episode in range(max_episode):
            self._env.start_simulation(time_us=self._run_time)
            self._loss1 = []
            self._loss2 = []
            self._reward1 = []
            self._reward2 = []
            for i in range(self._num_agents):
                self._agents[i].set_init()
            for step in range(max_steps):
                network, state, action, reward, action_dict = self.get_actions()
                observation_dict = self._env.step(action_dict)
                if observation_dict == 0:
                    break
                else:
                    reward = observation_dict['reward']['total_reward']
                    next_observation = self._agents[network].convert_observation_dict_to_arr(observation_dict)
                    next_state = np.concatenate((action, next_observation), axis=None)
                    self._replay_buffer[network].push(state, action, reward, next_state)

                    if len(self._replay_buffer[network]) > seq_size:
                        self.update(network, seq_size)

            if episode % self._target_update_interval == 0:
                for i in range(self._num_agents):
                    self._agents[i]._qnet_target.load_state_dict(self._agents[i]._qnet.state_dict())

            print(f"episode:{episode}, loss1:{np.mean(self._loss1)}, reward1:{np.mean(self._reward1)}, "
                  f"loss2:{np.mean(self._loss2)}, reward2:{np.mean(self._reward2)}")


        print("finish")
        self.save_model(save_file_name)

    def test(self, run_time: int, test_file_name):
        self._env.disable_video_logging()
        self._env.disable_text_logging()
        # self._env.enable_video_logging()
        # self._env.enable_text_logging()
        self._env.start_simulation(time_us=run_time)
        self.load_model(test_file_name)
        for i in range(self._num_agents):
            self._agents[i].set_init()
        while True:
            network, state, action, reward, action_dict = self.get_actions()
            observation_dict = self._env.step(action_dict)
            if observation_dict == 0:
                break

    def save_model(self, file_name):
        model_path = Path(file_name)
        model_dict = {f"agent_{i}": agent.state_dict() for i, agent in enumerate(self._agents)}
        torch.save(model_dict, model_path)

    def load_model(self, file_name):
        model_path = Path(file_name)
        model_dict = torch.load(model_path)
        for i, agent in enumerate(self._agents):
            agent.load_state_dict(model_dict[f"agent_{i}"])



class Agent:
    def __init__(self, env, agent_id):
        self._env = env
        self._agent_id = agent_id
        self._dnn_learning_rate = 1e-5
        self._discount_factor = 0.99
        self._num_freq_channel = 6
        self._max_num_unit_packet = 4
        self._num_freq_channel_combination = 2 ** self._num_freq_channel - 1
        self._freq_channel_combination = [np.where(np.flip(np.array(x)))[0].tolist()
                                          for x in itertools.product((0, 1), repeat=self._num_freq_channel)][1:]

        self._device = "cpu"
        #self._use_cuda = torch.cuda.is_available()
        # if self._use_cuda:
        #     self._device = "cuda"

        self._observation = np.zeros(self._num_freq_channel)
        self._observation_dict = {}
        self._num_action = self._num_freq_channel_combination * self._max_num_unit_packet + 1

        self._qnet = MLP(input_dim=self._num_freq_channel * 2, output_dim=self._num_action,
                         num_neurons=[self._num_action]).to(self._device)
        self._qnet_target = copy.deepcopy(self._qnet)
        self._optimizer = torch.optim.Adam(params=self._qnet.parameters(), lr=self._dnn_learning_rate)

        self._qnet_target.load_state_dict(self._qnet.state_dict())
        self._criteria = nn.SmoothL1Loss()


    def set_init(self):
        initial_action = {'type': 'sensing'}
        observation_dict = self._env.step(initial_action)
        self._observation_dict = observation_dict
        self._agent_id = observation_dict['network'][-1]
        self._observation = self.convert_observation_dict_to_arr(observation_dict)


    def get_action(self, state):
        state = torch.Tensor(state).type(torch.float32).to(self._device)
        logit = self._qnet(state)
        action_index = torch.argmax(torch.softmax(logit, dim=-1))
        return action_index

    def convert_action_to_dict(self, action):
        if action == 0:
            action_dict = {'type': 'sensing'}
        else:
            num_unit_packet = 1
            freq_channel_combination_index = (action - 1) % self._num_freq_channel_combination
            selected_freq_channel = self._freq_channel_combination[freq_channel_combination_index]
            action_dict = {'type': 'tx_data_packet', 'selected_freq_channel': selected_freq_channel,
                           'num_unit_packet': num_unit_packet}
        return action_dict

    def convert_observation_dict_to_arr(self, observation_dict):
        observation_type = observation_dict['observation']['type']
        observation_arr = np.ones(self._num_freq_channel)
        if observation_type == 'sensing':
            is_sensed = observation_dict['observation']['sensed_freq_channel']
            observation_arr[is_sensed] = 0
        elif observation_type == 'tx_data_packet':
            observation_arr[:] = 0
            success_freq_channel_list = observation_dict['observation']['success_freq_channel']
            observation_arr[success_freq_channel_list] = 1
        return observation_arr

    def update(self,  state_seq, action_seq, reward_seq, next_state_seq):
        state_seq = torch.Tensor(state_seq).type(torch.float32).to(self._device)
        action_seq = torch.Tensor(action_seq).type(torch.float32).to(self._device)
        reward_seq = torch.Tensor(reward_seq).type(torch.float32).to(self._device)
        next_state_seq = torch.Tensor(next_state_seq).type(torch.float32).to(self._device)

        with torch.no_grad():
            q_max, _ = self._qnet_target(next_state_seq).max(dim=-1, keepdims=True)
            q_max = q_max.squeeze(dim=-1)
            q_target = reward_seq + self._discount_factor * q_max

        q_val = self._qnet(state_seq).gather(1, action_seq.long())
        loss = self._criteria(q_val, q_target.unsqueeze(1))

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        reward = reward_seq.mean()
        return loss.item(), reward.item()


    def state_dict(self):
        return {
            'qnet_state_dict': self._qnet.state_dict(),
        }

    def load_state_dict(self, state_dicts):
        self._qnet.load_state_dict(state_dicts['qnet_state_dict'])



if __name__ == "__main__":
    env = EnvironmentInterface()
    env.connect()
    ma_controller = MultiAgent(env)
    ma_controller.train(max_episode=100, max_steps=1000, seq_size=64, save_file_name='marl_agent.pt')
    ma_controller.test(run_time=5000000, test_file_name='marl_agent.pt')
