import socket
import logging
import random
import numpy as np
import time
from .messenger import Messenger
from typing import Optional, Dict, List, Union

logging.basicConfig(level=logging.INFO, format='%(message)s')


class EnvironmentInterface:
    def __init__(self):
        self._identifier = 'p1'
        self._server_address: str = '127.0.0.1'
        self._server_port: int = 8888
        self._socket: socket = None
        self._messenger: Optional[Messenger] = None
        self._operator_info: Optional[Dict] = None
        self._current_network_name = None
        self._state: str = 'unconnected'
        self._score: Optional[Dict] = None

    @property
    def state(self):
        return self._state

    @property
    def current_network_name(self):
        return self._current_network_name

    @property
    def operator_info(self):
        return self._operator_info

    @property
    def sta_list(self):
        return self._operator_info[self._current_network_name]['sta_list']

    @property
    def freq_channel_list(self):
        return self._operator_info[self._current_network_name]['freq_channel_list']

    @property
    def num_unit_packet_list(self):
        return self._operator_info[self._current_network_name]['num_unit_packet_list']

    def connect(self):
        if self._state == 'unconnected':
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket_connected = False
            for trial in range(10):
                try:
                    logging.warning(f'Try to connect to the environment docker ({trial+1}/10).')
                    self._socket.connect((self._server_address, self._server_port))
                except socket.error:
                    logging.warning('Connection failed.')
                    time.sleep(0.5)
                else:
                    socket_connected = True
                    break
            if socket_connected:
                self._messenger = Messenger(self._socket)
                self._messenger.send('player_id', self._identifier)
                msg_type, msg = self._messenger.recv()
                if msg_type == 'connection_successful':
                    self._state = 'idle'
                    logging.info('Successfully connected to environment.')
                else:
                    self._environment_failed()
            else:
                logging.warning('Check out if the environment docker is up.')
        else:
            logging.info('Already connected to environment.')

    def start_simulation(self, time_us):
        if not ((isinstance(time_us, float) or isinstance(time_us, int)) and time_us > 0):
            logging.warning('Simulation time should be a positive number.')
            return
        if self._state == 'idle':
            self._messenger.send('start_simulation', time_us)
            msg_type, msg = self._messenger.recv()
            if msg_type == 'operator_info':
                self._operator_info = msg
                self._score = {n: {'packet_success': 0, 'collision': 0, 'total_score': 0}
                               for n in self._operator_info}
                self._score['composite'] = 0
                self._state = 'running'
                logging.info('Simulation started.')
                msg_type, msg = self._messenger.recv()
                if msg_type == 'observation':
                    msg = self._process_observation(msg)
                    return msg
                else:
                    return None
            else:
                self._environment_failed()
        elif self._state == 'unconnected':
            logging.info("Environment is not connected.")
        elif self._state == 'running':
            logging.info("Simulation is already running.")

    def _process_observation(self, msg):
        self._current_network_name = msg['network']
        msg['reward'].pop('packet_delayed')
        msg['reward'].pop('packet_dropped')
        score = msg.pop('score')
        self._score[self._current_network_name] = \
            {x: score[x] for x in self._score[self._current_network_name]}
        self._score['composite'] = min([self._score[n]['total_score'] for n in self._operator_info])
        return msg

    def step(self, action: Dict) -> Union[Optional[Dict], int]:
        if self._check_action(action):
            self._messenger.send('action', action)
        else:
            return -1
        msg_type, msg = self._messenger.recv()
        if msg_type == 'observation':
            msg = self._process_observation(msg)
            return msg
        elif msg_type == 'simulation_finished':
            logging.info("Simulation is finished.")
            self._state = 'idle'
            return 0
        else:
            self._environment_failed()

    def get_score(self) -> Dict:
        return self._score

    def _check_action(self, action: Dict) -> bool:
        if not isinstance(action, Dict):
            logging.warning("Action should be a dictionary.")
            return False
        if 'type' not in action:
            logging.warning("Key \'type\' should exist.")
            return False
        if action['type'] == 'sensing':
            return True
        elif action['type'] == 'tx_data_packet':
            if ('selected_freq_channel' not in action) or ('num_unit_packet' not in action):
                logging.warning("Keys \'selected_freq_channel\' and \'num_unit_packet\' should exist "
                                "if the type is \'tx_data_packet\'.")
                return False
            selected_freq_channel = action['selected_freq_channel']
            if len(selected_freq_channel) == 0:
                logging.warning("\'selected_freq_channel\' should not be empty.")
                return False
            if not all([f in self.freq_channel_list for f in selected_freq_channel]):
                logging.warning(f"Frequency channels in \'selected_freq_channel\' should be in {self.freq_channel_list}.")
                return False
            num_unit_packet = action['num_unit_packet']
            if not isinstance(num_unit_packet, int):
                logging.warning("Value of \'num_unit_packet\' should be an integer.")
                return False
            if action['num_unit_packet'] not in self.num_unit_packet_list:
                logging.warning(f"Value of \'num_unit_packet\' should be one of {self.num_unit_packet_list}.")
                return False
        else:
            logging.warning("Value of \'type\' should be either \'sensing\' or \'tx_data_packet\'.")
            return False
        return True

    def random_action_step(self, num_step=1):
        if self._state != 'running':
            logging.warning("Simulation is not running.")
        else:
            last_obs_rew = {}
            for step in range(num_step):
                action_type = random.choice(['sensing', 'tx_data_packet'])
                if action_type == 'tx_data_packet':
                    num_tx_freq_channel = random.randint(1, len(self.freq_channel_list))
                    selected_freq_channel = random.sample(self.freq_channel_list, num_tx_freq_channel)
                    num_unit_packet = random.choice(self.num_unit_packet_list)
                    action = {'type': 'tx_data_packet', 'selected_freq_channel': selected_freq_channel,
                              'num_unit_packet': num_unit_packet}
                else:
                    action = {'type': 'sensing'}
                logging.info(f"Action: {action}\n")
                obs_rew = self.step(action)
                if obs_rew == 0:
                    break
                else:
                    logging.info(f"Network: {obs_rew['network']}")
                    logging.info(f"Observation: {obs_rew['observation']}")
                    if 'reward' in obs_rew:
                        logging.info(f"Reward: {obs_rew['reward']}")
                    logging.info(f"Score: {self._score}")
                    last_obs_rew = obs_rew
            return last_obs_rew

    def _configure_logging(self, log_type: str, enable: bool):
        if self._state == 'idle':
            self._messenger.send('configure_logging', {'log_type': log_type, 'enable': enable})
            msg_type, msg = self._messenger.recv()
            if msg_type == 'logging_configured':
                logging.info("Logging is successfully configured.")
            else:
                self._environment_failed()
        elif self._state == 'running':
            logging.warning("Logging cannot be configured when the simulator is running.")
        elif self._state == 'unconnected':
            logging.warning("Environment is not connected.")

    def enable_video_logging(self):
        self._configure_logging(log_type='video', enable=True)

    def disable_video_logging(self):
        self._configure_logging(log_type='video', enable=False)

    def enable_text_logging(self):
        self._configure_logging(log_type='text', enable=True)

    def disable_text_logging(self):
        self._configure_logging(log_type='text', enable=False)

    def _environment_failed(self):
        raise Exception('Environment failed. Restart the environment docker.')

