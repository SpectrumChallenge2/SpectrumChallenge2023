from utility.environment_interface import EnvironmentInterface


class KeyboardAgent:
    def __init__(self, environment: EnvironmentInterface):
        self._env = environment

    def run(self, run_time):
        obs_rew = self._env.start_simulation(time_us=run_time)
        while True:
            print(f"Network: {obs_rew['network']}")
            print(f"Observation: {obs_rew['observation']}")
            if 'reward' in obs_rew:
                print("Reward: ", obs_rew['reward'])
            print(f"Score: {self._env.get_score()}")
            action = self.keyboard_control()
            if 'num_step' in action:
                obs_rew = self._env.random_action_step(num_step=action['num_step'])
            else:
                print(f"Action: {action}\n")
                obs_rew = self._env.step(action=action)

    def keyboard_control(self):
        sta_list = self._env.sta_list
        freq_channel_list = self._env.freq_channel_list
        num_unit_packet_list = self._env.num_unit_packet_list
        while True:
            action_type = input('Choose action (sensing: s, transmit: t, random: r): ')
            if action_type.lower() in ['s', 't', 'r']:
                break
        if action_type.lower() == 's':
            action = {'type': 'sensing'}
            return action
        elif action_type.lower() == 't':
            while True:
                selected_freq_channel = input(f'Choose the frequency channels (separated by comma) '
                                              f'among {freq_channel_list}: ')
                selected_freq_channel = [eval(f) for f in selected_freq_channel.split(',')]
                if all([f in freq_channel_list for f in selected_freq_channel]):
                    break
            while True:
                num_unit_packet = eval(input(f'Choose the number of unit packets '
                                             f'among {num_unit_packet_list}: '))
                if num_unit_packet in num_unit_packet_list:
                    break
            action = {'type': 'tx_data_packet', 'selected_freq_channel': selected_freq_channel,
                      'num_unit_packet': num_unit_packet}
            return action
        elif action_type.lower() == 'r':
            while True:
                num_step = int(input('Input number of steps: '))
                if num_step > 0:
                    break
            return {'num_step': num_step}


if __name__ == "__main__":
    env = EnvironmentInterface()
    env.connect()
    agent = KeyboardAgent(environment=env)
    agent.run(100000)
