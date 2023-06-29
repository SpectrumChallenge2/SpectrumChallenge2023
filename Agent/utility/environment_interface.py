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


