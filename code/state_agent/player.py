from os import path
import torch
from torch.distributions import Bernoulli
import sys
import random
sys.path.append("..")
from jurgen_agent.player import extract_featuresV2

random.seed()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ResNetBlock(torch.nn.Module):
    def __init__(self,neurons_in, neurons_out):
        super().__init__()
        layers = [
            torch.nn.Mish(),
            torch.nn.Dropout(),
            torch.nn.Linear(neurons_in,neurons_out),
            torch.nn.Mish(),
            torch.nn.Dropout(),
            torch.nn.Linear(neurons_out, neurons_out),
        ]
        self.res_block = torch.nn.Sequential(*layers)
        self.identity = torch.nn.Identity()
        if neurons_in != neurons_out:
            self.identity = torch.nn.Linear(neurons_in, neurons_out)

    def forward(self,x):
        return self.identity(x) + self.res_block(x)


class Team:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        if path.exists(path.join(path.dirname(path.abspath(__file__)), 'my_agent.pt')):
            self.k1_net = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'my_agent.pt'), map_location=device)
        else:
            layers = []#ResNetBlock(11, 32)]
            a = 16
            pa = 11
            for _ in range(10):
                layers.append(ResNetBlock(pa, a))
                pa = a
                if (pa < 512):
                    a *= 2
            layers.append(ResNetBlock(pa, 3))

            self.k1_net = torch.nn.Sequential(*layers)

        self.team = None
        self.num_players = None
        self.k1_net = self.k1_net.to(device)
        self.k1_net.eval()#


    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['sara_the_racer'] * num_players

    def act(self, player_state, opponent_state, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_state: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """

        actions = []
        for player_id, pstate in enumerate(player_state):
            features = extract_featuresV2(pstate, soccer_state, opponent_state, self.team)
            acceleration, steer, brake = self.k1_net(features.to(device))#

            steer = torch.sigmoid(torch.as_tensor(steer))
            brake = torch.sigmoid(torch.as_tensor(brake))
            if acceleration < 0.0:
                acceleration = 0.0
            elif acceleration > 1.0:
                acceleration = 1.0
            steer = float(-1 if steer < .5 else 1)
            brake = brake >= .5

            actions.append(dict(acceleration=acceleration, steer=steer, brake=brake))

        return actions

    def train_act(self, player_state, opponent_state, soccer_state):
        actions = []
        for player_id, pstate in enumerate(player_state):
            features = extract_featuresV2(pstate, soccer_state, opponent_state, self.team)

            if player_id == 0:
                acceleration, steer, brake = self.k1_net(features)
            else:
                acceleration, steer, brake = self.k2_net(features) * 0

            if torch.isnan(acceleration).item():
                acceleration = random.random()
            acc_dist = Bernoulli(logits=acceleration)

            if torch.isnan(steer).item():
                steer = random.random()
            steer_dist = Bernoulli(logits=steer)

            if torch.isnan(brake).item():
                brake = random.random()
            brake_dist = Bernoulli(logits=brake)


            acceleration = acc_dist.sample()
            steer = steer_dist.sample() * 2 - 1
            brake = brake_dist.sample() == 1

            actions.append(dict(acceleration=acceleration, steer=steer, brake=brake))
        return actions

class DummyTeam:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, opponent_state, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_state: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        actions = []
        for player_id, pstate in enumerate(player_state):
            actions.append(dict(acceleration=0, steer=0, brake=0))

        return actions
