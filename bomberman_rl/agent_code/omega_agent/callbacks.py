import os
import pickle
import random
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
from random import shuffle

import numpy as np

import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
n_actions = 6
GAMMA = 0.85
EPS_START = 0
EPS_END = 0
EPS_DECAY = 20000
h=w=9


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in trainmming mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #create new model or load exisitng one
    self.policy_net = bomb_net().to(device)
    if not os.path.isfile("my-saved-model.pt"):
        self.epoch = 0
    else:
        checkpoint = torch.load("my-saved-model.pt", map_location=device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.policy_net.eval()
        self.epoch = checkpoint['epoch']

#choose action at random, if epsilon is greedy, choose action according to policy_net else
def act(self, game_state: dict) -> str:
    state = state_to_features(self, game_state)
    # todo Exploration vs exploitation
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.epoch / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # exploitation
            a = self.policy_net(state[0],state[1]).max(1)[1].view(1, 1)

        #rint('exploitation: ', ACTIONS[a])
        return ACTIONS[a]
    else:
        a = random.randrange(n_actions)
        return ACTIONS[a]


# THE NEW FUNCTION!!!!!!!!!!!!!!!!!!!!!!!!!!!
#map input to feature (one-hot)
def state_to_features(self, game_state):
    # catch-all
    if game_state is None:
        return None


    field = game_state['field']
    bombs = game_state['bombs']
    coins = game_state['coins']
    selfs = game_state['self']
    others = game_state['others']

    # store basic information
    self.round = game_state['round']
    self.step = game_state['step']
    self.data = game_state['self']


    # push all data in one field
    wall_mask = (field == -1)
    crate_mask = (field == 1)

    #set all values to 0, mask entries with True set to placeholder
    wall_field = np.zeros(np.shape(field))
    wall_field[wall_mask] = 1

    crate_field = np.zeros(np.shape(field))
    crate_field[crate_mask] = 1

    bomb_field = np.zeros(np.shape(field))
    for i in [xy for (xy,t) in bombs]:
        bomb_field[i] = 1

    coin_field = np.zeros(np.shape(field))
    for (x,y) in coins:
        coin_field[x,y] = 1

    enemy_field = np.zeros(np.shape(field))
    for i in [xy for (_,_,_,xy) in others]:
        enemy_field[i] = 1
    (x,y) = game_state['self'][3]
    player_field = np.zeros(np.shape(field))
    player_field[x][y]=1



    # one-hot array (17x17x6)
    reduced_field = np.stack([wall_field, crate_field, bomb_field, coin_field, enemy_field])
    # rotation algorithm
    means = np.zeros(4)

    means[0] = np.mean(player_field[0:8,0:8])
    means[1] = np.mean(player_field[0:8,9:17])
    means[2] = np.mean(player_field[9:17,9:17])
    means[3] = np.mean(player_field[9:17,0:8])


    # bomb available?
    bomb_available = game_state['self'][2]
    # step to find targets
    bool_field = (game_state['field']==0)
    bomb_xys = [xy for (xy, t) in game_state['bombs']]
    targets = game_state['coins']
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]
    d = look_for_targets(bool_field, game_state['self'][3], game_state['coins'], self.logger)


    direction = np.zeros(4)
    if d == (x-1, y):
        direction[0] = 1
    elif d == (x+1, y):
        direction[1] = 1
    elif d == (x, y-1):
        direction[2] = 1
    elif d == (x, y+1):
        direction[3] = 1

    # convert feature arrays

    small_field = reduced_field[:,x-1:x+2, y-1:y+2]


    # convert feature arrays
    extra = np.append(direction, bomb_available).astype(np.float32)
    extra = torch.from_numpy(extra.copy())
    extra = extra.unsqueeze(0).to(device)

    small_field = small_field.astype(np.float32)
    small_field = torch.from_numpy(small_field.copy())
    small_field = small_field.unsqueeze(0).to(device)
    return [small_field,  extra]

class bomb_net(nn.Module):
    def __init__(self):
        super(bomb_net, self).__init__()
        self.fc1 = nn.Linear(in_features=50, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=6)


    def forward(self, state, extra):
        state = state.flatten(start_dim=1)
        state = torch.cat((state,extra), dim=1)
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        state = self.fc3(state)
        return state
# initalize policy and target network and sync them

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
