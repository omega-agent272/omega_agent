import pickle
import random
from collections import namedtuple, deque
from typing import List

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

import events as e
from .callbacks import state_to_features, bomb_net

import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'extra1', 'action', 'next_state', 'extra2', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 15000  # keep only ... last transitions
BATCH_SIZE = 120
GAMMA = 0.85
SAVE_INTERVAL = 100
lr = 0.00005
ALPHA_END = 0.75
ALPHA_DECAY = 50000

# Events
#PLACEHOLDER_EVENT = "PLACEHOLDER"

# experience buffer
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.epoch = -1

    # if buffer is full begin overwriting the first elements
    def push(self, transition):
        self.epoch += 1
        """Saves a transition."""
        if len(self.memory)+1 > self.epoch:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = transition
            self.position = (self.position + 1) % self.capacity
        else:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[1000 + self.position] = transition
            self.position = (self.position + 1) % (self.capacity - 1000)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.rewards = []
    self.sum_rewards = 0
    self.transitions = ReplayMemory(TRANSITION_HISTORY_SIZE)
    self.target_net = bomb_net().to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())

    # define optimizer
    self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
st
    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py

    rACTIONS = {'UP' : 0, 'RIGHT' : 1, 'DOWN' : 2, 'LEFT' : 3, 'WAIT' : 4, 'BOMB' : 5}
    if self_action!=None:
        old_state = state_to_features(self, old_game_state)
        new_state = state_to_features(self, new_game_state)
        self.transitions.push(Transition(old_state[0], old_state[1], torch.tensor([rACTIONS[self_action]], device=device), new_state[0], new_state[1], reward_from_events(self, events)))

        # optimize model
    optimize_model(self)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    rACTIONS = {'UP' : 0, 'RIGHT' : 1, 'DOWN' : 2, 'LEFT' : 3, 'WAIT' : 4, 'BOMB' : 5}
    if self.epoch%100 == 0:
        if not self.epoch == 0:
            self.rewards = np.append(self.rewards, self.sum_rewards)
            print('mean reward=',self.rewards[len(self.rewards)-1]/100)
            self.sum_rewards = 0


    last_state = state_to_features(self, last_game_state)
    if last_action!=None:
        self.transitions.push(Transition(last_state[0], last_state[1], torch.tensor([rACTIONS[last_action]], device=device), None, None, reward_from_events(self, events)))

    # update tagget network
    self.target_net.load_state_dict(self.policy_net.state_dict())
    # update epoch +1
    self.epoch += 1

    # Store the model
    if self.epoch%SAVE_INTERVAL == 0:
        torch.save({
        'epoch': self.epoch,
        'model_state_dict': self.policy_net.state_dict(),
        'rewards': self.rewards
        }, "my-saved-model.pt")


def reward_from_events(self, events: List[str]):
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        'COIN_COLLECTED': 5,
        'KILLED_OPPONENT': 7,
        'MOVED_LEFT': -.1,
        'MOVED_RIGHT': -.1,
        'MOVED_UP': -.1,
        'MOVED_DOWN': -.1,
        'WAITED': -.3,
        'INVALID_ACTION': -.5,
        'BOMB_DROPPED': -.1,
        'BOMB_EXPLODED': 0,
        'CRATE_DESTROYED': .6,
        'COIN_FOUND': 0,
        'KILLED_SELF': -2,
        'GOT_KILLED': -2,
        'OPPONENT_ELIMINATED': 0,
        'SURVIVED_ROUND': 10

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.sum_rewards += reward_sum
    return torch.tensor([reward_sum], device=device)

def optimize_model(self):
    if len(self.transitions) < BATCH_SIZE:
        return
    transitions = self.transitions.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    non_final_next_extras = torch.cat([s for s in batch.extra2
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    extra_batch = torch.cat(batch.extra1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch, extra_batch).gather(dim=1, index=action_batch.unsqueeze(-1))
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states, non_final_next_extras).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    loss = (ALPHA_END + (1-ALPHA_END)*np.exp(-self.epoch /ALPHA_DECAY)) * F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    #for param in self.policy_net.parameters():
    #    param.grad.data.clamp_(0, 1)
    self.optimizer.step()
