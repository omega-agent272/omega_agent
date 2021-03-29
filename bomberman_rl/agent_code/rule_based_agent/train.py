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
from .callbacks import state_to_features

import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions
BATCH_SIZE = 800
GAMMA = 0.99
SAVE_INTERVAL = 10
lr = 0.001

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

# experience buffer
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # if buffer is full begin overwriting the first elements
    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def setup_training(self):
    #print("executing setup in training\n")
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')

    self.transitions = ReplayMemory(TRANSITION_HISTORY_SIZE)
    self.target_net = bomb_net(9, 9, 6).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    #print('self.target_net.load_state_dict ', self.target_net.load_state_dict)

    # define optimizer
    self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    #print("executing game_events_occured in training\n")
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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py

    rACTIONS = {'UP' : 0, 'RIGHT' : 1, 'DOWN' : 2, 'LEFT' : 3, 'WAIT' : 4, 'BOMB' : 5}

    if new_game_state['step']!=1:
        self.logger.info("Step:" + str(new_game_state['step']) + "\nAction: "  + str(rACTIONS[self_action]))
        self.transitions.push(Transition(state_to_features(self, old_game_state), torch.tensor([rACTIONS[self_action]], device=device), state_to_features(self, new_game_state), reward_from_events(self, events)))
        #print('state to feature(old): ', state_to_features(self, old_game_state))
        #print('state to feature(new): ', state_to_features(self, new_game_state))
        #print('self_action', torch.tensor([rACTIONS[self_action]]))
        #print('reward: ', reward_from_events(self, events))


    # optimize model
    optimize_model(self)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    #print("executing end_of_round in training\n")
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    rACTIONS = {'UP' : 0, 'RIGHT' : 1, 'DOWN' : 2, 'LEFT' : 3, 'WAIT' : 4, 'BOMB' : 5}


    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.push(Transition(state_to_features(self, last_game_state), torch.tensor([rACTIONS[last_action]], device=device), None, reward_from_events(self, events)))

    # update tagget network
    self.target_net.load_state_dict(self.policy_net.state_dict())
    # update epoch +1
    self.epoch += 1

    # Store the model
    if self.epoch%SAVE_INTERVAL == 0:
        torch.save({
        'epoch': self.epoch,
        'model_state_dict': self.policy_net.state_dict(),
        }, "my-saved-model.pt")
    #print('model stored')


def reward_from_events(self, events: List[str]):
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 2,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.WAITED: -0.2,
        e.INVALID_ACTION: -0.5,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 0.2,
        e.COIN_FOUND: 0.3,
        e.KILLED_SELF: -2,
        e.GOT_KILLED: -1,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
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
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch).gather(dim=1, index=action_batch.unsqueeze(-1))
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #print('loss= ', loss)
    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    #for param in self.policy_net.parameters():
    #    param.grad.data.clamp_(0, 1)
    self.optimizer.step()
    #print('finished optimizer')
class bomb_net(nn.Module):
    def __init__(self, h, w, outputs): #h,w=9
        super().__init__()
        self.fc1 = nn.Linear(in_features=h*w, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=40)
        self.fc3 = nn.Linear(in_features=40, out_features=12)
        self.out = nn.Linear(12,6)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = self.out(t)
        return t
