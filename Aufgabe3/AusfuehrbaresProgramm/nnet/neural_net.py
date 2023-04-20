# This is a Deep Q-Learning Network, inspired by
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from __future__ import annotations

import time
import typing
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from random import randint
from Stack import Stack, FlipNotPossible

import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # Compute on GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing on " + ("GPU" if torch.cuda.is_available() else "CPU"))

    IMPROVE_OLD_NET = True
    
    # Training is really slow with realtime plots, they look good though
    REALTIME_PLOTS = False
    
    BATCH_SIZE = 128
    # Effectively defines how far in the future the net "thinks"
    GAMMA = 0.99
    # Mutation probability decreases by EPS_DECAY down to EPS_END
    EPS_START = 0.9
    EPS_END = 0.
    EPS_DECAY = 1000
    # Defines how strong the policy nets parameters affect the target net
    TAU = 0.005
    # The optimizers learning rate
    LR = 1e-2

    # Size of pancakestack = amount of possible flips
    STACK_HEIGHT = 16

    if torch.cuda.is_available():
        num_epochs = 10000
    else:
        num_epochs = 1000

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class Environment:
    """The environment is responsible for providing data and computing the nets decisions effects."""
    def __init__(self, size: int, data=None) -> None:
        self.size = size
        self.stack = data
        
    def get_rand_action(self) -> int:
        """Returns a random action."""
        return randint(0, self.size - 1)

    def init_rand_stack(self) -> torch.Tensor:
        """Initializes a new random stack in this environment.

        Returns:
            `torch.Tensor`: The initialized stack as `torch.Tensor`
        """
        while True:
            self.stack = Stack.create_random(self.size)
            if not self.stack.is_sorted(): break
        return self.stack.as_tensor().to(device)
    
    def init_from(data: list | Stack) -> Environment:
        """Initializes a new environment where `self.stack` equals `data`

        Args:
            data (`list | Stack`): List of pancakes or `Stack` of pancakes

        Returns:
            `object`: The `Environment` object
        """
        return Environment(Stack(data) if type(data) == list else data)

    def step(self, action: int) -> typing.Tuple[torch.Tensor | None, float, bool, bool]:
        """Computes the state a given action has caused and the corresponding reward.

        Args:
            action (`int`): The position to flip the stack at

        Returns:
            (`torch.Tensor | None`, `float`, `bool`, `bool`): (resulting state, reward, terminated, truncated)
        """
        try:
            self.stack.flip(action)
            if self.stack.is_sorted():
                return (self.stack.as_tensor(fill=True, normalized=False).to(device),
                        len(self.stack) * 2.,
                        False,
                        True)
            else:
                for n in lower_nets:
                    res = n(self.stack.as_tensor())
                    self.stack.flip(res.argmax() + 1)
                    if self.stack.is_sorted():
                        return (self.stack.as_tensor(fill=True,
                                                     normalized=False).to(device),
                                                    len(self.stack) * 2.,
                                                    False,
                                                    True)
                return (self.stack.as_tensor(fill=True, normalized=False).to(device),
                        len(self.stack) * 2.,
                        False,
                        True)
        except FlipNotPossible:
            return (None, -10., True, False)

class ReplayMemory(object):
    """A memory class for the DQN to be able to "remember" past actions"""
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Packs `args` in a named Tuple `Transition` and saves it"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        """Returns a random sample from memory where size is `batch_size`"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# The actual NN
class DQN(nn.Module):
    """Linear neural network"""
    def __init__(self, n_observations=5, n_actions=5) -> DQN:
        """Initializes the net.

        Args:
            n_observations (`int, optional`): Amount of input nodes. Defaults to 5.
            n_actions (`int, optional`): Amount of output nodes. Defaults to 5.
        """
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Runs given data through `self.net` and returns the output.

        Args:
            data (`torch.Tensor`): Input data

        Returns:
            `torch.Tensor`: Tensor of probabilities for different actions
        """
        return self.net(data)

if __name__ == "__main__":
    lower_nets = []
    for i in range(STACK_HEIGHT - 1, 1, -1):
        try:
            with open(f"./nnet/trained_nets/net{i}.save", "rb") as file:
                lower_nets.append(torch.load(file))
        except FileNotFoundError:
            raise FileNotFoundError
    # The net is dynamic and can deal with a varying input and output size
    n_actions = STACK_HEIGHT
    n_observations = STACK_HEIGHT

    # The environment is responsible for providing data and computing the nets decisions effects
    env = Environment(n_actions)

    # The policy net is the one actually being trained, it then creates "policies" to train the target net
    if IMPROVE_OLD_NET:
        try:
            with open(f"./nnet/trained_nets/net{STACK_HEIGHT}.save", "rb") as saved_net:
                policy_net = torch.load(saved_net).to(device)
                target_net = DQN(n_observations, n_actions).to(device)
        except FileNotFoundError:
            print("WARNING: Savefile not found, resuming with new nets instead")
            policy_net = DQN(n_observations, n_actions).to(device)
            target_net = DQN(n_observations, n_actions).to(device)
    else:
        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
            

    # To make sure both nets weights are initially the same
    target_net.load_state_dict(policy_net.state_dict())

    # AdamW has proven to work relatively good compared to other optimizers
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    memory = ReplayMemory(10000)

    steps_done = 0

# Next action is either random (mutation) or chosen by policy net
def select_action(state: torch.Tensor) -> torch.Tensor:
    """Either returns an action chosen by the policy net or a random action with an exponentially decreasing probability.

    Args:
        state (`torch.Tensor`): The state the environment is currently in

    Returns:
        `torch.Tensor`: Tensor probabilities of suggested actions
    """
    global steps_done
    sample = random.random()
    
    # Formula to decrease the decrease of mutation probability exponentially by step and EPS_DECAY
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.get_rand_action()]], device=device, dtype=torch.long)

episode_durations = []
actually_sorted = []
rewards = []

def plot_mean100(data: list, label: str) -> None:
    """Plots the mean of 100 data nodes from given `data` and labels the graph.

    Args:
        data (`list`): Data to plot
        label (`str`): What the graph depicts
    """
    data_t = torch.tensor(data, dtype=torch.float)
    if len(data_t) >= 100:
        means = data_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label=label)
        plt.legend()

def plot_information(show_result=False):
    """Plots important information about the current training state of the neural network in realtime.

    Args:
        show_result (`bool, optional`): Only change this to `True` when you don't want to update the plot ever again. Defaults to `False`.
    """
    plt.figure(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Episode")
    plt.ylabel("Pancakes left")
    plt.plot(rewards_t.numpy(), label="Pancakes left")
    
    plot_mean100(rewards, "Avg reward")
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.display(plt.gcf())
        if not show_result:
            display.clear_output(wait=True)
            
            
def optimize_model():
    # The model will only optimize after enough data is accumulated
    if len(memory) < BATCH_SIZE:
        return
    
    # Gets a random sample and turns the tensor of Transitions to 
    # a single Transition with tensors of all values
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # s is None when the epoch was terminated, not truncated (see training loop)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # This creates values that represent how likely the policy net would have
    # chosen the action that it remembered
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # This is for computing how certain the target net would have been in its decision
    # Later this will be used to compute the difference between 
    # the policy and target nets decisions, which results in the loss
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # next_state_values represents the confidence of the target net,
    # adding the reward results in what the confidence values should have been
    # With a low GAMMA value > 0 the instant reward values get more important than
    # the certainty to get more rewards in the future
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Computes loss between chosen actions and highest-reward-actions
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backprogation to update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
if __name__ == "__main__":
    t = time.time()
    for i_episode in range(num_epochs):
        # Initialize the environment and get its state
        state = env.init_rand_stack()
        state = state.clone().detach().unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated = env.step(action.item() + 1)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = observation.clone().detach().unsqueeze(0)

            # Save the transition to learn from later
            memory.push(state, action, next_state, reward)

            state = next_state

            # Update policy_net
            optimize_model()

            # The target nets weights are only partially affected by the policy net to even out
            # drastic updates on the policy net
            tnsd = target_net.state_dict()
            pnsd = policy_net.state_dict()
            for key in pnsd:
                tnsd[key] = pnsd[key]*TAU +tnsd[key]*(1-TAU)
            target_net.load_state_dict(tnsd)
            
            if truncated:
                actually_sorted.append(STACK_HEIGHT)
                episode_durations.append(t + 1)
            elif terminated:
                episode_durations.append(STACK_HEIGHT)
                actually_sorted.append(0)
            if done:
                rewards.append(reward / 2)
                if REALTIME_PLOTS:
                    plot_information()
                break
            

    print(f"Completed in {time.time() - t}")
    plot_information(show_result=True)
    if input("Would you like to save? (y/N)").lower() == "y":
        try:
            f = open(f"./nnet/trained_nets/net{STACK_HEIGHT}.save", "x")
            f.close()
        except:
            pass
        f = open(f"./nnet/trained_nets/net{STACK_HEIGHT}.save", "wb")
        
        target_net.to("cpu")
        torch.save(target_net, f)
        f.close()
        print("Saved successfully")

    plt.ioff()
    plt.show()
    plt.close(1)