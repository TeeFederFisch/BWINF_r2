from __future__ import annotations
import torch
from random import shuffle

class FlipNotPossible(Exception): pass
class SortNotPossible(Exception): pass

class Stack:
    """Class to represent pancake stacks as declared by BWINF 41 in 2023 challenge 3."""
    def __init__(self, items: list) -> None:
        self.height = len(items)
        self.pancakes = items
        
    def __len__(self) -> int:
        return len(self.pancakes)
    
    def flip(self, pos: int) -> None:
        """Takes all pancakes before pos, flips them and then deletes the topmost pancake
        
        Args:
            pos (int): Where the stack will be flipped

        Raises:
            FlipNotPossible: When the given pos is outside of the range 0-height
        """
        if pos > len(self.pancakes): raise FlipNotPossible
        rest = self.pancakes[pos:]
        toFlip = self.pancakes[:pos]
        toFlip.reverse()
        self.pancakes = toFlip + rest
        self.pancakes.pop(0)
        
    def is_sorted(self) -> bool:
        """
        Returns:
            bool: True when self.pancakes is sorted.
        """
        return self.pancakes == sorted(self.pancakes)

    def print(self, pre="", post=""):
        """Prints `self.pancakes`"""
        print(pre, self.pancakes, post)
    
    def create_random(height: int) -> Stack:
        """Returns a randomized Stack where object.height equals given height.

        Args:
            height (int): Height of random Stack.

        Returns:
            object: Stack object
        """
        pancakes = [*range(1, height + 1)]
        shuffle(pancakes)
        return Stack(pancakes)
    
    def as_tensor(self, fill=False, normalized=True) -> torch.Tensor:
        """Returns self.pancakes as torch.Tensor with size self.height by adding zeros in front if necessary.

        Returns:
            torch.Tensor: self.pancakes as Tensor
        """
        out = [0 for _ in range(self.height - len(self.pancakes)) if fill]
        if fill and not normalized:
            out += self.pancakes
            return torch.tensor(out, dtype=torch.float32)
        
        for i in self.pancakes:
            out.append(sorted(self.pancakes).index(i) + 1)

        return torch.tensor(out, dtype=torch.float32)

    def from_file(filename: str) -> Stack:
        data = []
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in range(1, len(lines)):
                out = ""
                for i in lines[line]:
                    if i == "\\": break
                    out += i
                data.append(int(out))
        return(Stack(data))
    
    def unpack(self) -> list:
        return self.pancakes

    def sort_by_ai(self) -> list:
        nets = []
        for i in range(len(self), 1, -1):
            try:
                with open(f"./nnet/trained_nets/net{i}.save", "rb") as file:
                    nets.append(torch.load(file))
            except FileNotFoundError:
                raise FileNotFoundError("Not every necessary pretrained NN was found")
        
        flips = []
        for (i, net) in enumerate(nets):
            if self.is_sorted(): break
            flips.append(net(self.as_tensor()).argmax().item() + 1)
            self.flip(flips[-1])
        return flips