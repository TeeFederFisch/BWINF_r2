from Stack import *
import copy
from typing import Tuple

def brute_force(stack: Stack, pos: int=0, init: bool=True) -> Tuple[int, list]:
    """Checks every possible combination of flips on given stack and returns one of the best ways to sort it

    Args:
        stack (Stack): Stack to solve. The original stack will not be changed.
        pos (int): This is only needed within the recursion, will be ignored if `init == True`. Defaults to `0`
        init (bool, optional): You want this to always be True when manually calling the function. Defaults to `True`.

    Returns:
        Tuple[int, list]: (Amount of flips, positions where to flip)
    """
    if stack.is_sorted():
        return (stack.height - len(stack), [])
    
    if not init:
        stack.flip(pos)
    
    results = []
    paths = []
    length = stack.height - len(stack)
    
    for i in range(1, len(stack) + 1):
        s = copy.deepcopy(stack)
        res = brute_force(s, pos=i, init=False)
        results.append(res[0])
        paths.append(res[1])
        
        if res[0] == length:
            b = res[1]
            b.append(pos)
            return (results[-1], b)
    
    path = paths[results.index(min(results))]
    
    if init: path.reverse()
    else: path.append(pos)
        
    return (min(results), path)


data = input("Eigene Beispiele (z.B: 1, 2, 3):\n").split(", ")
while data != "":
    stack = Stack([int(d) for d in data])
    old_pancakes = stack.pancakes
    res = brute_force(stack)
    for i in res[1]:
        stack.flip(i)
    print(f"""
Stapel: {old_pancakes}
WUEOs: {res[0]} bei {res[1]}
Ergebnisstapel: {stack.pancakes}
    """)
    data = input("Eigene Beispiele (z.B: 1, 2, 3):\n").split(", ")

# [1]                                               1  -> 0
# [2, 1]                                            2  -> 1
# [2, 3, 1]                                         3  -> 2
# [3, 2, 4, 1]                                      4  -> 2
# [3, 4, 2, 5, 1]                                   5  -> 3
# [4, 3, 5, 2, 6, 1]                                6  -> 3
# [4, 5, 3, 6, 2, 7, 1]                             7  -> 4
# [5, 4, 6, 3, 7, 2, 8, 1]                          8  -> 4
# [5, 6, 4, 7, 3, 8, 2, 9, 1]                       9  -> 5
# [6, 5, 7, 4, 8, 3, 9, 2, 10, 1]                   10 -> 5
# [6, 7, 5, 8, 4, 9, 3, 10, 2, 11, 1]               11 -> 6
# [7, 6, 8, 5, 9, 4, 10, 3, 11, 2, 12, 1]           12
# [7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1]       13