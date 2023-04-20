from Stack import Stack
# Doesn't run without this import due to DQN being loaded from a file
from neural_net import DQN

for i in range(10):
    print(f"Beispiel: {i + 1}")
    stack = Stack.from_file(f"./inputs/pancake{i}.txt")
    stack.print(pre="Stapel:")
    print(f"WUEOs bei: {stack.sort_by_ai()}")
    stack.print(pre="Ergebnisstapel:", post="\n")

data = input("Eigene Beispiele (z.B: 1, 2, 3):\n").split(", ")
while data != "":
    stack = Stack([int(d) for d in data])
    print(f"WUEOs bei: {stack.sort_by_ai()}")
    stack.print(pre="Ergebnisstapel:", post="\n")
    data = input("Weitere Beispiele:\n").split(", ")