from turtle import *
from PIL import Image

def show(*args):
    reset()
    speed(50)
    penup()
    goto(args[0])
    pendown()
    shape("circle")
    color("green")
    shapesize(.4, .4, .4)
    stamp()
    color("black")
    shapesize(.2, .2, .2)

    for i in args[1:]:
        goto(i)
        stamp()

    shapesize(.4, .4, .4)
    color("red")
    stamp()
    
    global don
    don = False
    def yes():
        global don
        don = True
        clear()
    
    penup()
    hideturtle()
    color("white")
    # getscreen().getcanvas().postscript(file=str(args[0]) + ".ps")
    while not don:
        onkeypress(yes)
        listen()
        update()

def get_new():
    clear()
    global points
    points = []
    
    speed(500)
    penup()
    shape("circle")
    color("black")
    shapesize(.2, .2, .2)
    hideturtle()
    global done
    done = False
    def no():
        global done
        done = True
    
    def setposl(x, y):
        showturtle()
        setposition((x, y))
        stamp()
        global points
        points.append((x, y))
    
    while not done:
        onkeypress(no)
        onscreenclick(setposl)
        update()
        listen()
    return points