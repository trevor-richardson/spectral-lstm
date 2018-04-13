import sys
import random
import pygame
from PIL import Image
import numpy as np
import torch

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

pygame.init()

size = width, height = 10, 10
black = (0, 0, 0)
white = (255, 255, 255)
screen = pygame.display.set_mode(size)

class Ball(object):

    def __init__(self, radius=1, color=(255, 255, 255)):
        self.radius = radius
        self.speed = [random.random()*2.0 - 1.0, random.random()*2.0 - 1.0]
        self.position = [random.uniform(self.radius, width-self.radius-1), random.uniform(self.radius, height-self.radius-1)]

    def move(self):
        self.position[0] += self.speed[0]
        self.position[1] += self.speed[1]
        if self.position[0] - self.radius < 0 or self.position[0] + self.radius > width:
            self.speed[0] = -self.speed[0]
        if self.position[1] - self.radius < 0 or self.position[1] + self.radius > height:
            self.speed[1] = -self.speed[1]

    def draw(self):
        pygame.draw.circle(screen, white, (int(round(self.position[0])), int(round(self.position[1]))), self.radius)



N = 5000
T = 50
n_balls = 1
if __name__ == "__main__":
    d = np.empty((N, T, 10, 10))

    for i in range(N): # N simulations:
        print ("Sim: ", i)
        balls = [Ball() for i in range(n_balls)]
        for t in range(T): # T Steps
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            [b.move() for b in balls]
            screen.fill(black)
            [b.draw() for b in balls]
            pygame.display.flip()
            rgb = pygame.surfarray.array3d(pygame.display.get_surface())
            I = Image.fromarray(rgb).convert("L")
            gray = np.array(I)
            d[i, t] = gray
            # print (gray.shape)
            # input("")
            # # I.show()
            # # input("")
            # # print (rgb.shape)
            # pygame.time.wait(50)

    d = d.astype(np.float32)
    torch.save(d, open('traindata.pt', 'wb'))
