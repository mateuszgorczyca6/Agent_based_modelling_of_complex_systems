import numpy as np
from random import random as rand
from random import randint
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class Car:
    def __init__(self, nr):
        self.velocity = 0
        self.nr = nr

    def accelerate(self, closest_road: np.array, velocity_decrease_prob: float, velocity_max: int):
        self.velocity += 1
        if np.sum(closest_road) > 0:
            self.velocity = min(np.argmax(closest_road), self.velocity)
        if rand() < velocity_decrease_prob:
            self.velocity -= 1
        self.velocity = max(min(self.velocity, velocity_max), 0)

    def __repr__(self):
        return f"Car({self.velocity})"


class Nagel_Schreckenberg_model:
    __car_class = Car

    def __init__(self, velocity_decrease_prob: float, car_appearance_prob: float, road_length: int = 100,
                 velocity_max: int = 5):
        self.velocity_decrease_prob = velocity_decrease_prob
        self.car_appearance_prob = car_appearance_prob
        self.road_len = road_length
        self.velocity_max = velocity_max

    def simulate(self, steps: int = 100):
        self.history = np.zeros((steps + 1, self.road_len), object)
        self.spread_cars()
        self.history[0] = self.road.copy()
        for step in range(steps):
            self.iterate()
            self.history[step + 1] = self.road.copy()

    def spread_cars(self):
        #  making logical array with cars positioned
        self.road_logic = np.random.binomial(1, self.car_appearance_prob, self.road_len)
        #  replacing True with Car objects
        self.road = np.array(list(map(lambda x: self.__car_class(randint(0, 2)) if x else 0, self.road_logic)))

    def iterate(self):
        road_longer = np.concatenate([self.road_logic, self.road_logic[:self.velocity_max]])
        for id, cell in enumerate(self.road):
            if isinstance(cell, self.__car_class):
                cell.accelerate(road_longer[id + 1: id + cell.velocity + 2], self.velocity_decrease_prob,
                                self.velocity_max)
        self.move_cars()

    def move_cars(self):
        new_road = np.zeros(self.road_len, object)
        for id, cell in enumerate(self.road):
            if isinstance(cell, self.__car_class):
                new_pos = id + cell.velocity
                new_road[new_pos % self.road_len] = cell
        self.road = new_road.copy()
        self.road_logic = self.road != 0


images_info = {
    0: {"img": Image.open("cars/car1.png"), "zoom": 0.15},
    1: {"img": Image.open("cars/car2.png"), "zoom": 0.15},
    2: {"img": Image.open("cars/car3.png"), "zoom": 0.15}
}


def plot_history(self, fname: str):
    def draw_image(x, y, nr):
        imagebox = OffsetImage(images_info[nr]["img"], zoom=images_info[nr]["zoom"])
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

    filenames = []
    images = []
    for t, flashback in enumerate(self.history):
        _, ax = plt.subplots(figsize=(30, 0.3), facecolor="white")
        car_pos = np.array(np.where(flashback != 0)).T
        for pos in car_pos:
            draw_image(pos[0], 0, self.history[t, pos[0]].nr)
        plt.xticks([i - 0.5 for i in range(self.road_len + 2)], color="white")
        plt.yticks([-0.5, 0.5], color="white")
        plt.xlim(-0.5, self.road_len - 0.5)
        plt.ylim(-0.5, 0.5)
        plt.grid(color="black")
        plt.draw()
        image_name = f"time{t}.png"
        filenames.append(image_name)
        plt.savefig(image_name)
        images.append(imageio.imread(image_name))
        plt.close()
    imageio.mimsave(fname, images, fps=round(len(self.history) / 15))
    for i in filenames:
        os.remove(i)
