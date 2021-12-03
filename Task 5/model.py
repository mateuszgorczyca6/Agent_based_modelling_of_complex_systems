import numpy as np
from random import random as rand
from random import randint
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
from os.path import exists
import pandas as pd


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

    def mc_average_velocity(self, MC_N: int):
        avg_vel = 0
        for _ in range(MC_N):
            self.simulate(100)
            avg_vel += self.calc_average_velocity()
        avg_vel /= MC_N
        return avg_vel

    def calc_average_velocity(self):
        avg_vel = self.history[self.history != 0]
        if avg_vel.size == 0:
            return 0
        avg_vel = map(lambda car: car.velocity, avg_vel)
        return np.array(list(avg_vel)).mean()


images_info = {
    0: {"img": Image.open("cars/car1.png"), "zoom": 0.15, "color": "r"},
    1: {"img": Image.open("cars/car2.png"), "zoom": 0.15, "color": "b"},
    2: {"img": Image.open("cars/car3.png"), "zoom": 0.15, "color": "g"},
    3: {"img": Image.open("cars/car4.png"), "zoom": 0.15, "color": "pink"}
}


def plot_history(model, fname: str, title: str, rectangles=False):
    def draw_image(x, y, nr):
        imagebox = OffsetImage(images_info[nr]["img"], zoom=images_info[nr]["zoom"])
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

    filenames = []
    images = []
    for t, flashback in enumerate(model.history):
        fig, ax = plt.subplots(figsize=(30, 0.8), facecolor="white")
        car_pos = np.array(np.where(flashback != 0)).T
        for pos in car_pos:
            if rectangles:
                ax.add_patch(Rectangle(
                    (pos[0] - 0.5, -0.5),
                    1,
                    1,
                    edgecolor="black",
                    facecolor=images_info[model.history[t, pos[0]].nr]["color"],
                    lw=0))
            else:
                draw_image(pos[0], 0, model.history[t, pos[0]].nr)
                plt.draw()
        plt.xticks([i - 0.5 for i in range(model.road_len + 2)], color="white")
        plt.yticks([-0.5, 0.5], color="white")
        plt.title(title + f"; t = {t}", fontsize=22)
        plt.xlim(-0.5, model.road_len - 0.5)
        plt.ylim(-0.5, 0.5)
        plt.grid(color="black")
        plt.draw()
        fig.subplots_adjust(top=0.5)

        image_name = f"time{t}.png"
        filenames.append(image_name)
        plt.savefig(image_name)
        images.append(imageio.imread(image_name))
        plt.close()

    speed_fname = "".join([fname, "_fast.gif"])
    imageio.mimsave("".join([fname, ".gif"]), images, fps=2)
    imageio.mimsave(speed_fname, images, fps=10)
    for i in filenames:
        os.remove(i)


def mc_average_velocity_array(ps: list, rhos: list, MC_N: int, gif_plot: bool = True):
    average_velocity = np.zeros((3, 3))
    for i, p in enumerate(ps):
        for j, rho in enumerate(rhos):
            model = Nagel_Schreckenberg_model(p, rho, 100)
            average_velocity[i][j] = model.mc_average_velocity(MC_N)
            print(f"Done: p = {p}, rho = {rho}")

            fname = f"base__p_{p}__rho_{rho}.gif"
            if gif_plot and not exists(fname):
                plot_history(model, f"base__p_{p}__rho_{rho}", f"Road simulation; p = {p}; rho = {rho}")

    average_velocity = pd.DataFrame(average_velocity, index=map_str(ps, "p = "), columns=rhos).transpose()
    return average_velocity


def map_str(array, string):
    return list(map(lambda x: "".join((string, str(x))), array))


def plot_avg_velocity(average_velocity: np.ndarray, fname: str = "average_velocity.png"):
    ax = average_velocity.plot(figsize=(12, 8))
    ax.set_xlabel(r"$\rho$", fontsize=14)
    ax.set_ylabel("average velocity", fontsize=14)
    ax.set_title(r"Avg. velocity vs slowdown probability", fontsize=18)
    ax.legend(fontsize=14)
    plt.savefig(fname)
    plt.close()
