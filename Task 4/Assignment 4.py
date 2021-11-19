import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio
import os
import time

class ShellingSegregation:
    def __init__(self):
        self.lattice_list = []
    def simulate(self, L, R, B, j_r, j_b, k):
        lattice = np.zeros(L*L)
        self.Js = []
        lattice[0:R] = 1
        lattice[R:B+R]= 2
        np.random.shuffle(lattice)
        lattice = lattice.reshape((L,L))
        self.lattice_list.append(lattice.copy())
        J = np.zeros((L,L))
        for i in range(L):
            for j in range(L):
                if lattice[i,j] != 0:
                    neighborhood = self.neighbors(lattice,i,j,k)
                    num_neighbors = np.sum(neighborhood>0)-1
                    if num_neighbors == 0:
                        J[i,j] = 1 # isolated agent is happy
                    else:
                        same_type = np.sum(neighborhood==lattice[i,j])-1
                        J[i,j] = same_type/num_neighbors
        self.Js.append(J)
        iteration = 0
        while np.logical_and(iteration<1000,np.any(np.logical_or(np.logical_and(J<j_r,lattice==1),np.logical_and(J<j_b,lattice==2)))):
            if iteration%100==0:
                print(iteration)
            iteration+=1
            unhappy_red_map = np.logical_and(lattice==1, J<j_r)
            unhappy_blue_map = np.logical_and(lattice==2, J<j_b)
            empty_map = lattice==0
            possible_map = np.logical_or(np.logical_or(unhappy_blue_map,unhappy_red_map),empty_map)
            values = lattice[possible_map]
            np.random.shuffle(values)
            lattice[possible_map] = values
            self.lattice_list.append(lattice.copy())
            average_happiness = np.mean(J[lattice>0])
            self.Js.append(J)
            
            J = np.zeros((L,L))
            for i in range(L):
                for j in range(L):
                    if lattice[i,j] != 0:
                        neighborhood = self.neighbors(lattice,i,j,k)
                        num_neighbors = np.sum(neighborhood>0)-1
                        if num_neighbors == 0:
                            J[i,j] = 1 # isolated agent is happy
                        else:
                            same_type = np.sum(neighborhood==lattice[i,j])-1
                            J[i,j] = same_type/num_neighbors
            
        return iteration, average_happiness
    def neighbors(self, arr, x, y, k):
        ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
        n = 2*k+1
        arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
        return arr[:n,:n]

L = 100
R = 4000
B = 4000
jr = 0.7
jb = 0.7
k=1

t1 = time.time()
segregation = ShellingSegregation()
iters, arrays_list = segregation.simulate(L, R, B, jr,jb,k)
print(time.time()-t1)
print(iters)

def make_gif(fname,array_list,L,R,B,jr,jb,k):
    cmap = colors.ListedColormap(['white','red','blue'])
    filenames = []
    images = []
    for i in range(len(array_list)):
            plt.figure(figsize=(6, 6))
            plt.imshow(array_list[i], cmap=cmap, vmin=0, vmax=2)
            plt.title('Iteration number {}'.format(i))
            plt.text(100,20,'L={}\n R={}\n B={}\n jr={}\n jb={}\n k={}'.format(L,R,B,jr,jb,k))
            image_name = 'graph{}.png'.format(i)
            plt.axis('off')
            plt.savefig(image_name)
            filenames.append(image_name)
            images.append(imageio.imread(image_name))
            plt.close()
    imageio.mimsave(fname, images, fps=12)
    for i in filenames:
        os.remove(i)


make_gif('baseline.gif',segregation.lattice_list,L,R,B,jr,jb,k)
