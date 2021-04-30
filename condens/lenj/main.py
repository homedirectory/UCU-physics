'''
Created on Apr 30, 2021

@author: jjnkn
'''

import numpy as np
from scipy.stats import maxwell
import plotly.express as px
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io


class Particle(object):
    '''
    classdocs
    '''

    def __init__(self, r, v):
        '''
        r : np.array(float) ---- coordinates vector
        v : np.array(float) ---- velocities vector
        '''
        self.r = r
        self.v = v
        

def distance(p1, p2):
    '''
    p1 : Particle
    p2 : Particle
    '''
    
    return np.sqrt(np.power(p1.r - p2.r, 2).sum()) 


def ljp(distance, eps, sigma):
    sd = sigma / distance
    u = 4 * eps * (np.power(sd, 12) - np.power(sd, 6))
    
    return u


def forces(pi, pj, eps, sigma):
    '''
    pi : Particle
    pj : Particle
    '''
    
    d = distance(pi, pj)
    d2 = np.power(d, -2)
    sd = sigma / d
    Fi = 4 * eps * ((-12 * np.power(sd, 12) * d2) + (6 * np.power(sd, 6) * d2)) * (pi.r - pj.r)
    Fj = -Fi
    
    return Fi, Fj

def vverlet(pi, pj, eps, sigma, delta_t, m):
    '''
    Velocity Verlet integration
    r : np.array() ---- particle coordinates
    v : np.array() ---- particle velocity
    F : np.array() ---- force
    '''
    Fi, Fj = forces(pi, pj, eps, sigma)
    # r(t + delta_t)
    pi.r = pi.r + (pi.v + (Fi / m) * (delta_t / 2)) * delta_t
    
    Fi_new, Fj_new = forces(pi, pj, eps, sigma)
    # v(t + delta_t)
    pi.v = pi.v + 0.5 * ((Fi/m) + (Fi_new/m)) * delta_t 
    

class MD(object):
    
    
    def __init__(self, sigma, m, eps, delta_t, size):
        self.sigma = sigma
        self.m = m
        self.eps = eps
        self.delta_t = delta_t
        self.size = size
        
        self.particles = None
    
    def setup(self):
        self.particles = [Particle(np.array([x, y, z]), np.array(maxwell.rvs(size=3))) for x in range(self.size) for y in range(self.size) for z in range(self.size)]
        
    def start(self, epochs, write_video=False, test=False):
        N = len(self.particles) if not test else 5
        r_c = np.power(2., 1./6.)
        # r_min = np.power(2., 1./6.) * self.sigma
        # U_r_min = ljp(r_min, self.eps, self.sigma)
        # U_r_c = 0.00001 * U_r_min
        c = 0
        
        for _ in tqdm(range(epochs)):
            for i in range(N):
                for j in range(i+1, N):
                    pi, pj = self.particles[i], self.particles[j]
                    r_ij = distance(pi, pj)
                    
                    if r_ij <= r_c:
                        vverlet(pi, pj, self.eps, self.sigma, self.delta_t, self.m)
            
            if write_video:
                self.__get_frame(c, self.size, test)
                c += 1
            
                    
    
    def __get_frame(self, c, size, test=False):
        dir = "figures-test" if test else "figures"
        xs = [p.r[0] % size for p in self.particles]
        # print(xs)
        ys = [p.r[1] % size for p in self.particles]
        zs = [p.r[2] % size for p in self.particles]
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_zlim(0, self.size)
        

        ax.scatter(xs, ys, zs)
        
        plt.savefig(f"{dir}/fig{c}.png")
        plt.close(fig)
        # plt.savefig(fig_img, format = 'png')
        # image = cv2.imread(f"{dir}/fig{c}.png")
        # nparr = np.frombuffer(fig_img, np.uint8)
        # frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # video.write(image)
        # return image
                    

























