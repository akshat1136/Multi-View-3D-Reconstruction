import numpy as np
import math
from tqdm import tqdm
from itertools import count
import sys
import heapq
import warnings

warnings.filterwarnings('ignore')

class DenseMatching:
    
    def __init__(self, img1, img2, points1, points2, i, j):
        self.Imgs = [img1, img2]
        self.points1 = points1.astype(int)
        self.points2 = points2.astype(int)
        self.H, self.W = self.Imgs[0].shape[:2]
        self.img1_check = np.zeros((self.H, self.W))
        self.img2_check = np.zeros((self.H, self.W))
        self.i = i
        self.j = j
        self.IntPnts = -1 * np.ones((self.H, self.W, 2))
        self.tiebreaker = count()
        
    def ZNCC(self, x, x_):
        
        if x.shape != x_.shape:
            return 0
        
        m_x = x - np.mean(x)
        m_x_ = x_ - np.mean(x_)
        
        num = np.sum(m_x * m_x_)
        den = math.sqrt(np.sum(m_x**2) * np.sum(m_x_**2))
        return num / den
    
    def Seed_Selection(self, w, thld_zncc_seed):
        self.Seeds = []
        pbar = tqdm(total=self.points1.shape[0] + self.points2.shape[0], file=sys.stdout)
        check_array = np.ones((2, self.H, self.W, 2)) * -1
        
        for i, p1 in enumerate(self.points1):
            pbar.update(1)
            pbar.set_description(f'Working on image {self.i} and {self.j}, Seed Selection Progress')
            x, y = p1
            x = self.Imgs[0][int(max(0, y-w)) : int(min(self.H-1, y+w+1)), int(max(0, x-w)) : int(min(self.W-1, x+w+1))]
            zncc_best = 1
            for j, p2 in enumerate(self.points2):
                x_, y_ = p2
                x_ = self.Imgs[1][int(max(0, y_-w)) : int(min(self.H-1, y_+w+1)), int(max(0, x_-w)) : int(min(self.W-1, x_+w+1))]
                zncc = self.ZNCC(x, x_)
                
                if zncc > thld_zncc_seed and zncc < zncc_best:
                    zncc_best = zncc
                    check_array[0, p1[1], p1[0]] = [i, j]
                    heapq.heappush(self.Seeds, (1-zncc, next(self.tiebreaker), [p1, p2]))
                    self.IntPnts[p1[1],p1[0]] = [p2[1],p2[0]]
                    self.img1_check[p1[1],p1[0]] = 1
                    self.img2_check[p2[1],p2[0]] = 1
        
        for j, p2 in enumerate(self.points2):
            pbar.update(1)
            pbar.set_description(f'Working on image {self.i} and {self.j}, Seed Selection Progress')
            x_, y_ = p2
            x_ = self.Imgs[1][int(max(0, y_-w)) : int(min(self.H-1, y_+w+1)), int(max(0, x_-w)) : int(min(self.W-1, x_+w+1))]
            zncc_best = 1
            for i, p1 in enumerate(self.points1):
                x, y = p1
                x = self.Imgs[0][int(max(0, y-w)) : int(min(self.H-1, y+w+1)), int(max(0, x-w)) : int(min(self.W-1, x+w+1))]
                zncc = self.ZNCC(x, x_)
                
                if zncc > thld_zncc_seed and zncc < zncc_best:
                    zncc_best = zncc
                    check_array[1, p2[1], p2[0]] = [i, j]
                    heapq.heappush(self.Seeds, (1-zncc, next(self.tiebreaker), [p1, p2]))
                    self.IntPnts[p1[1],p1[0]] = [p2[1],p2[0]]
                    self.img1_check[p1[1],p1[0]] = 1
                    self.img2_check[p2[1],p2[0]] = 1
        pbar.close()
        
        yies, xes = np.where(check_array[0,:,:,0] != -1)
        for y, x in zip(yies, xes):
            i, j = check_array[0, y, x]
            x_, y_ = self.points2[int(j)]
            if list(check_array[1, y_, x_]) == [int(i), int(j)] and not self.img1_check[y, x] and not self.img2_check[y_, x_]:
                self.IntPnts[y, x] = [y_, x_]
                self.img1_check[y, x] = 1
                self.img2_check[y_, x_] = 1
                check_array[0, y, x] = check_array[1, y_, x_] = -1
        '''
        yies, xes = np.where(check_array[1,:,:,0] != -1)
        for y_, x_ in zip(yies, xes):
            i, j = check_array[1, y_, x_]
            y, x = self.points1[i]
            if list(check_array[0, y, x]) == [i, j] and not self.img1_check[y, x] and not self.img2_check[y_, x_]:
                self.IntPnts[y, x] = [y_, x_]
                self.img1_check[y, x] = 1
                self.img2_check[y_, x_] = 1
        '''
        
    def Neighbor(self, X, N):
        x, y = X
        neigh = np.array(np.meshgrid(range(max(0, x-N), min(self.W-1, x+N+1)), range(max(0, y-N), min(self.H-1, y+N+1)))).T.reshape((-1, 2))
        return neigh
    
    def Neighbors(self, X, X_, N, epsilon):
        
        neigh_X = self.Neighbor(X, N)
        neigh_X_ = self.Neighbor(X_, N)
        X_neigh_X_ = []
        
        for U in neigh_X:
            for U_ in neigh_X_:
                if (U==X).all() and (U_==X_).all():
                    continue
                if np.linalg.norm((U - U_) - (X - X_), ord=np.inf) < epsilon:
                    X_neigh_X_.append([U, U_])
        
        return X_neigh_X_
    
    def confidence_score(self, pt, p, img):
        deltas = np.array([ [1, 0], [0, 1], [-1, 0], [0, -1] ])
        
        def fun(pt):
            pts = []
            for p in deltas:
                p = p + pt
                if p[0]>=0 and p[1]>=0 and p[0]<self.W and p[1]<self.H:
                    pts.append(p)
            return np.array(pts)
        
        ps = fun(p)
        I = self.Imgs[img]
        confidence = max([ I[pt[1],pt[0]] - I[p[1],p[0]] for pt in ps]) / 255
        return confidence
    
    def convert_IntPnts_form(self):
        IntPnts1 = []
        IntPnts2 = []
        Y, X = np.where(self.IntPnts[:,:,0] != -1)
        for y, x in zip(Y, X):
            IntPnts1.append([y, x])
            y, x = self.IntPnts[int(y), int(x)]
            IntPnts2.append([y, x])
        self.IntPnts = np.array([IntPnts1, IntPnts2]).astype(int)
    
    def Propagation(self, thld_zncc_propagation=0.5, thld_confidence=0.01, w=2, N_window=2, epsilon=1):
        
        pbar = tqdm(range(self.H*self.W), file=sys.stdout)
        for i in pbar:
            pbar.set_description(f'Working on image {self.i} and {self.j}, Propagating on Seeds. Current Length of Seeds = {len(self.Seeds)}')
            
            _, _, (X, X_) = heapq.heappop(self.Seeds)
            neighbors = self.Neighbors(X, X_, N_window, epsilon)
            
            Local = []
            for neigh in neighbors:
                u, u_ = neigh
                if self.confidence_score(u, X, 0)>thld_confidence and self.confidence_score(u_, X_, 1)>thld_confidence:
                    x, y = u
                    x = self.Imgs[0][max(0, y-w) : min(self.H-1, y+w+1), max(0, x-w) : min(self.W-1, x+w+1)]
                    x_, y_ = u_
                    x_ = self.Imgs[1][max(0, y_-w) : min(self.H-1, y_+w+1), max(0, x_-w) : min(self.W-1, x_+w+1)]
                    zncc = self.ZNCC(x, x_)
                    if zncc > thld_zncc_propagation:
                        Local.append([u, u_, zncc])
            
            if len(Local) != 0:
                Local = np.array(Local)
                Local = Local[ np.argsort(Local[:,-1])[::-1] ]
                for pnt in Local:
                    u, u_, zncc = pnt
                    if not self.img1_check[u[1], u[0]] and not self.img2_check[u_[1], u_[0]]:
                        self.IntPnts[u[1], u[0]] = [u_[1], u_[0]]
                        heapq.heappush(self.Seeds, (1-zncc, next(self.tiebreaker), [u, u_]))
                        self.img1_check[u[1], u[0]] = 1
                        self.img2_check[u_[1], u_[0]] = 1
            
            if len(self.Seeds) == 0:
                break
        pbar.close()
        
        self.convert_IntPnts_form()
    
    def PerformDenseMatching(self, thld_zncc_seed, w_seed, thld_zncc_propagation, thld_confidence, w_propagation, N_window, epsilon_propagation):
        
        self.Seed_Selection(w_seed, thld_zncc_seed)
        self.Propagation(thld_zncc_propagation, thld_confidence, w_propagation, N_window, epsilon_propagation)
        return self.IntPnts[0], self.IntPnts[1]