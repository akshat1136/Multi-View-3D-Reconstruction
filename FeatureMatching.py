import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import scipy.signal
import scipy.linalg
from scipy import ndimage
import heapq
from tqdm import tqdm
from itertools import combinations
import sys
import math
from operator import itemgetter
import IntPntsDetectorDescriptor as itp

class FeatureMatching:
    
    def __init__(self):
        return
    
    def IPDetector(self, Img, alpha):
        keypts1, descriptors1 = itp.computeKeypointsAndDescriptors(Img, alpha, sigma=1.5, num_intervals=3, assumed_blur=0.5, image_border_width=5)
        
        ls=[]
        for i in range(len(keypts1)):
            ls.append(np.array([int(keypts1[i].pt[0]),int(keypts1[i].pt[1])]))
        keypts1 = np.array(ls)
        keypts1 = np.round(keypts1)
      
        for i in range(len(descriptors1)):
            for j in range(16):
                if(math.sqrt(np.sum(np.multiply(descriptors1[i][j*8:(j+1)*8],descriptors1[i][j*8:(j+1)*8]))) == 0.0):
                    continue
                descriptors1[i][j*8:(j+1)*8] = descriptors1[i][j*8:(j+1)*8]/math.sqrt(np.sum(np.multiply(descriptors1[i][j*8:(j+1)*8],descriptors1[i][j*8:(j+1)*8])))
    
        return keypts1, descriptors1
    
    def FeatureExtraction_and_Detection(self, Imgs):
        self.Imgs = Imgs
        keypts = list([i for i in range(len(Imgs))])
        descriptors = list([i for i in range(len(Imgs))])
        #feature detection
        for i in range(len(Imgs)):
            print(f'Keypoint & Feature Descriptor Extraction for Image {i+1}')
            keypts[i], descriptors[i] = self.IPDetector(Imgs[i], i+1)
        self.keypts = keypts
        self.descriptors = np.array(descriptors)
    
    '''

        Des1, Des2 = self.descriptors[a], self.descriptors[b]
        
        for i, (des1, p1) in enumerate(zip(self.descriptors[a], self.keypts[a])):
            pbar.update(1)
            pbar.set_description(f">>> Finding Matches between Images {a}<-->{b}, {at}/{int(total)} Completed")
            
            x, y = p1
            des1 = des1.reshape((1, -1))
            #min1 = min2 = 128.0
            
            corr_arr = np.sqrt(np.sum(np.abs(Des2 - des1)**2, axis=1))
            min1, min2 = np.argsort(corr_arr)[:2]
            index, min1, min2 = min1, corr_arr[min1], corr_arr[min2]
            
                        
            if min1 < 20 and min2 < 128.0 and min2*0.75 > min1:
                array[0, y, x] = [i, index, min1]
        
        for j, (des2, p2) in enumerate(zip(self.descriptors[b], self.keypts[b])):
            pbar.update(1)
            pbar.set_description(f">>> Finding Matches between Images {b}<-->{a}, {at}/{int(total)} Completed")
            
            x_, y_ = p2
            des2 = des2.reshape((1, -1))
            min1 = min2 = 128.0
            
            corr_arr = np.sqrt(np.sum(np.abs(Des1 - des2)**2, axis=1))
            min1, min2 = np.argsort(corr_arr)[:2]
            index, min1, min2 = min1, corr_arr[min1], corr_arr[min2]
            
                        
            if min1 < 20 and min2 < 128.0 and min2*0.75 > min1:
                array[1, y_, x_] = [index, j, min1]

    '''
    
    def Match_(self, a, b, at):
        
        N = len(self.Imgs)
        H, W = self.Imgs[0].shape
        array = np.ones((2, H, W, 3)) * -1
        total = math.factorial(N) / (math.factorial(2)*math.factorial(N-2))
        pbar = tqdm(total=self.descriptors[a].shape[0]+self.descriptors[b].shape[0], leave=False, file=sys.stdout)
        
        for i, (des1, p1) in enumerate(zip(self.descriptors[a], self.keypts[a])):
            pbar.update(1)
            pbar.set_description(f">>> Finding Matches between Images {a}<-->{b}, {at}/{int(total)} Completed")
            
            x, y = p1
            min1 = min2 = 128.0
            corr_arr = np.sqrt(np.sum(np.abs(self.descriptors[b] - des1.reshape((1, -1)))**2, axis=1))
            
            for j, (corr, p2) in enumerate(zip(corr_arr, self.keypts[b])):
                if corr < min2:
                    if corr < min1:
                        index = j
                        min2 = min1
                        min1 = corr
                    else:
                        min2 = corr
                        
            if min1 < 10 and min2 != 128.0 and min2*0.90 > min1:
                array[0, y, x] = [i, index, min1]
        
        for j, (des2, p2) in enumerate(zip(self.descriptors[b], self.keypts[b])):
            pbar.update(1)
            pbar.set_description(f">>> Finding Matches between Images {b}<-->{a}, {at}/{int(total)} Completed")
            
            x_, y_ = p2
            min1 = min2 = 128.0
            corr_arr = np.sqrt(np.sum(np.abs(self.descriptors[a] - des2.reshape((1, -1)))**2, axis=1))
            
            for i, (corr, p1) in enumerate(zip(corr_arr, self.keypts[a])):
                if corr < min2:
                    if corr < min1:
                        index = i
                        min2 = min1
                        min1 = corr
                    else:
                        min2 = corr
                        
            if min1 < 10 and min2 != 128.0 and min2*0.90 > min1:
                array[1, y_, x_] = [index, j, min1]
        pbar.close()
        
        Match = []
        Match_corr = []
        pbar = tqdm(total=(array[0,:,:,0]!=-1).sum(), leave=False, file=sys.stdout)
        
        yies, xes = np.where(array[0,:,:,0] != -1)
        for y, x in zip(yies, xes):
            pbar.update(1)
            pbar.set_description(f">>> Crosschecking Matched Keypoints for Image pair {a}<-->{b}, {at}/{int(total)} Completed")
            i, j, mini = array[0, y, x]
            i = int(i); j = int(j)
            x_, y_  = self.keypts[b][j]
            if list(array[1, y_, x_, :2]) == [i, j]:
                Match.append([i, j])
                Match_corr.append(mini)
                array[0, y, x] = array[1, y_, x_] = -1
        pbar.close()
        
        match_array = np.ones((2, H, W, 2))
        for match in Match:
            i, j = match
            x, y = self.keypts[a][i]
            x_, y_ = self.keypts[b][j]
            match_array[0, y, x] = match_array[1, y_, x_] = [i, j]
        
        return np.array(Match), np.array(Match_corr), match_array
    
    
    def MatchMultiple(self, L1, L2, comb):
        
        if L1.shape[0] == 0 or L2.shape[0] == 0:
            return np.zeros((0, len(comb)))
        
        c1, c2 = int(comb[0]), int(comb[-1])
        Match = []
        for a, bc in zip(L1[:,0], L1[:,1:]):
            x, y = self.keypts[c1][a]
            for bc_, d in zip(L2[:,:-1], L2[:,-1]):
                if (bc == bc_).all():
                    y_, x_ = self.keypts[c2][d]
                    i, j = self.Matches[str(c1)+str(c2)][-1][0, y, x]
                    if [i, j] == [a, d]:
                        Match.append([a] + [p for p in bc] + [d])
        return np.array(Match)
    
    def Matcher_(self, keypts):
        H, W = self.Imgs[0].shape
        N = len(self.Imgs)
        keypts = np.array(keypts)
        
        Comb = lambda r : [''.join(list(str(x)[1:-1].split(', '))) for x in combinations(list(range(N)),r)]
        Sort = lambda st : ''.join(sorted(st))
        self.Matches = dict()
        
        print(">>> Finding Matches among pair of Images")
        for at, comb in enumerate(Comb(2)):
            comb = Sort(comb)
            a, b = int(comb[0]), int(comb[-1])
            # match : (m, 2), match_array : (2, H, W, 2)
            match, match_corr, match_array = self.Match_(a, b, at)
            self.Matches[comb] = [match, match_corr, match_array]
        
        print(">>> Finding Matches across all Images")
        for r in range(3, N+1):
            for comb in Comb(r):
                comb = Sort(comb)
                a, b = self.Matches[comb[:-1]][0], self.Matches[comb[1:]][0]
                # match : (m, 2)
                match = self.MatchMultiple(a, b, comb)
                self.Matches[comb] = [match]
        
        print(">>> Merging the matches to obtain Num of World Points and their presence in each image.")
        Match = []
        check_array = np.zeros((N, H, W))
        for key in sorted(list(self.Matches.keys()), reverse=True, key=len):
            if len(key) == 2:
                continue
            match = self.Matches[key][0]
            if match.shape[0] == 0:
                continue
            key = np.array(list(key), dtype=int)
            for pnt in match:
                check = 0
                for i, p in zip(key, pnt):
                    x, y = self.keypts[i][p]
                    check = 1 if check or check_array[i, y, x] else 0
                
                if not check:
                    add = np.ones(N)*-1
                    add[key] = pnt
                    Match.append(add)
                    for i, p in zip(key, pnt):
                        x, y = self.keypts[i][p]
                        check_array[i, y, x] = 1
        
        from itertools import count
        tiebreaker = count()
        Priority = []
        for key, val in self.Matches.items():
            if len(key) != 2:
                continue
            for match, corr in zip(val[0], val[1]):
                heapq.heappush(Priority, (corr, next(tiebreaker), [int(key[0]), int(key[1]), match]))
        
        for i in range(len(Priority)):
            _, _, (a, b, match) = heapq.heappop(Priority)
            x, y = self.keypts[int(a)][match[0]]
            x_, y_ = self.keypts[int(b)][match[1]]
            if not check_array[a, y, x] and not check_array[b, y_, x_]:
                add = np.ones(N)*-1
                add[[int(a), int(b)]] = match
                Match.append(add)
                check_array[a, y, x] = check_array[b, y_, x_] = 1
        self.Match = Match
    
    def FeatureDetectionDescriptor(self, Imgs):
        self.FeatureExtraction_and_Detection(Imgs)
        print('Keypoint & Feature Descriptor Extraction Complete')
        self.Matcher_(self.keypts)
        
        keypts = []
        for pts in self.keypts:
            keypts.append(np.hstack((pts, np.ones((pts.shape[0], 1)))))
        self.keypts = keypts
        return self.keypts, np.stack(self.Match).astype(int), self.descriptors