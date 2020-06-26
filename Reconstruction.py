import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from scipy.optimize import least_squares
import cv2
import glob
import sys
import time
import os
import pickle
import open3d
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import FeatureMatching
import CameraMatrix
import BundleAdjustment
import DenseMatching

#%%
class Reconstruction:
    
    def __init__(self, img_adds, param_dict='default', output_folder=None):
        self.Imgs = []
        self.ImgsBW = []
        for file in img_adds:
            img = cv2.imread(file)[...,::-1]
            self.Imgs.append(img)
            self.ImgsBW.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)>2 else img)
        self.Imgs = np.array(self.Imgs)
        self.ImgsBW = np.array(self.ImgsBW)
        print(f'Number of Input Images: {self.Imgs.shape[0]}')
        if len(self.Imgs[0].shape) == 3:
            self.H, self.W, self.channels = self.Imgs[0].shape
        else:
            (self.H, self.W), self.channels = self.Imgs[0].shape, 1
        
        if param_dict == 'default':
            self.param_dict = {}
            self.param_dict['loop_F_mat'] = 200
            self.param_dict['thld_reprojection'] = 100
            self.param_dict['thld_F_inliers'] = 2
            self.param_dict['thld_zncc_seed'] = 0.88
            self.param_dict['thld_confidence'] = 0.01
            self.param_dict['w_seed'] = 5
            self.param_dict['thld_zncc_propagation'] = 0.68
            self.param_dict['w_propagation'] = 2
            self.param_dict['N_window'] = 2
            self.param_dict['epsilon_propagation'] = 1
            self.param_dict['thld_inliers_dense_matching'] = 4
            self.param_dict['thld_bundle_adjustment_loss'] = 1
        
        if os.path.exists(output_folder):
            t = time.ctime()
            output_folder = os.path.join(output_folder, t[4:7]+t[9]+t[11:13]+t[14:16]+t[17:19]+t[-4:])
            os.mkdir(output_folder)
            self.output_folder = output_folder
        else:
            raise ValueError("Output path specified does not exist")
    
    def convert(self, seconds): 
        seconds = seconds % (24 * 3600) 
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
          
        return "%d:%02d:%02d" % (hour, minutes, seconds) 
    
    def delete(self):
        
        if self.S1 != None:
            del self.S1
        if self.S2 != None:
            del self.S2
        if self.S4 != None:
            del self.S4
        if self.S5 != None:
            del self.S5
    
    def get_match_pair_list(self, i, j):
        
        points1 = self.PntsMatch[:,i]
        points2 = self.PntsMatch[:,j]
        true_index = (points1 != -1) & (points2 != -1)
        
        points1 = self.IntPnts[i][points1[true_index]]
        points2 = self.IntPnts[j][points2[true_index]]
        return points1, points2 
    
    def triangulate_n(self, indexes, K, R_t_Matrices):
        
        Pnts2D = []
        M_matrices = []
        for i, index in enumerate(indexes):
            if index != -1:
                p = self.IntPnts[i][index]
                Pnts2D.append(p)
                M_matrices.append(np.matmul(K, R_t_Matrices[i]))
        
        Q = np.zeros((3*len(Pnts2D), 4 + len(Pnts2D)))
        for i, (p, P) in enumerate(zip(Pnts2D, M_matrices)):
            Q[3*i : 3*i + 3, :4] = P
            Q[3*i : 3*i + 3, 4 + i] = -p
        
        U, S, Vt = np.linalg.svd(Q)
        X = Vt[-1, :4]
        X = X[:3] / X[3]
        return X
    
    def triangulate(self, p1, p2, K, P1, P2):
        
        Q = np.zeros((6, 6))
        Q[:3, :4] = np.matmul(K, P1)
        Q[3:, :4] = np.matmul(K, P2)
        Q[:3, 4] = -p1
        Q[3:, 5] = -p2
        
        U, S, Vt = np.linalg.svd(Q)
        X = Vt[-1, :4]
        X = X[:3] / X[3]
        return X
    
    def triangulate_three(self, p1, p2, p3, K, P1, P2, P3):
        
        Q = np.zeros((9, 7))
        
        Q[:3,:4] = np.matmul(K, P1); Q[3:6,:4] = np.matmul(K, P2); Q[6:,:4] = np.matmul(K, P3)
        Q[:3,4] = p1; Q[3:6,5] = p2; Q[6:,6] = p3
        
        U, S, Vt = np.linalg.svd(Q)
        X = Vt[-1,:4]
        X = X[:3] / X[-1]
        return X
    
    def plot_compare(self, Pnts1, Pnts2D1, Pnts2, Pnts2D2, P1, P2, P1_, P2_, i, j):
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        plt.title(f"3D plot comparision between Image 0 & {j}. Before & After Bundle Adjustment")
        plt.axis('off')
        
        ax = fig.add_subplot(2, 2, 1)
        ax.axis('off')
        ax.imshow(self.ImgsBW[i], cmap='gray')
        for p in Pnts2D1:
            ax.plot([p[0]], [p[1]], 'o')

        ax = fig.add_subplot(2, 2, 2)
        ax.axis('off')
        ax.imshow(self.ImgsBW[j], cmap='gray')
        for p in Pnts2D2:
            ax.plot([p[0]], [p[1]], 'o')
        
        C1 = -1*P1[:,-1]
        C2 = -1*P2[:,-1]
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        #ax._axis3don = False
        for p in Pnts1:
            ax.plot([p[0]], [p[1]], [p[2]], 'o')
        for p in Pnts1:
            ax.plot([C1[0], p[0]], [C1[1], p[1]], [C1[2], p[2]], '--.', alpha=0.3)
            ax.plot([C2[0], p[0]], [C2[1], p[1]], [C2[2], p[2]], '--.', alpha=0.3)
        
        C1 = -1*P1_[:,-1]
        C2 = -1*P2_[:,-1]
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        #ax._axis3don = False
        for p in Pnts2:
            ax.plot([p[0]], [p[1]], [p[2]], 'o')
        for p in Pnts2:
            ax.plot([C1[0], p[0]], [C1[1], p[1]], [C1[2], p[2]], '--.', alpha=0.3)
            ax.plot([C2[0], p[0]], [C2[1], p[1]], [C2[2], p[2]], '--.', alpha=0.3)
            
        plt.savefig(os.path.join(self.output_folder, f"3D_Compare_1{j+1}.png"))
            
    def most_common_points(self):
        best_matches = np.ones((self.Imgs.shape[0], 2))
        for i in range(self.Imgs.shape[0]):
            num = 0
            for j in range(self.Imgs.shape[0]):
                if i != j:
                    Pnts, _ = self.get_match_pair_list(i, j)
                    if Pnts.shape[0] > num:
                        if i == 0 or j in best_matches:
                            best_j = j
                            num = Pnts.shape[0]
            best_matches[i] = [best_j, i] if i != 0 else [i, best_j]
        return np.array(best_matches).astype(int)
        
    def discard_single_points(self):
        
        IntPnts = [ [] for i in range(self.Imgs.shape[0]) ]
        PntsMatch = np.ones(self.PntsMatch.shape) * -1
        
        for i, row in enumerate(self.PntsMatch):
            true_index = row != -1
            for j, index in enumerate(true_index):
                if index:
                    PntsMatch[i, j] = len(IntPnts[j])
                    x, y, _ = self.IntPnts[j][row[j]]
                    IntPnts[j].append([x, y, 1])
        
        IntPnts = [ np.array(arr) for arr in IntPnts ]
        return IntPnts, PntsMatch.astype(int)
    
    def discard_outliers(self, all_inliers):
        
        IntPnts = np.array([ np.zeros(self.IntPnts[i].shape[0]).astype(bool) for i in range(self.Imgs.shape[0]) ])
        for key, inliers in all_inliers.items():
            i, j = int(key[0]), int(key[1])
            Pnts1 = self.PntsMatch[:,i]
            Pnts2 = self.PntsMatch[:,j]
            true_index = (Pnts1 != -1) & (Pnts2 != -1)
            
            IntPnts[i][Pnts1[true_index][inliers]] = True
            IntPnts[j][Pnts2[true_index][inliers]] = True
        
        for i, pnts in enumerate(self.PntsMatch):
            for j, p in enumerate(pnts):
                if IntPnts[j][p] != True:
                    self.PntsMatch[i, j] = -1
        
        check = self.PntsMatch == -1
        for i, pnt_check in enumerate(check):
            if pnt_check.sum() == self.Imgs.shape[0]-1:
                j = np.where(pnt_check != -1)[0][0]
                IntPnts[j][self.PntsMatch[i, j]] = False
                self.PntsMatch[i] = -1
        
        self.PntsMatch = self.PntsMatch[np.where(np.sum(self.PntsMatch == -1, axis=1) != self.Imgs.shape[0])[0]]
        IntPntsNew = [ 0 for i in range(self.Imgs.shape[0]) ]
        for j, pnts in enumerate(self.PntsMatch.T):
            Pnts = []
            for i, p in enumerate(pnts):
                if p != -1:
                    self.PntsMatch[i, j] = len(Pnts)
                    Pnts.append(self.IntPnts[j][p])
            IntPntsNew[j] = np.array(Pnts)
        return self.PntsMatch, IntPntsNew
    
    def discard_outliers_dense_matching(self):
        
        NewMatchedPointPairs = {}
        for key, val in self.NewMatchedPointPairs.items():
            i, j = np.array(key.split('_'), dtype=int)
            P1, P2, K = self.R_t_matrices[i], self.R_t_matrices[j], self.K
            t, R = P2[:,-1] - P1[:,-1], np.matmul(np.linalg.inv(P1[:,:3]), P2[:,:3])
            t = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
            F = np.matmul(np.linalg.inv(K).T, np.matmul(t, np.matmul(R, np.linalg.inv(K))))
            
            Pnts1, Pnts2 = val
            inliers = np.matmul(Pnts2, F)
            inliers = np.sum(inliers * Pnts1, axis=1) / np.sqrt(np.sum(inliers[:,:2]**2, axis=1))
            inliers = np.abs(inliers) < self.param_dict['thld_inliers_dense_matching']
            
            Pnts1, Pnts2 = Pnts1[inliers], Pnts2[inliers]
            NewMatchedPointPairs[key] = [Pnts1, Pnts2]
        
        return NewMatchedPointPairs
    
    def select_sample_points(self):
        
        self.SampleIntPnts = {}
        for comb in np.array(list(combinations(range(self.Imgs.shape[0]), 2)), dtype=int):
            i, j = int(comb[0]), int(comb[1])
            Pnts1, Pnts2 = self.get_match_pair_list(i, j)
            
            sample = np.random.choice(list(range(Pnts1.shape[0])), min(6, Pnts1.shape[0]), replace=False)
            if len(sample):
                Pnts1 = Pnts1[sample]
                Pnts2 = Pnts2[sample]
                self.SampleIntPnts[str(i)+str(j)] = [Pnts1, Pnts2]
    
    def F_matrix_error(self, R_t_Matrices, K):
        
        Errors = 0
        for comb in np.array(list(combinations(range(R_t_Matrices.shape[0]), 2)), dtype=int):
            i, j = comb
            Pnts1, Pnts2 = self.get_match_pair_list(i, j)
            
            R_, R__ = R_t_Matrices[i, :, :3], R_t_Matrices[j, :, :3]
            t_, t__ = R_t_Matrices[i, :, -1], R_t_Matrices[j, :, -1]
            
            R = np.matmul(np.linalg.inv(R_), R__)
            t = t__ - t_; t = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
            F = np.matmul(np.linalg.inv(K).T, np.matmul(t, np.matmul(R, np.linalg.inv(K))))  
            
            Errors += (np.sum(np.matmul(Pnts2, F) * Pnts1, axis=1)**2).mean()
        return Errors
    
    def triangulate_least_squares(self, Pnts, K, Ps):
        # Pnts.shape = n, 2
            
        def fun(x):
            X, Y, Z = x
            PntsErr = []
            for i in range(Pnts.shape[0]):
                pnt = np.matmul(K, np.matmul(Ps[i], np.array([[X], [Y], [Z], [1]])))
                pnt = pnt[:-1] / pnt[-1]
                pnt = pnt.ravel() - Pnts[i].ravel()
                PntsErr.append(list(pnt))
            return np.array(PntsErr).ravel()
        
        result = least_squares(fun, np.random.random(3), method='trf')
        return result.x, result.cost
    
    def reprojection_error(self, R_t_matrices):
        
        Pnts3D = []
        Error = []
        for row in self.PntsMatch:
            index = row != -1
            Pnts2D = []
            for i, val in enumerate(index):
                if val:
                    Pnts2D.append(list(self.IntPnts[i][row[i]][:2]))
            R_t_mats = R_t_matrices[index]
            
            X, _ = self.triangulate_least_squares(np.array(Pnts2D), self.K, R_t_mats)
            Pnts3D.append(list(X))
            
            for i, val in enumerate(index):
                if val:
                    pnt = self.IntPnts[i][row[i]][:2]
                    cal_pnt = np.matmul(self.K, np.matmul(R_t_matrices[i], list(X)+[1]))
                    cal_pnt = cal_pnt[:2] / cal_pnt[-1]
                    err = cal_pnt - pnt
                    Error.append(err[0]); Error.append(err[1])
        
        return np.array(Pnts3D), np.mean(np.abs(np.array(Error)))
    
    def Step1(self):
        
        start = time.time()
        print()
        print("Step 1: Finding Keypoints/Interest Points")
        #Step 1: Feature Point Detection, Descriptor, Matching
        self.S1 = FeatureMatching.FeatureMatching()
        self.IntPnts, self.PntsMatch, _ = self.S1.FeatureDetectionDescriptor(self.ImgsBW)
        self.IntPnts, self.PntsMatch = self.discard_single_points()
        print("Step 1 Completed in", self.convert(time.time() - start), "time")
        
        if self.output_folder != None:
            self.Check1()
    
    def Check1(self):
        
        x = self.Imgs.shape[0] if self.Imgs.shape[0]<3 else 3
        y = self.Imgs.shape[0]//3 if not self.Imgs.shape[0]%3 else 1 + self.Imgs.shape[0]//3
        
        fig = plt.figure(figsize=(y*5, x*5))
        plt.title("All Images with Detected Interest Points")
        plt.axis('off')
        for i, img in enumerate(self.Imgs):
            ax = fig.add_subplot(x, y, i+1)
            ax.axis('off')
            ax.imshow(img)
            ax.plot(self.IntPnts[i][:,0], self.IntPnts[i][:,1], '.')
        plt.savefig(os.path.join(self.output_folder, "IntPnts.png"))
        plt.show()
    
    def Step2_(self):
        
        #Step 2: Calculate F-mat, E-mat, Camera Matrix
        #Assuming camera position for image indexed 0 be [0, 0, 0] in world coordinate
        start = time.time()
        print()
        print('Step 2: Calculating Camera Matrices')
        
        self.R_t_matrices = np.zeros((self.Imgs.shape[0], 3, 4))
        self.R_t_matrices[0] = np.hstack((np.eye(3), [[0], [0], [0]]))
        Pj_from_P_Pi = lambda P, Pi: np.hstack((np.matmul(Pi[:,:3], P[:,:3]), P[:,-1:] + Pi[:,-1:] / np.linalg.norm(P[:,-1:] + Pi[:,-1:])))
        
        done = np.array([1] + [0 for i in range(self.Imgs.shape[0]-1)])
        inlier_pairs = {}
        pairs = self.most_common_points()
        for i, j in zip(pairs[:,0], pairs[:,1]):
            if done[j]:
                continue
            done[j] = 1
            print(f"Finding Matrices for pair {i+1}<-->{j+1}")
            points1, points2 = self.get_match_pair_list(i, j)
            self.S2 = CameraMatrix.FundamentalCameraMatrix()
            self.S2.ImgPairDetails(points1, points2, self.Imgs[i], self.Imgs[j])
            _, _, _, _, _, self.K, P, _, err, inliers = self.S2.Calculate_F_E_C_Matrix(loop_F_mat=self.param_dict['loop_F_mat'],
                                                                    thld_reprojection=self.param_dict['thld_reprojection'],
                                                                    thld_F_inliers=self.param_dict['thld_F_inliers'], 
                                                                    plot_epipolar_line=True)
            inlier_pairs[str(i)+str(j)] = inliers
            self.R_t_matrices[j] = Pj_from_P_Pi(P, self.R_t_matrices[i])
            print(f'>>> Camera Matrix for pair {i+1}<-->{j+1} found with Reprojection Error {err}')
            print()
        
        self.temp = inlier_pairs
        self.PntsMatch, self.IntPnts = self.discard_outliers(inlier_pairs)
        self.select_sample_points()
        print()
        print(f"Current number of Interest Points in Images: {[ Pnts.shape[0] for Pnts in self.IntPnts]}")
        print("Step 2 Completed in", self.convert(time.time() - start), "time")
        
        if self.output_folder != None:
            self.Check1()
            
    def Step3(self, PntsMatch, R_t_Matrices):
        
        #Step 3: Find approximate 3D points for 2D correspondences using triangulation
        IntPnts3DOld = []
        for row in PntsMatch:
            Choices = np.where(row != -1)[0]
            if Choices.shape[0] < 2:
                continue
            i, j = np.random.choice(Choices, 2, replace=False)
            pi, pj = self.IntPnts[i][row[i]], self.IntPnts[j][row[j]]
            pnt3D = self.triangulate(pi, pj, self.K, R_t_Matrices[i], R_t_Matrices[j])
            #pnt3D = self.triangulate_n(row, self.K, R_t_Matrices)
            IntPnts3DOld.append(pnt3D)
            
        return np.array(IntPnts3DOld)
    
    def Step3_(self, PntsMatch, R_t_Matrices):
        
        #Step 3: Finding approximate 3D points for 2D correspondences using least square method
        IntPnts3DOld = []; Cost = []
        for row in PntsMatch:
            index = np.where(row != -1)[0]
            
            def fun(x):
                X, Y, Z = x
                PntsErr = []
                for i in index:
                    pnt = np.matmul(self.K, np.matmul(R_t_Matrices[i], np.array([[X], [Y], [Z], [1]])))
                    pnt = pnt[:-1] / pnt[-1]
                    pnt = pnt.ravel() - self.IntPnts[i][row[i], :2].ravel()
                    PntsErr.append(list(pnt))
                PntsErr = np.array(PntsErr).ravel()
                return PntsErr
            
            result = least_squares(fun, np.random.random(3), method='trf')
            IntPnts3DOld.append(result.x)
            Cost.append(result.cost)
            
        return np.array(IntPnts3DOld), np.array(Cost)
    
    def Step4(self):
        
        #Step 4: Perform Bundle Adjustmen on IntPnts3D & R_t_matrices
        start = time.time()
        print()
        print("Step 4: Applying Bundle Adjustment to fine tune Camera Matrices & 3D point approximation")
        
        index = np.array(range(self.PntsMatch.shape[0])); np.random.shuffle(index)
        self.PntsMatch = self.PntsMatch[index];
        loop = self.PntsMatch.shape[0] / 500
        loop = int(loop) if loop % 1 == 0 else int(loop + 1)
        self.R_t_Matrices = self.R_t_matrices
        
        Error = self.F_matrix_error(self.R_t_matrices, self.K); 
        _, Repro_Loss = self.reprojection_error(self.R_t_matrices); ErrorCheck = Repro_Loss
        Loss = Error + Repro_Loss
        while(Loss > self.param_dict['thld_bundle_adjustment_loss'] or Error > 0.1):
            print(f"Loop starting with Epipolar Error: {Error} & Reprojection Error: {Repro_Loss}")
            for i in range(loop):
                if self.PntsMatch[i*500 : (i+1)*500].shape[0] <= self.Imgs.shape[0]*7:
                    continue
                
                IntPnts3DOld, _ = self.Step3_(self.PntsMatch[i*500 : (i+1)*500], self.R_t_Matrices)
                #IntPnts3DOld = self.Step3(self.PntsMatch[i*500 : (i+1)*500], self.R_t_Matrices)
                self.S4 = BundleAdjustment.BundleAdjustment(self.PntsMatch[i*500 : (i+1)*500].T, self.R_t_Matrices, IntPnts3DOld, self.IntPnts, self.K, self.param_dict['thld_bundle_adjustment_loss'])
                R_t_Matrices, _ = self.S4.LeastSquaresCalculator()
                
                err = self.F_matrix_error(R_t_Matrices, self.K)
                _, loss = self.reprojection_error(R_t_Matrices)
                print(f"Current Iteration Error in Epipolar Lines: {err} and Error in Reprojection: {loss}")
                
                if err < 0.1 and loss < Repro_Loss:
                    print("Reprojection Error reduced in this iteration")
                    Error = err; Repro_Loss = loss; Loss = err + loss
                    self.R_t_Matrices = R_t_Matrices
                    
                print(f"Current minimum Epipolar Error: {Error} and Reprojection Error: {Repro_Loss}")
                
                if Error < 0.1 and Loss < self.param_dict['thld_bundle_adjustment_loss']:
                    break
            
            if Repro_Loss == ErrorCheck:
                self.Step2_()
                self.Step4()
            ErrorCheck = Repro_Loss
                
        print("Step 4 Completed in", self.convert(time.time() - start), "time")
        
        if self.output_folder != None:
            self.Check2()
            #self.Check2_(self.R_t_matrices); self.Check2_(self.R_t_Matrices)
    
    def Check2_(self, R_t_Matrices):
        
        Pnts3D = []; Colors = []
        for row in self.PntsMatch:
            index = row != -1
            Pnts2D = []; done = 0
            for i, val in enumerate(index):
                if val:
                    p = self.IntPnts[i][row[i]][:2]
                    Pnts2D.append(list(p))
                    if not done:
                        col = self.Imgs[i, int(p[1]), int(p[0])] / 255
                        done = 1
            R_t_mats = R_t_Matrices[index]
            
            X, _ = self.triangulate_least_squares(np.array(Pnts2D), self.K, R_t_mats)
            Pnts3D.append(list(X)); Colors.append(list(col))
        Pnts3D, Colors = np.array(Pnts3D), np.array(Colors)
        Pnts3D[:, 1:] *= -1
        
        mini = np.mean(Pnts3D, axis=0) - 2*np.std(Pnts3D, axis=0)
        maxi = np.mean(Pnts3D, axis=0) + 2*np.std(Pnts3D, axis=0)
        index = ((Pnts3D > mini.reshape((1, -1))).sum(axis=1) == 3) & ((Pnts3D < maxi.reshape((1, -1))).sum(axis=1) == 3) 
        Pnts3D = Pnts3D[index]; Colors = Colors[index]
        
        PointCloud = open3d.geometry.PointCloud()
        Points = open3d.utility.Vector3dVector(Pnts3D)
        Color = open3d.utility.Vector3dVector(Colors)
        PointCloud.points = Points
        PointCloud.colors = Color
        open3d.visualization.draw_geometries([PointCloud])        
            
    def Check2(self):
        
        for comb, Pnts2D in self.SampleIntPnts.items():
            i, j = int(comb[0]), int(comb[1])
            P1, P1_ = self.R_t_matrices[i], self.R_t_Matrices[i]
            P2, P2_ = self.R_t_matrices[j], self.R_t_Matrices[j]
            
            Pnts2D, Pnts2D_ = Pnts2D
            Pnts, Pnts_ = [], []
            for p1, p2 in zip(Pnts2D, Pnts2D_):
                X, _ = self.triangulate_least_squares(np.stack([p1[:2], p2[:2]]), self.K, np.stack([P1, P2]))
                X_, _ = self.triangulate_least_squares(np.stack([p1[:2], p2[:2]]), self.K, np.stack([P1_, P2_]))
                
                Pnts.append(X); Pnts_.append(X_)
                
            self.plot_compare(np.stack(Pnts), Pnts2D, np.stack(Pnts_), Pnts2D_, P1, P2, P1_, P2_, i, j)
    
    def Step5(self):
        
        #Step 5: Use Dense Matching to generate more points for point cloud
        start = time.time()
        print()
        print("Applying Dense Matching to generate more 2D correspondences.")
        done = []
        self.NewMatchedPointPairs = {}
        for i in range(self.Imgs.shape[0]):
            for j in range(self.Imgs.shape[0]):
                if i != j and [i, j] not in done and [j, i] not in done:
                    done.append([i, j])
                    img1 = self.Imgs[i] if self.channels==1 else cv2.cvtColor(self.Imgs[i], cv2.COLOR_BGR2GRAY)
                    img2 = self.Imgs[j] if self.channels==1 else cv2.cvtColor(self.Imgs[j], cv2.COLOR_BGR2GRAY)
                    
                    self.S5 = DenseMatching.DenseMatching(img1, img2, self.IntPnts[i][:,:2], self.IntPnts[j][:,:2], i, j)
                    #IntPnts1, IntPnts2 of form (y, x)
                    IntPnts1, IntPnts2 = self.S5.PerformDenseMatching(self.param_dict['thld_zncc_seed'], self.param_dict['w_seed'], 
                                                                 self.param_dict['thld_zncc_propagation'], 
                                                                 self.param_dict['thld_confidence'], self.param_dict['w_propagation'], 
                                                                 self.param_dict['N_window'], self.param_dict['epsilon_propagation'])
                    self.NewMatchedPointPairs[f'{i}_{j}'] = [np.hstack((IntPnts1, np.ones((IntPnts1.shape[0], 1)))).astype(int), np.hstack((IntPnts2, np.ones((IntPnts2.shape[0], 1)))).astype(int)]
                    print()
        temp = [ val[0].shape[0] for key, val in self.NewMatchedPointPairs.items()]
        print("Discarding the outlier points")
        print(f"Number of points in the list before discarding {temp}")
        self.NewMatchedPointPairs = self.discard_outliers_dense_matching()
        temp = [ val[0].shape[0] for key, val in self.NewMatchedPointPairs.items()]
        print(f"Number of points in the list before discarding {temp}")
        print("Step 5 Completed in", self.convert(time.time() - start), "time")
        
        if self.output_folder != None:
            self.Check3()
            self.Check3_()
    
    def Check3(self):
        
        for key, val in self.NewMatchedPointPairs.items():
            a, b = np.array(key.split('_'), dtype=int)
            Old1, Old2 = self.get_match_pair_list(a, b)
            New1, New2 = val[0], val[1]
            
            fig = plt.figure(figsize=(10, 10))
            plt.title(f"Resulting Matched Points after Dense Matching on Images {a+1}-{b+1}")
            plt.axis('off')
            for i, pair in enumerate([Old1.astype(int), Old2.astype(int), New1.astype(int), New2.astype(int)]):
                ax = fig.add_subplot(3, 2, 1+i)
                img = np.zeros((self.Imgs[0].shape[0], self.Imgs[0].shape[1]))
                img[pair[:,1], pair[:,0]] = 1
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
            ax = fig.add_subplot(3, 2, 5)
            ax.imshow(self.Imgs[a])
            ax.axis('off')
            
            ax = fig.add_subplot(3, 2, 6)
            ax.imshow(self.Imgs[b])
            ax.axis('off')
            
            plt.savefig(os.path.join(self.output_folder, "DenseMatch_"+key+".png"))
    
    def Check3_(self):
    
        for key, val in self.NewMatchedPointPairs.items():
            i, j = np.array(key.split('_'), dtype=int)
            P1, P2 = self.R_t_Matrices[i], self.R_t_Matrices[j]
            t1, t2 = -1*P1[:,-1], -1*P2[:,-1]
            
            index = np.random.choice(range(val[0].shape[0]), min(val[0].shape[0], 6), replace=False)
            if index.shape[0] == 0:
                continue
            val = [val[0][index], val[1][index]]
            
            Pnts = []
            for p1, p2 in zip(val[0], val[1]):
                pnts = np.array([[p1[1], p1[0]], [p2[1], p2[0]]])
                p, _ = self.triangulate_least_squares(pnts, self.K, np.stack([P1, P2]))
                Pnts.append(p)
            
            fig = plt.figure(figsize=(10, 5))
            
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(self.Imgs[i])
            for p in val[0]:
                ax.plot([p[1]], [p[0]], 'o')
            
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            for p in Pnts:
                ax.plot([p[0]], [p[1]], [p[2]], 'o')
            for p in Pnts:
                ax.plot([p[0], t1[0]], [p[1], t1[1]], [p[2], t1[2]], '--.', alpha=0.3)
            for p in Pnts:
                ax.plot([p[0], t2[0]], [p[1], t2[1]], [p[2], t2[2]], '--.', alpha=0.3)
    
    def Step6(self):
        
        #Step 6: Triangulating NewMatchedPointPairs to get 3D coordinates for Point Cloud
        start = time.time()
        print()
        print("Step 6: 3D points will be calculated using 2D correspondences from Dense Matching")
        self.New3DPnts = []
        self.Color3DPnts = []
        total = np.array([ pair[0].shape[0] for i, pair in self.NewMatchedPointPairs.items() ]).sum()
        pbar = tqdm(self.NewMatchedPointPairs.items(), total=total, file=sys.stdout)
        for key, val in pbar:
            i, j = list(map(int, key.split('_')))
            IntPnts1, IntPnts2 = val
            P1, P2 = self.R_t_Matrices[i], self.R_t_Matrices[j]
            for p1, p2 in zip(IntPnts1, IntPnts2):
                pbar.update(1)
                p1 = np.array([p1[0], p1[1], p1[2]])
                p2 = np.array([p2[0], p2[1], p2[2]])
                #pnt3D, _ = self.triangulate_least_squares(np.stack([p1[:2], p2[:2]]), self.K, np.stack([P1, P2]))
                pnt3D = self.triangulate(p1, p2, self.K, P1, P2)
                self.New3DPnts.append(pnt3D)
                
                clr3D = self.Imgs[i][p1[1], p1[0]]
                if type(clr3D) == int or type(clr3D) == float:
                    self.Color3DPnts.append([clr3D]*3)
                else:
                    self.Color3DPnts.append(clr3D)
        
        self.New3DPnts = np.array(self.New3DPnts)
        self.Color3DPnts = np.array(self.Color3DPnts) / 255
        
        mini = self.New3DPnts.mean(axis=0) - 2*self.New3DPnts.std(axis=0)
        maxi = self.New3DPnts.mean(axis=0) + 2*self.New3DPnts.std(axis=0)
        index = np.sum((self.New3DPnts >= mini) & (self.New3DPnts <= maxi), axis=1) == 3
        self.New3DPnts, self.Color3DPnts = self.New3DPnts[index], self.Color3DPnts[index]
        
        self.delete()
        print("Step 6 Completed in", self.convert(time.time() - start), "time")
    
    def Step7(self):
        
        #Step 7: Visualization
        fig = plt.figure(figsize=(30, 30))
        ax = fig.gca(projection='3d')
        ax.scatter(self.New3DPnts[:,0], self.New3DPnts[:,1], self.New3DPnts[:,2], c=self.Color3DPnts)
        for P2 in self.R_t_Matrices:
            t = P2[:,-1]
            ax.plot([t[0]], [t[1]], [t[2]], 'o')
    
    def Step7_(self):
        
        #Step 7: Visualisation
        PointCloud = open3d.geometry.PointCloud()
        #self.New3DPnts[:,1] = self.New3DPnts[:,1] * -1
        Points = open3d.utility.Vector3dVector(self.New3DPnts)
        #self.New3DPnts[:,1] = self.New3DPnts[:,1] * -1
        Color = open3d.utility.Vector3dVector(self.Color3DPnts)
        PointCloud.points = Points
        PointCloud.colors = Color
        open3d.visualization.draw_geometries([PointCloud])
    
    def Reconstruct(self):
        
        self.Step1()
        self.Step2_()
        self.Step3()
        self.Step4()
        self.Step5()
        self.Step6()
        self.Step7_()

#%%
Construct = Reconstruction(glob.glob('Data/ImageSet/Table/*jpg'), output_folder="Data/Output")
#Construct.Reconstruct()
#%%
Construct.Step1()
Construct.Step2_()
Construct.Step4()
Construct.Step5()
Construct.Step6()
Construct.Step7_()
#%%
file = open("PCL.txt", "w")
file.write("ply\n")
file.write("format ascii 1.0\n")
file.write(f"element vertex {Construct.New3DPnts.shape[0]}\n")
file.write("property double x\nproperty double y\nproperty double z\n")
file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
file.write("element face 1\n")
file.write("property list uint8 int32 vertex_indices\n")
file.write("end_header\n")
for p, c in zip((Construct.New3DPnts*10000).astype(np.float32), (Construct.Color3DPnts*255).astype(int)):
    file.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
file.write("3 0 1 2")
file.close()