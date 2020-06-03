import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import glob
import time
import os

import warnings
warnings.filterwarnings('ignore')

from Functions import FeatureMatching
from Functions import CameraMatrix
from Functions import BundleAdjustment
from Functions import DenseMatching

#%%
class Reconstruction:
    
    def __init__(self, img_adds, param_dict='default', output_folder=None):
        self.Imgs = []
        self.ImgsBW = []
        for file in img_adds:
            img = cv2.imread(file)
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
            self.param_dict['thld_F_inliers'] = 0.2
            self.param_dict['thld_zncc_seed'] = 0.8
            self.param_dict['thld_confidence'] = 0.01
            self.param_dict['w_seed'] = 5
            self.param_dict['thld_zncc_propagation'] = 0.5
            self.param_dict['w_propagation'] = 2
            self.param_dict['N_window'] = 2
            self.param_dict['epsilon_propagation'] = 1
        
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
    
    def get_match_pair_list(self, i, j):
        
        points1 = self.PntsMatch[:,i]
        points2 = self.PntsMatch[:,j]
        true_index = (points1 != -1) & (points2 != -1)
        
        points1 = self.IntPnts[i][points1[true_index]]
        points2 = self.IntPnts[j][points2[true_index]]
        return points1, points2 #np.hstack((points1, np.ones((points1.shape[0], 1)))), np.hstack((points2, np.ones((points2.shape[0], 1))))
    
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
    
    def plot_compare(self, Pnts1, Pnts2):
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax._axis3don = False
        for p in Pnts1:
            ax.plot([p[0]], [p[1]], [p[2]], 'o')
        
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax._axis3don = False
        for p in Pnts2:
            ax.plot([p[0]], [p[1]], [p[2]], 'o')
            
    def discard_single_points(self):
        
        IntPnts = []
        PntsMatch = np.ones(self.PntsMatch.shape) * -1
        for i, col in enumerate(self.PntsMatch.T):
            index = col != -1
            Pnts = []
            for j, (k, pnt) in enumerate(zip(index, col)):
                if k:
                    PntsMatch[j, i] = len(Pnts)
                    x, y, _ = self.IntPnts[i][j]
                    Pnts.append([x, y, 1])
            IntPnts.append(Pnts)
            
        return IntPnts, PntsMatch
    
    def Step1(self):
        
        start = time.time()
        print()
        print("Step 1: Finding Keypoints/Interest Points")
        #Step 1: Feature Point Detection, Descriptor, Matching
        S1 = FeatureMatching.FeatureMatching()
        self.IntPnts, self.PntsMatch, _ = S1.FeatureDetectionDescriptor(self.ImgsBW)
        self.discard_single_points()
        print("Step 1 Completed in", self.convert(time.time() - start), "time")
        
        if self.output_folder != None:
            self.Check1()
    
    def Check1(self):
        
        x = self.Imgs.shape[0] if self.Imgs.shape[0]<3 else 3
        y = self.Imgs.shape[0]//3 if not x%3 else 1 + self.Imgs.shape[0]//3
        
        fig = plt.figure(figsize=(y*5, x*5))
        plt.title("All Images with Detected Interest Points")
        plt.axis('off')
        for i, img in enumerate(self.Imgs):
            ax = fig.add_subplot(x, y, i+1)
            ax.axis('off')
            ax.imshow(img)
            ax.plot(self.IntPnts[i][:,0], self.IntPnts[i][:,1], '.')
        plt.show()
    
    def Step2(self):
        
        #Step 2: Calculate F-mat, E-mat, Camera Matrix
        #Assuming camera position for image indexed 0 be [0, 0, 0] in world coordinate
        start = time.time()
        print()
        print('Step 2: Calculating Camera Matrices')
        self.R_t_matrices = [np.hstack((np.eye(3), [[0], [0], [0]]))]
        for i in range(1, self.Imgs.shape[0]):
            print(f"Finding Matrices for pair 1<-->{i+1}")
            points1, points2 = self.get_match_pair_list(0, i)
            S2 = CameraMatrix.FundamentalCameraMatrix()
            S2.ImgPairDetails(points1, points2, self.Imgs[0], self.Imgs[i])
            _, _, _, _, _, self.K, P, _, err = S2.Calculate_F_E_C_Matrix(loop_F_mat=self.param_dict['loop_F_mat'],
                                                                    thld_reprojection=self.param_dict['thld_reprojection'],
                                                                    thld_F_inliers=self.param_dict['thld_F_inliers'],
                                                                    plot_epipolar_line=True)
            self.R_t_matrices.append(P)
            print(f'>>> Camera Matrix for pair 0<-->{i} found with Reprojection Error {err}')
            print()
        self.R_t_matrices = np.array(self.R_t_matrices)
        print("Step 2 Completed in", self.convert(time.time() - start), "time")
            
    def Step3(self):
        
        #Step 3: Find approximate 3D points for 2D correspondences using triangulation
        start = time.time()
        print()
        print("Finding approximate 3D position of 2D correspondences using Triangulation")
        self.IntPnts3DOld = []
        for row in self.PntsMatch:
            Choices = np.where(row != -1)[0]
            if Choices.shape[0]<2:
                continue
            i, j = np.random.choice(Choices, 2, replace=False)
            pi, pj = self.IntPnts[i][row[i]], self.IntPnts[j][row[j]]
            pnt3D = self.triangulate(pi, pj, self.K, self.R_t_matrices[i], self.R_t_matrices[j])
            self.IntPnts3DOld.append(pnt3D)
        self.IntPnts3DOld = np.array(self.IntPnts3DOld)
        print("Step 3 Completed in", self.convert(time.time() - start), "time")
        
    def Step4(self):
        
        #Step 4: Perform Bundle Adjustmen on IntPnts3D & R_t_matrices
        start = time.time()
        print()
        print("Step 4: Applying Bundle Adjustment to fine tune Camera Matrices & 3D point approximation")
        S4 = BundleAdjustment.BundleAdjustment(self.PntsMatch.T, self.R_t_matrices, self.IntPnts3DOld, self.IntPnts, self.K)
        self.R_t_Matrices, self.IntPnts3D = S4.LeastSquaresCalculator()
        print("Step 4 Completed in", self.convert(time.time() - start), "time")
        
        if self.output_folder != None:
            self.Check2()
    
    def Check2(self):
        
        P1 = self.R_t_matrices[0]
        P1_ = self.R_t_Matrices[0]
        for j in range(1, self.Imgs.shape[0]): 
            P2 = self.R_t_matrices[j]
            P2_ = self.R_t_Matrices[j]
            
            Pnts2D1, Pnts2D2 = self.get_match_pair_list(0, j)
            Pnts1 = []
            Pnts2 = []
            for p1, p2 in zip(Pnts2D1, Pnts2D2):
                Pnts1.append(self.triangulate(p1, p2, self.K, P1, P2))
                Pnts2.append(self.triangulate(p1, p2, self.K, P1_, P2_))
            
            self.plot_compare(Pnts1, Pnts2)
    
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
                    
                    S5 = DenseMatching.DenseMatching(img1, img2, self.IntPnts[i][:,:2], self.IntPnts[j][:,:2], i, j)
                    #IntPnts1, IntPnts2 of form (y, x)
                    IntPnts1, IntPnts2 = S5.PerformDenseMatching(self.param_dict['thld_zncc_seed'], self.param_dict['w_seed'], 
                                                                 self.param_dict['thld_zncc_propagation'], 
                                                                 self.param_dict['thld_confidence'], self.param_dict['w_propagation'], 
                                                                 self.param_dict['N_window'], self.param_dict['epsilon_propagation'])
                    self.NewMatchedPointPairs[f'{i}_{j}'] = [np.hstack((IntPnts1, np.ones((IntPnts1.shape[0], 1)))).astype(int), np.hstack((IntPnts2, np.ones((IntPnts2.shape[0], 1)))).astype(int)]
                    print()
        print("Step 5 Completed in", self.convert(time.time() - start), "time")
        
        if self.output_folder != None:
            self.Check3()
    
    def Check3(self):
        
        for key, val in self.NewMatchedPointPairs.items():
            a, b = np.array(key.split('_'), dtype=int)
            Old1, Old2 = self.get_match_pair_list(a, b)
            New1, New2 = val[0], val[1]
            
            fig = plt.figure(figsize=(10, 10))
            plt.title(f"Resulting Matched Points after Dense Matching on Images {a}-{b}")
            plt.axis('off')
            for i, pair in enumerate([Old1.astype(int), Old2.astype(int), New1.astype(int), New2.astype(int)]):
                ax = fig.add_subplot(3, 2, 1+i)
                img = np.zeros((self.Imgs[0].shape[0], self.Imgs[0].shape[1]))
                if i < 2:
                    img[pair[:,1], pair[:,0]] = 1
                else:
                    img[pair[:,0], pair[:,1]] = 1
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                
            ax = fig.add_subplot(3, 2, 5)
            ax.imshow(self.Imgs[a])
            ax.axis('off')
            
            ax = fig.add_subplot(3, 2, 6)
            ax.imshow(self.Imgs[b])
            ax.axis('off')
    
    def Step6(self):
        
        #Step 6: Triangulating NewMatchedPointPairs to get 3D coordinates for Point Cloud
        start = time.time()
        print()
        print("Step 6: 3D points will be calculated using 2D correspondences from Dense Matching")
        self.New3DPnts = []
        self.Color3DPnts = []
        for key, val in self.NewMatchedPointPairs.items():
            i, j = list(map(int, key.split('_')))
            IntPnts1, IntPnts2 = val
            P1, P2 = self.R_t_Matrices[i], self.R_t_Matrices[j]
            for p1, p2 in zip(IntPnts1, IntPnts2):
                p1 = np.array([p1[1], p1[0], p1[2]])
                p2 = np.array([p2[1], p2[0], p2[2]])
                pnt3D = self.triangulate(p1, p2, self.K, P1, P2)
                self.New3DPnts.append(pnt3D)
                
                clr3D = self.Imgs[i][p1[1], p1[0]]
                if type(clr3D) == int or type(clr3D) == float:
                    self.Color3DPnts.append([clr3D]*3)
                else:
                    self.Color3DPnts.append(clr3D)
        
        self.New3DPnts = np.array(self.New3DPnts)
        self.Color3DPnts = np.array(self.Color3DPnts)
        print("Step 5 Completed in", self.convert(time.time() - start), "time")
    
    def Step7(self):
        
        #Step 7: Visualization
        fig = plt.figure(figsize=(15, 15))
        ax = fig.gca(projection='3d')
        ax.scatter(self.New3DPnts[:,0], self.New3DPnts[:,1], self.New3DPnts[:,2], c=self.Color3DPnts/255)
    
    def Reconstruct(self):
        
        self.Step1()
        self.Step2()
        self.Step3()
        self.Step4()
        self.Step5()
        self.Step6()
        self.Step7()

#%%
Construct = Reconstruction(glob.glob('Data/ImageSet/Random/*'), output_folder="Data/Output")
Construct.Reconstruct()
import pickle
pickle.dump(Construct, open("Reconstruct.pkl", "ab"))
#%%
Construct.Step1()
#%%
Construct.Step2()
#%%
Construct.Step3()
#%%
Construct.Step4()
#%%
Construct.Step5()
#%%
Construct.Step6()
#%%
Construct.Step7()
#%%
import pickle
pickle.dump(Construct, open("Reconstruct.pkl", "ab"))
#%%
import pickle
Cons = pickle.load(open("Reconstruct.pkl", "rb"))
Construct.IntPnts = Cons.IntPnts
Construct.IntPnts3D = Cons.IntPnts3D
Construct.IntPnts3DOld = Cons.IntPnts3DOld
Construct.K = Cons.K
Construct.NewMatchedPointPairs = Cons.NewMatchedPointPairs
Construct.PntsMatch = Cons.PntsMatch
Construct.R_t_Matrices = Cons.R_t_Matrices
Construct.R_t_matrices = Cons.R_t_matrices
#%%
fig = plt.figure()
plt.title('akshat')
plt.axis('off')
img = np.ones((5, 10))
img[1, 1] = 1
ax = fig.add_subplot(2, 2, 1)
ax.imshow(img)

