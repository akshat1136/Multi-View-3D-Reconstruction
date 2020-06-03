import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class FundamentalCameraMatrix:
    
    def __init__(self):
        return
    
    def ImgPairDetails(self, points1, points2, img1, img2):
        self.points1 = points1
        self.points2 = points2
        self.Imgs = [img1, img2]
        self.x0 = img1.shape[1] // 2
        self.y0 = img1.shape[0] // 2
        self.num_runs = 0
        self.create_intrinsic_K()
        
    def create_intrinsic_K(self):
        
        alpha = self.y0 / self.x0
        if self.x0 < self.y0:
            f = self.x0*2 + abs(self.x0*2 - self.y0*2) / (1 + alpha)
        else:
            f = self.y0*2 + alpha*abs(self.x0*2 - self.y0*2) / (1 + alpha)
            
        self.K = np.array([[f, 0, self.x0],
                           [0, f, self.y0],
                           [0, 0, 1]])
        
        self.K = np.array([[1390.30384, 0, self.x0],
                        [0, 1388.07396, self.y0],
                        [0, 0, 1]])
    
    def create_matrix_A(self, points1, points2):
        '''
        Arguements
            points1 : np.ndarray of shape (8, 3), represents 8 points from image1.
            points2 : np.ndarray of shape (8, 3), represents 8 points from image2.
        Return
            A : `A` matrix of shape (8, 9)
        '''
        A = []
        for i in range(points1.shape[0]):
            A.append(np.matmul(points2[i].reshape((3, 1)), points1[i].reshape((1, 3))).ravel())
        A = np.array(A)
        return A
    
    def normalized_fundamental(self, points1, points2):
        '''
        Arguments
            points1 : np.ndarray of shape (8, 3), represents 8 points from image1.
            points2 : np.ndarray of shape (8, 3), represents 8 points from image2.
        Return
            F_mat : Fundamental Matrix for image1 and image2
        Procedure
            1) Normalize the image coordinates
            2) Find f of A*f=0 using SVD of A matrix
            3) Shape f into a 3x3 matrix and normalize it and perform `Rank Enforcement`
            4) Denormalize F (Undo the 1st step)
        '''
        #Convert Homogenous coordinates to Image Coordinates
        points1 = points1 / points1[:,2].reshape((-1,1))
        points2 = points2 / points2[:,2].reshape((-1,1))
        
        # First, normalize the points
        std_x, std_y = math.sqrt(2) / np.std(range(int(self.x0*2))), math.sqrt(2) / np.std(range(int(self.y0*2)))
        mean_x, mean_y = np.mean(range(int(self.x0*2))), np.mean(range(int(self.y0*2)))
        T = np.array( [[std_x, 0, -1*std_x*mean_x],
                       [0, std_y, -1*std_y*mean_y],
                       [0, 0, 1]] )        
        points_hat1 = np.matmul(T, points1.T).T
        points_hat2 = np.matmul(T, points2.T).T
        
        #Second, Find f using SVD on A
        A_mat = self.create_matrix_A(points_hat1, points_hat2)
        U, S, V = np.linalg.svd(A_mat)
        F = V[-1].reshape((3, 3))
        
        # Apply `Rank Enforcement`
        U, S, V = np.linalg.svd(F)
        S = np.diag(S)
        S[-1] = 0
        F = np.matmul(U, np.matmul(S, V))
        F = F/F[2,2]
        
        # De-normalizing
        F = np.matmul(T.T, np.matmul(F, T))
        F = F/F[2,2]
        
        return F
    
    def epipolar_line_points(self, F, points_main, points_plot, img, ax, epipole):
        '''
        
        '''
        color = ['blue', 'orange', 'green', 'red', 'purple']
        Img = self.Imgs[img]
        ax.imshow(Img, cmap='gray')
        for i, point in enumerate(points_main):
            ax.plot(points_plot[i][0]/points_plot[i][2], points_plot[i][1]/points_plot[i][2], 'o', color=color[i])
            if img:
                line = np.matmul(F, point.T)
            else:
                line = np.matmul(point, F)
            x_axis = np.linspace(20, Img.shape[1]-20, 200)
            y_axis = (line[0]*x_axis + line[2]) / (-1*line[1])
            
            valid_index = (y_axis>20) & (y_axis<Img.shape[0] - 20)
            x_axis = x_axis[valid_index]
            y_axis = y_axis[valid_index]
            ax.plot(x_axis, y_axis, alpha=0.5, color=color[i])
            
        if epipole:
            F = F.T if img else F
            _, _, V = np.linalg.svd(F)
            epipole = V[-1]
            epipole = epipole / epipole[2]
            ax.plot(epipole[0], epipole[1], 'o', 'black')
    
    def plot_epipoler_epipole(self, F, points1, points2, epipole):
        '''
        
        '''
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        self.epipolar_line_points(F, points2, points1, 0, axs[0], epipole)
        self.epipolar_line_points(F, points1, points2, 1, axs[1], epipole)
        plt.show()
    
    def fundamental_matrix_RANSAC(self, loop, thld):
        '''
        Arguments
            points1 : np.ndarray of shape(n, 3), `n` is number of points in image1 matched to image1.
            points2 : np.ndarray of shape(n, 3), `n` is number of points in image2 matched to image1.
        Return
            F_best : Fundamental Matrix (3x3), with least error in matching points.
        '''
        
        for i in range(loop):
            eight_points = np.random.choice(range(self.points1.shape[0]), 8, replace=False)
            F = self.normalized_fundamental(self.points1[eight_points], self.points2[eight_points])
            
            inliers = np.matmul(self.points2, F)
            inliers = np.sum(inliers * self.points1, axis=1) / np.sqrt(np.sum(inliers[:,:2]**2, axis=1))
            inliers = np.sum(np.abs(inliers) < thld)
            
            if i == 0 or inliers > num_inliers:
                num_inliers = inliers
                F_best = F
        self.F = F_best
    
    def essential_matrix(self):
        E = np.matmul(self.K.T, np.matmul(self.F, self.K))
        U, S, Vt = np.linalg.svd(E)
        S = np.diag([1, 1, 0])
        
        if np.linalg.det(np.matmul(U, Vt)) < 0:
            self.E = np.matmul(U, np.matmul(S, -1*Vt))
        else:
            self.E = np.matmul(U, np.matmul(S, Vt))
    
    def camera_matrix(self, reprojection):              
        W = np.array( [ [0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1] ] )
        Z = np.array( [ [0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 0] ] )
        self.P1 = np.array( [ [1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0]] )
        self.M1 = np.matmul(self.K, self.P1)
        
        U, S, Vt = np.linalg.svd(self.E)
        
        T = np.matmul(U, np.matmul(Z, U.T))
        U3 = np.array( [ T[2, 1], T[0, 2], T[1, 0] ] ).reshape((3, 1)) * 10
        
        R1 = np.matmul(U, np.matmul(W, Vt))
        R2 = np.matmul(U, np.matmul(W.T, Vt))
        R1 = R1 if np.linalg.det(R1)>0 else -1*R1
        R2 = R2 if np.linalg.det(R1)>0 else -1*R2
        
        self.P2_possible = np.array([ np.hstack((R1, U3)), 
                                      np.hstack((R2, U3)), 
                                      np.hstack((R1, -1 * U3)), 
                                      np.hstack((R2, -1 * U3)) ])
        
        self.Pnts3D = []
        self.reprojection_err = []
        num_points = np.zeros(4)
        for i, P2 in enumerate(self.P2_possible):
            M2 = np.matmul(self.K, P2)
            num_points[i] = self.calP2possibility(P2, M2, reprojection)
        
        self.reprojection_err = self.reprojection_err[np.argmax(num_points)]
        self.P2 = self.P2_possible[np.argmax(num_points)]
        self.Pnts3Dbest = self.Pnts3D[np.argmax(num_points)]
        self.M2 = np.matmul(self.K, self.P2)
        
        return (num_points == self.points1.shape[0]).sum()
    
    def check_point_presence(self, points, P2, M2):
        
        front1 = points[:,2] > 0
        points = np.matmul(self.P1, points.T).T
        front2 = points[:,2] > 0
        
        visible = front1 & front2
        visible = visible.sum()
        
        return visible
    
    def plot3D_P2(self, P2, points):
        ax1 = plt.figure().gca(projection='3d')
        ax1._axis3don = False
        
        Cam1 = np.array([0, 0, 0])
        Cam2 = -1 * P2[:,-1]
        
        ax1.plot3D([Cam1[0], Cam2[0]], [Cam1[1], Cam2[1]], [Cam1[2], Cam2[2]], 'v')
        for i, p in enumerate(points):
            ax1.plot3D([p[0]], [p[1]], [p[2]], 'o', alpha=0.5)
    
    def calP2possibility(self, P2, M2, reprojection):
        points = []
        for p1, p2 in zip(self.points1, self.points2):
            Q = np.zeros((6, 6))
            Q[:3,:4] = self.M1
            Q[3:,:4] = M2
            Q[:3,4] = -p1
            Q[3:,5] = -p2
            U, S, Vt = np.linalg.svd(Q)
            X = Vt[-1, :4]
            X = X / X[3]
            points.append(X)
            
        points = np.array(points)
        self.Pnts3D.append(points)
        
        visible = self.check_point_presence(points, P2, M2)
        self.check_reprojection(M2, points, reprojection)
        
        return visible
    
    def check_reprojection(self, M2, points, reprojection):
        
        points1 = np.matmul(self.M1, points.T).T
        points1 = points1 / points1[:,-1].reshape((-1, 1))
        points1 = points1.astype(int)
        
        points2 = np.matmul(M2, points.T).T
        points2 = points2 / points2[:,-1].reshape((-1, 1))
        points2 = points2.astype(int)
        
        err = np.abs(points1 - self.points1).sum() / self.points1.shape[0] + np.abs(points2 - self.points2).sum() / self.points2.shape[0]
        self.reprojection_err.append(err)
        
        if reprojection:
            print('Reprojection Error')
            for i in range(points1.shape[0]):
                print('>>>  ', (points1[i] - self.points1[i]), (points2[i] - self.points2[i]))

    def Calculate_F_E_C_Matrix(self, reprojection=False, loop_F_mat=200, thld_reprojection=30, thld_F_inliers=0.2, plot_epipolar_line=False, plot_3D_P2=False, plot_P2_best=False, plot_epipole=False):        
        
        #num_run = 1
        #self.fundamental_matrix_RANSAC(loop_F_mat, thld_F_inliers)
        #self.essential_matrix()
        #check = self.camera_matrix(reprojection)
        reprojection_err = 1000
        pbar = tqdm(range(100), file=sys.stdout)
        #while check != 1 or self.reprojection_err > thld_reprojection:
        for i in pbar:
            #if num_run >= 1000:
            #    raise ValueError('Increase thld_reprojection, could not find a suitable matrix for current value.')
            self.fundamental_matrix_RANSAC(loop_F_mat, thld_F_inliers)
            self.essential_matrix()
            check = self.camera_matrix(reprojection)
            
            if check and self.reprojection_err < reprojection_err:
                data = self.F, self.E, self.K, self.P1, self.M1, self.K, self.P2, self.M2, self.reprojection_err
                reprojection_err = self.reprojection_err
            pbar.set_description(">>> Current Best Reprojection Error " + str(reprojection_err)[:6])
        pbar.close()
        
        if plot_epipolar_line:
            eight_points = np.random.choice(range(self.points1.shape[0]), 5, replace=False)
            self.plot_epipoler_epipole(self.F, self.points1[eight_points], self.points2[eight_points], plot_epipole)
        
        if plot_3D_P2:
            if plot_P2_best:
                self.plot3D_P2(self.P2, self.Pnts3Dbest)
            else:
                for points, P2 in zip(self.Pnts3D, self.P2_possible):
                    self.plot3D_P2(P2, points)
        
        self.F, self.E, self.K, self.P1, self.M1, self.K, self.P2, self.M2, self.reprojection_err = data
        return self.F, self.E, self.K, self.P1, self.M1, self.K, self.P2, self.M2, self.reprojection_err