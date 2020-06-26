import numpy as np
import math 
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sympy import *
import warnings
from itertools import combinations
import sys
from tqdm import tqdm

warnings.filterwarnings('ignore')

class BundleAdjustment:
    def __init__(self, PntsMatch, R_t_matrices, WorldPntsOld, IntPnts, K, thld):
        """
        Arguments
            PntsMatch : Matrix of shape (No. of Images, No. of World Points), information about index of matched points in respective images.
            R_t_matrices : numpy.ndarray of shape (No. of Images, 4, 3), Array of approximated Extrinsic Camera Matrix for all images
            WorldPntsOld : numpy.ndarray of shape (No. of World Pnts, 3), Array of all approximated 3D World Pnts.
            IntPnts : numpy array of shape (No. of Images, ), Each element is a numpy array of Intrest Points extracted in Feature Matching section.
            K : numpy.ndarray of shape (3, 3), Intrinsic Matrix calculated in CameraMatrix section. Assumed all the images are taken with same camera.
        """
        
        self.WorldPntsOld = WorldPntsOld   ## no. of world points X 3 numpy array
        self.K = K
        self.PntsMatch = PntsMatch   ## no. of images X no. of world points. This will tell index of matched image points in respective images
        self.R_t_matrices = R_t_matrices  ## no.of cameras X 4 X 3
        self.IntPnts = IntPnts   ##no. of images X no. interest points with 2 or more matches in that image
        self.thld = thld
        self.Rotation_to_quat()
        self.JacCalculator()
        self.EpipolarFuns()
        self.sparsity_and_imagep_calculator()
        self.loss = np.inf; self.err_repro = np.inf; self.err_epipolar = np.inf
    
    def Rotation_to_quat(self):
        Final = []    ## shape = no.of cameras X (x,y,z,theta,t1,t2,t3)

        for i in self.R_t_matrices:
            m = i[:,:3]
            trans = i[:,3]
            
            R = Rotation.from_matrix(m)
            x, y, z, w = R.as_quat()
            
            Final.append([w, x, y, z, trans[0], trans[1], trans[2]])
            
        self.initR_t = np.array(Final).flatten('C')
        self.WorldPntsOld = self.WorldPntsOld.flatten('C')
        self.init_W_R_t = np.zeros((1,self.initR_t.shape[0]+self.WorldPntsOld.shape[0]))
        self.init_W_R_t[0,0:self.WorldPntsOld.shape[0]] = self.WorldPntsOld
        self.init_W_R_t[0,self.WorldPntsOld.shape[0]:] = self.initR_t
        self.yies, self.xes = np.where(self.PntsMatch != -1)

    def find_epipolar_pairs(self):
        
        self.EpipolarPairs = []
        for row in self.PntsMatch.T:
            index = np.where(row != -1)[0]
            for comb in np.array(list(combinations(range(index.shape[0]), 2))):
                i, j = comb
                p1, p2 = self.IntPnts[i][row[i]][:2], self.IntPnts[j][row[j]][:2]
                self.EpipolarPairs.append(list(p1) + list(p2) + [i, j])
        self.EpipolarPairs = np.array(self.EpipolarPairs, dtype=int)
        
    def EpipolarFuns(self):
        
        w, x, y, z, p, q, r = symbols("w x y z p q r")
        w_, x_, y_, z_, p_, q_, r_ = symbols("w_ x_ y_ z_ p_ q_ r_")
        a, b, a_, b_ = symbols("a b a_ b_")
        
        R_ = Matrix([ [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                      [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                      [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y] ])
        R__ = Matrix([ [1 - 2*y_*y_ - 2*z_*z_, 2*x_*y_ - 2*z_*w_, 2*x_*z_ + 2*y_*w_],
                       [2*x_*y_ + 2*z_*w_, 1 - 2*x_*x_ - 2*z_*z_, 2*y_*z_ - 2*x_*w_],
                       [2*x_*z_ - 2*y_*w_, 2*y_*z_ + 2*x_*w_, 1 - 2*x_*x_ - 2*y_*y_] ])
        R = R_.inv() * R__
        self.create_R = utilities.lambdify((w, x, y, z, w_, x_, y_, z_), R)
        
        t_ = Matrix([ p, q, r ]); t__ = Matrix([ p_, q_, r_ ])
        t = t__ - t_
        skew_t = Matrix([ [0, -t[2], t[1]],
                          [t[2], 0, -t[0]],
                          [-t[1], t[0], 0] ])
        self.create_skew_t = utilities.lambdify((p, q, r, p_, q_, r_), skew_t)
        #1.783367136838699
        #-0.002226478859272407
        
        F = Matrix(self.K).inv().T * skew_t * R * Matrix(self.K).inv()
        p1 = Matrix([a, b, 1]); p2 = Matrix([a_, b_, 1])
        
        x_Fx = (p2.T * F * p1)
        x_Fx = x_Fx[0] * self.thld
        self.create_x_Fx = utilities.lambdify((a, b, w, x, y, z, p, q, r, 
                                               a_, b_, w_, x_, y_, z_, p_, q_, r_), x_Fx)
        
        x_Fx_diff = diff(x_Fx, Matrix([w, x, y, z, p, q, r, 
                                       w_, x_, y_, z_, p_, q_, r_]), 1)
        self.create_x_Fx_diff = utilities.lambdify((a, b, w, x, y, z, p, q, r, 
                                                    a_, b_, w_, x_, y_, z_, p_, q_, r_), x_Fx_diff)        
        
        self.find_epipolar_pairs()
        
    def JacCalculator(self):
        
        from sympy.abc import X, Y, Z, w, x, y, z, p, q, r

        P = Matrix([ [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, p],
                     [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w, q],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, r] ])
        self.create_P = utilities.lambdify((w, x, y, z, p, q, r), P)
        
        A = Matrix([ [X], [Y], [Z], [1] ])   ### World point matrix
    
        temp = self.K*P*A
        temp = temp / temp[-1] 
        temp = Matrix([temp[0], temp[1]])  ### for calculating image point estimate 
        self.xhat = utilities.lambdify((X, Y, Z, w, x, y, z, p, q, r), temp)
        
        J = diff(temp, Matrix([X, Y, Z, w, x, y, z, p, q, r ]), 1)
        self.jacfun = utilities.lambdify((X, Y, Z, w, x, y, z, p, q, r), J)
        
    def JacCreator(self, current_matrix):  ## current matrix has (X, R|t) at current iteration and shape (1,no.)   
        ### This function calculates Jacobian at every iteration, by creating matrix of shape self.jacdims & using sympy 
        ### to fill jacobian values to be dealt with
        current_matrix = current_matrix.reshape((1,current_matrix.shape[0]))
        num_3D = self.PntsMatch.shape[1]
        
        Jacobian = np.zeros((2*self.xes.shape[0] + self.EpipolarPairs.shape[0], current_matrix.shape[1]))
        for i, (x, y) in enumerate(zip(self.xes, self.yies)):
            X_, Y_, Z_ = current_matrix[0][3*x : 3*x+3]
            w_, x_, y_, z_, p_, q_, r_ = current_matrix[0][3*num_3D + 7*y : 3*num_3D + 7*y + 7]
            
            ## now calculating temporary jacobian matrix
            tempe = self.jacfun(X_, Y_, Z_, w_, x_, y_, z_, p_, q_, r_)
            tempe = np.asarray(tempe)
            tempe = tempe.reshape((tempe.shape[0],tempe.shape[2])).T
            
            ## now putting values of temporary jacobian into final jacobian & calculating sparse_matrix
            Jacobian[2*i : 2*i+2, 3*x : 3*x+3] = np.array([tempe[0,:3],tempe[1,:3]])
            Jacobian[2*i : 2*i+2, 3*num_3D + 7*y : 3*num_3D + 7*y+7] = np.array([tempe[0,3:10],tempe[1,3:10]])
            
        for k, row in enumerate(self.EpipolarPairs):
            a, b, a_, b_, i, j = row
            w, x, y, z, p, q, r = current_matrix[0][3*num_3D + 7*i : 3*num_3D + 7*i + 7]
            w_, x_, y_, z_, p_, q_, r_ = current_matrix[0][3*num_3D + 7*j : 3*num_3D + 7*j + 7]
            
            x_Fx_diff = self.create_x_Fx_diff(a, b, w, x, y, z, p, q, r, 
                                              a_, b_, w_, x_, y_, z_, p_, q_, r_).ravel()
            
            Jacobian[2*self.xes.shape[0] + k, 3*num_3D + 7*i : 3*num_3D + 7*i + 7] = x_Fx_diff[:7]
            Jacobian[2*self.xes.shape[0] + k, 3*num_3D + 7*j : 3*num_3D + 7*j + 7] = x_Fx_diff[:7]
            #done
            
        return Jacobian[:-1*self.EpipolarPairs.shape[0]]
    
    def difference_for_ls(self, current_matrix):
        ''' First argument in least squares'''
        
        self.iter += 1
        current_matrix = current_matrix.reshape((1,current_matrix.shape[0]))
        num_3D = self.PntsMatch.shape[1]
        
        PredVal = []
        for i, (x, y) in enumerate(zip(self.xes, self.yies)):
            X_, Y_, Z_ = current_matrix[0][3*x : 3*x+3]
            w_, x_, y_, z_, p_, q_, r_ = current_matrix[0][3*num_3D + 7*y : 3*num_3D + 7*y + 7]
            
            ## now calculating image_point_hat (i.e. estimate) 
            pred = self.xhat(X_, Y_, Z_, w_, x_, y_, z_, p_, q_, r_)
            PredVal.append(pred[0,0]); PredVal.append(pred[1,0])
        
        for k, row in enumerate(self.EpipolarPairs):
            a, b, a_, b_, i, j = row
            w, x, y, z, p, q, r = current_matrix[0][3*num_3D + 7*i : 3*num_3D + 7*i + 7]
            w_, x_, y_, z_, p_, q_, r_ = current_matrix[0][3*num_3D + 7*j : 3*num_3D + 7*j + 7]
            
            x_Fx = self.create_x_Fx(a, b, w, x, y, z, p, q, r, 
                                    a_, b_, w_, x_, y_, z_, p_, q_, r_)
            PredVal.append(x_Fx)
        PredVal = np.asarray(PredVal).flatten('C')
        
        err_repro = PredVal[:-1*self.EpipolarPairs.shape[0]] - self.ActualVal[:-1*self.EpipolarPairs.shape[0]]
        err_epipolar = PredVal[-1*self.EpipolarPairs.shape[0]:] - self.ActualVal[-1*self.EpipolarPairs.shape[0]:]
        err_repro = np.mean(np.abs(err_repro)); err_epipolar = np.mean(np.abs(err_epipolar))
        
        loss = err_repro + err_epipolar
        if loss < self.loss:
            self.loss = loss; self.err_repro = err_repro; self.err_epipolar = err_epipolar
            print(f"\t\t{self.iter}\t\t\t\t{self.err_repro}\t\t\t\t{self.err_epipolar}")
            
        return (PredVal - self.ActualVal)[:-1*self.EpipolarPairs.shape[0]]
    
    def sparsity_and_imagep_calculator(self):
        '''
           Will be called only once during initialization.  
        '''
        
        sparse_matrix = np.zeros((2*self.xes.shape[0] + self.EpipolarPairs.shape[0], 3*self.PntsMatch.shape[1] + 7*self.PntsMatch.shape[0]))
        
        ActualVal = []
        for i, (y, x) in enumerate(zip(self.yies, self.xes)):
            ActualVal.append(self.IntPnts[y][self.PntsMatch[y, x]][0])
            ActualVal.append(self.IntPnts[y][self.PntsMatch[y, x]][1])
            
            sparse_matrix[(2*i):(2*i+2), (3*x):(3*x+3)] = np.ones((2,3))
            sparse_matrix[(2*i):(2*i+2), (3*self.PntsMatch.shape[1] + 7*y):(3*self.PntsMatch.shape[1] + 7*y + 7)] = np.ones((2,7))
            
        for k, row in enumerate(self.EpipolarPairs):
            i, j = row[-2:]
            
            ActualVal.append(0)
            sparse_matrix[2*self.xes.shape[0] + k, (3*self.PntsMatch.shape[1] + 7*i):(3*self.PntsMatch.shape[1] + 7*i + 7)] = np.ones(7)
            sparse_matrix[2*self.xes.shape[0] + k, (3*self.PntsMatch.shape[1] + 7*j):(3*self.PntsMatch.shape[1] + 7*j + 7)] = np.ones(7)
            #done
            
        self.sparse_matrix = sparse_matrix[:-1*self.EpipolarPairs.shape[0]]
        ActualVal = np.asarray(ActualVal).flatten('C')
        self.ActualVal = ActualVal
    
    def LeastSquaresCalculator(self):
        '''
        Calulates optimal values of World co-ordinates & R & t matrices
        '''
        
        print(f">>> Number of variables: {self.init_W_R_t.reshape(-1).shape[0]}, Number of 3D Pnts: {self.PntsMatch.shape[1]}")
        print(">>> Bundle Adjustment in process ...")
        print("\tIterations\t\t\tReprojection Err\t\t\t\tEpipolar Error")
        self.iter = 0
        
        result = least_squares(self.difference_for_ls, self.init_W_R_t.reshape(-1), self.JacCreator, ftol=1e-2, method="lm") 
        #result = least_squares(self.difference_for_ls, self.init_W_R_t.reshape(-1), self.JacCreator, ftol=1e-2, method="lm", x_scale='jac') 
        #result = least_squares(self.difference_for_ls, self.init_W_R_t.reshape(-1), '3-point', ftol=1e-4, method="trf", jac_sparsity=self.sparse_matrix, verbose=2)
        
        TunedParams = result.x
        #if result.status == 0:
        #    print(">>> Process terminated because the maximum number of function evaluations is exceeded")
        #elif result.success:
        #    print(f">>> Process Successful with cost: {result.cost}")
        IntWorldPnts = TunedParams[:self.WorldPntsOld.shape[0]].reshape((self.WorldPntsOld.shape[0]//3, 3))
        R_t_Matrices = []
        for R_t in TunedParams[self.WorldPntsOld.shape[0]:].reshape((self.R_t_matrices.shape[0], 7)):
            R_t_Matrices.append(self.create_P(R_t[0], R_t[1], R_t[2], R_t[3], R_t[4], R_t[5], R_t[6]))
        
        return np.array(R_t_Matrices), IntWorldPnts