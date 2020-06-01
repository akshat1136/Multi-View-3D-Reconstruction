import numpy as np
import math 
from scipy.optimize import least_squares
from sympy import *
import warnings

warnings.filterwarnings('ignore')

class BundleAdjustment:
    def __init__(self, world_to_imagematrix, initRtvalues, initWorldp, imagecoords, K_intrinsic_matrix):
        """
        Arguments
            world_to_imagematrix : Matrix of shape (No. of Images, No. of World Points), information about index of matched points in respective images.
            initRtvalues : numpy.ndarray of shape (No. of Images, 4, 3), Array of approximated Extrinsic Camera Matrix for all images
            initWorldp : numpy.ndarray of shape (No. of World Pnts, 3), Array of all approximated 3D World Pnts.
            imagecoords : numpy array of shape (No. of Images, ), Each element is a numpy array of Intrest Points extracted in Feature Matching section.
            K_intrinsic_matrix : numpy.ndarray of shape (3, 3), Intrinsic Matrix calculated in CameraMatrix section. Assumed all the images are taken with same camera.
        """
        
        self.initWorldp = initWorldp   ## no. of world points X 3 numpy array
        self.K_intrinsic_matrix = K_intrinsic_matrix
        self.world_to_imagematrix = world_to_imagematrix   ## no. of images X no. of world points. This will tell index of matched image points in respective images
        self.initRtvalues = initRtvalues  ## no.of cameras X 4 X 3
        self.imagecoords = imagecoords   ##no. of images X no. interest points with 2 or more matches in that image
        self.Rotation_to_axisangle()
        self.JacCalculator()
        self.sparsity_and_imagep_calculator()
    
    def Rotation_to_axisangle(self):
        Final = []    ## shape = no.of cameras X (x,y,z,theta,t1,t2,t3)

        for i in self.initRtvalues:
            m = i[:,:3]
            trans = i[:,3]
            
            epsilon = 0.01
            epsilon2 = 0.1
            if ((abs(m[0][1]-m[1][0])< epsilon) and (abs(m[0][2]-m[2][0])< epsilon) and (abs(m[1][2]-m[2][1])< epsilon)):
                # singularity found...
                # first check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
                if ((abs(m[0][1]+m[1][0]) < epsilon2) and (abs(m[0][2]+m[2][0]) < epsilon2) and (abs(m[1][2]+m[2][1]) < epsilon2) and (abs(m[0][0]+m[1][1]+m[2][2]-3) < epsilon2)):
                    # angle= 0 degrees
                    Final.append([1,0,0,0,trans[0],trans[1],trans[2]]) 
                else:
                    # angle = 180 degrees
                    angle = np.pi;
                    xx = (m[0][0]+1)/2
                    yy = (m[1][1]+1)/2
                    zz = (m[2][2]+1)/2
                    xy = (m[0][1]+m[1][0])/4
                    xz = (m[0][2]+m[2][0])/4
                    yz = (m[1][2]+m[2][1])/4
                    
                    if ((xx > yy) and (xx > zz)):    # m[0][0] is the largest diagonal term
                        if (xx< epsilon):
                            x = 0
                            y = 0.7071
                            z = 0.7071
                        else:
                            x = math.sqrt(xx)
                            y = xy/x
                            z = xz/x

                    elif (yy > zz):    # m[1][1] is the largest diagonal term
                        if (yy< epsilon):
                            x = 0.7071
                            y = 0
                            z = 0.7071
                        else: 
                            y = math.sqrt(yy);
                            x = xy/y
                            z = yz/y
                            
                    else:    # m[2][2] is the largest diagonal term so base result on this
                        if (zz< epsilon): 
                            x = 0.7071
                            y = 0.7071
                            z = 0
                        else:
                            z = math.sqrt(zz)
                            x = xz/z
                            y = yz/z
                    Final.append([x,y,z,angle,trans[0],trans[1],trans[2]])
            else:
                # as we have reached here there are no singularities so we can handle normally
                s = math.sqrt((m[2][1] - m[1][2])*(m[2][1] - m[1][2])+(m[0][2] - m[2][0])*(m[0][2] - m[2][0])+(m[1][0] - m[0][1])*(m[1][0] - m[0][1])) # used to normalise
                if (abs(s) < 0.001):
                    s=1
                angle = math.acos((m[0][0] + m[1][1] + m[2][2] - 1)/2);
                x = (m[2][1] - m[1][2])/s
                y = (m[0][2] - m[2][0])/s
                z = (m[1][0] - m[0][1])/s
                Final.append([x,y,z,angle,trans[0],trans[1],trans[2]])

        self.initR_t = np.array(Final).flatten('C')
        ## flattening world points array
        self.initWorldp = self.initWorldp.flatten('C')
        ## calculating self.init_W_R_t 
        self.init_W_R_t = np.zeros((1,self.initR_t.shape[0]+self.initWorldp.shape[0]))
        self.init_W_R_t[0,0:self.initWorldp.shape[0]] = self.initWorldp
        self.init_W_R_t[0,self.initWorldp.shape[0]:] = self.initR_t
        
    def JacCalculator(self):
        from sympy.abc import X, Y, Z, x, y, z, theta, p, q, r
        s = sin(theta)
        c = cos(theta)
        t = 1 - c
        
        '''
        R = Matrix([ [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                     [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                     [t*x*z - y*s, t*y*z + x*s, t*z*z + c] ])

        t = Matrix([ [p], [q], [r] ])
        '''

        P = Matrix([ [t*x*x + c, t*x*y - z*s, t*x*z + y*s, p],   ### Camera Matrix
                     [t*x*y + z*s, t*y*y + c, t*y*z - x*s, q],
                     [t*x*z - y*s, t*y*z + x*s, t*z*z + c, r] ])
        self.create_P = utilities.lambdify((x, y, z, theta, p, q, r), P)

        A = Matrix([ [X], [Y], [Z], [1] ])   ### World point matrix
    
        
        temp = self.K_intrinsic_matrix*P*A
        temp = temp / temp[-1] 
        temp = Matrix([temp[0], temp[1]])
        self.xhat = utilities.lambdify((X, Y, Z, x, y, z, theta, p, q, r), temp)  ### for calculating image point estimate 
        J = diff(temp, Matrix([X, Y, Z, x, y, z, theta, p, q, r ]), 1)
        self.jacfun = utilities.lambdify((X, Y, Z, x, y, z, theta, p, q, r), J)
    
    def JacCreator(self, current_matrix):  ## current matrix has (X, R|t) at current iteration and shape (1,no.)   
        ### This function calculates Jacobian at every iteration, by creating matrix of shape self.jacdims & using sympy 
        ### to fill jacobian values to be dealt with 
        current_matrix = current_matrix.reshape((1,current_matrix.shape[0]))
        num_3D = self.world_to_imagematrix.shape[1]
        
        yies, xes = np.where(self.world_to_imagematrix>=0)
        #xes, yies = np.where(self.world_to_imagematrix>=0)
        Jacobian = np.zeros((2*xes.shape[0], current_matrix.shape[1]))
        
        for i, (x, y) in enumerate(zip(xes, yies)):
            X_, Y_, Z_ = current_matrix[0][3*y : 3*y+3]
            x_, y_, z_, theta_, p_, q_, r_ = current_matrix[0][3*num_3D + 7*x : 3*num_3D + 7*x + 7]            
            
            ## now calculating temporary jacobian matrix
            tempe = self.jacfun(X_,Y_,Z_,x_,y_,z_,theta_,p_,q_,r_)
            tempe = np.asarray(tempe)
            tempe = tempe.reshape((tempe.shape[0],tempe.shape[2])).T
            
            ## now putting values of temporary jacobian into final jacobian & calculating sparse_matrix
            Jacobian[2*i : 2*i+2, 3*y : 3*y+3] = np.array([tempe[0,:3],tempe[1,:3]])
            Jacobian[2*i : 2*i+2, 3*num_3D + 7*x : 3*num_3D + 7*x+7] = np.array([tempe[0,3:10],tempe[1,3:10]])
            #done
        return Jacobian
    
    def difference_for_ls(self, current_matrix):
        ''' First argument in least squares'''
        current_matrix = current_matrix.reshape((1,current_matrix.shape[0]))
        num_3D = self.world_to_imagematrix.shape[1]
        
        #xes, yies = np.where(self.world_to_imagematrix>=0)
        yies, xes = np.where(self.world_to_imagematrix>=0)
        
        imagep_hat_ordered = []
        
        for i, (x, y) in enumerate(zip(xes, yies)):
            X_, Y_, Z_ = current_matrix[0][3*y : 3*y+3]
            x_, y_, z_, theta_, p_, q_, r_ = current_matrix[0][3*num_3D + 7*x : 3*num_3D + 7*x + 7]
            
            ## now calculating image_point_hat (i.e. estimate) 
            tempo = self.xhat(X_,Y_,Z_,x_,y_,z_,theta_,p_,q_,r_)
            imagep_hat_ordered.append([tempo[0,0],tempo[1,0]])
            
        imagep_hat_ordered = np.asarray(imagep_hat_ordered).flatten('C')
        #self.imagep_hat_ordered = imagep_hat_ordered
        return self.ordered_imagep - imagep_hat_ordered
    
    def sparsity_and_imagep_calculator(self):
        '''
           Will be called only once during initialization.  
        '''
        xes, yies = np.where(self.world_to_imagematrix>=0)
        sparse_matrix = np.zeros((2*xes.shape[0], (3*self.world_to_imagematrix.shape[1]+7*self.world_to_imagematrix.shape[0])))
        ordered_imagep = []
        
        for i, (x, y) in enumerate(zip(xes, yies)):
            ordered_imagep.append([self.imagecoords[x][self.world_to_imagematrix[x,y]][0],self.imagecoords[x][self.world_to_imagematrix[x,y]][1]])
            
            ## now calculating sparse_matrix
            sparse_matrix[(2*i):(2*i+2), (3*y):(3*y+3)] = np.ones((2,3))
            sparse_matrix[(2*i):(2*i+2), (3*self.world_to_imagematrix.shape[1] + 7*x):(3*self.world_to_imagematrix.shape[1] + 7*x+7)] = np.ones((2,7))
            #done
            
        self.sparse_matrix = sparse_matrix
        ordered_imagep = np.asarray(ordered_imagep).flatten('C')    
        self.ordered_imagep = ordered_imagep
    
    def LeastSquaresCalculator(self):
        '''
        Calulates optimal values of World co-ordinates & R & t matrices
        '''
        result = least_squares(self.difference_for_ls, self.init_W_R_t.reshape(-1), self.JacCreator, method = "lm", jac_sparsity=self.sparse_matrix) 
        TunedParams = result.x
        IntWorldPnts = TunedParams[:self.initWorldp.shape[0]].reshape((self.initWorldp.shape[0]//3, 3))
        R_t_Matrices = []
        for R_t in TunedParams[self.initWorldp.shape[0]:].reshape((self.initRtvalues.shape[0], 7)):
            R_t_Matrices.append(self.create_P(R_t[0], R_t[1], R_t[2], R_t[3], R_t[4], R_t[5], R_t[6]))
            
        return np.array(R_t_Matrices), IntWorldPnts