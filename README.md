# Multi View 3D-Reconstruction
This repo contains the code for 3D reconstruction of objects from multiple images of different views. The implementation follows <a href=https://inst.eecs.berkeley.edu/~cs194-26/fa17/upload/files/projFinalProposed/cs194-26-ace/>this post</a>.
Implementation is divided into following parts:

1) Feature Matching : In this section, each image is searched for interest points using `Multilevel Harris Corner Detection` and for each interest point a `SIFT Descriptor` is generated. For all possible pairs of images, these descriptors are crosschecked with each other to find the match between the correspoinding interest points. (This section needs more management)

2) Fundamental Matrix & Camera Matrix : It is assumed all the images are taken from same camera(that means intrinsic parameters will be same for all images). For all pair of images, a Fundamental Matrix `F` will be calculated using point correspondence generated in last section. Then, using intrinsic matrix `K`, we find Essential Matrix `E = np.matmul(K.T, np.matmul(F, K))`. Factorizing this `E` matrix we will get extrinsic matrix `P = [R | t]` of each image view.




https://user-images.githubusercontent.com/39570428/150210136-789de768-0523-43f9-8f1f-ee2739a0af21.mp4

