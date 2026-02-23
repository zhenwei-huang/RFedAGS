STRUCTURES:

The main files are located in "./mycode". Call it PATH for short.
In PATH, there are 8 folders (data, FMCSPD_PATHMNIST, HSPHyper_WordNet, LRMCGrass_MovieLens, LRMCGrass_Syn, PCASphere_MNIST,
PCAStief_CIFAR10, and PCAStief_Syn) and 10 function files of matlab.

Folders: each folder (as its name implies) includes the Matlab scripts reproducing the results corresponding to the figure 
in this paper. 


Functions:
-----
*) PATH/RFedAGS_.m  : the implementation of Riemannian Federated Averaging Gradient Stream.
-----
*) PATH/RFedAvg_.m  : the implementation of Riemannian Federated Averaging draw from [LM23, Algorithm 2].
-----
*) PATH/RFedProj_.m : the implementation of Algorithm 1 draw from [ZHSJ24].
-----
*) PATH/RFedSVRG_.m : the implementation of Riemannian Federated SVRG draw from [LM23, Algorithm 1].
-----
*) PATH/ConstructRFLFMCProb_SPD.m    : the implementation of constructing the problem of computing Frechet mean of points 
   over the SPD manifold.
-----
*) PATH/ConstructRFLHSPProb_Hyper    : the implementation of constructing the problem of hyperbolic structured prediction 
   over the hyperbolic manifold.
-----
*) PATH/ConstructRFLPCAProb_Stief.m  : the implementation of constructing the problem of PCA over the Stiefel manifold. 
-----
*) PATH/ConstructRFLLRMCProb_Grass.m : the implementation of constructing the problem of low-rank matrix completion over 
   the Grassmann manifold. 


Others: 
*) "manopt" is a Riemannian optimization toolbox [BMAS14]. The Riemannian operations in our code mostly are based on this 
   toolbox. 

HOW TO RUN THE CODE:

*) Run importDir.m to import all the necessary directories.
*) Run all of the testing file to conduct the same experiments as that in the paper.


----------------------------------------------------------------------------------------------------------------

[LM23]   Jiaxiang Li and Shiqian Ma. Federated learning On Riemannian manifolds. Applied Set-Valued Analysis and 
         Optimization, 5(2):213-232, 08 2023.

[ZHSJ24] Jiaojiao Zhang, Jiang Hu, Anthony Man-Cho So, and Mikael Johansson. Nonconvex federated learning on 
         compact smooth submanifolds with heterogeneous data. In Aduances in Neural Information Processing Systems 
         (NeurIPS), 2024.

[BMAS14] Nicolas Boumal, Bamdev Mishra, P-A Absil, and Rodolphe Sepulchre. Manopt, a Matlab toolbox for optimization
         on manifolds. The Journal of Machine Learning Research, 15(1):1455-1459, 2014.


