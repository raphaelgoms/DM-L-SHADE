# DM-L-SHADE

This repository present the code of a proposal of hybridization of the evolutionary algorithm L-SHADE (Success-History based Adaptive Differential Evolution with  Linear Population Size Reduction) with Data Mining methods. 

The implementation was written above the L-SHADE code of Tanabe and Fukunaga [[1](#references)], available [HERE](https://ryojitanabe.github.io/code/LSHADE1.0.1_CEC2014.zip).

In our implementation L-SHADE was hybridized with two Clustering Methods (K-means [[2](#references)] and X-means [[3](#references)]). But, we pretend try the hybridization with other DM methods. The implementation of clustering algorithms used is from the [Pyclustering Library](https://github.com/annoviko/pyclustering).

## References

[1] Tanabe, Ryoji and Alex S. Fukunaga. “Improving the search performance of SHADE using linear population size reduction.” *2014 IEEE Congress on Evolutionary Computation (CEC)* (2014): 1658-1665. (Available [HERE](https://ryojitanabe.github.io/pdf/tf-cec2014.pdf)).

[2] R. Duda and P. Hart. Pattern Classification and Scene Analysis. John
Wiley & Sons, 1973.

[3] D. Pelleg and A. W. Moore. “X-means: Extending K-means with Efficient Estimation of the Number of Clusters”. In *Proceedings of the Seventeenth International Conference on Machine Learning (ICML '00)*. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 2000, pp. 727–734 (Available [HERE](https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf)).
