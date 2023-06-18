# Approximate-Online-Convolutional-Dictionary-Learning
MATLAb implementations of algorithms proposed in "An Efficient Approximate Method for Online Convolutional Dictionary Learning"
https://arxiv.org/abs/2301.10583

and in conference paper "Efficient Online Convolutional Dictionary Learning Using Approximate Sparse Components", ICASSP, 2023




(I) For the experiments on City and Fruit datasets use the codes below:
1- script_proposed_fruit_city.m


(II) For the experiments on SIPI and Flickr datasets use the codes below:
(use K=100 for Flickr and K=80 for SIPI)
1- script_proposed_flickr_SIPI.m


(III) For the experiments on Flickr-large dataset use the codes below:
(use K=100)
1- script_proposed_flickr_large.m   (the proposed algorithm)
2- script_test_large_dataset.m     (code for testing the dictionaries) comment/uncomment alg1 and alg2 to load the dictionaries


(IV) For learning large dictionarues on Flickr dataset use the codes below:
(use K=200, 300, and 400)
1- script_proposed_flickr_SIPI.m   (the proposed algorithm)
