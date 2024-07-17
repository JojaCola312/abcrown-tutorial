# Alpha-Beta-CROWN Robustness Verification Tutorial 
This repository contains a simple implementation of roboustness verification using CROWN, alpha-CROWN and beta-CROWN algorithms for fully connected sequential ReLU networks. 

## Prerequisites 
To run the tutorial, you need to install the `torch` and `numpy` library. You can install it using pip: 
```
pip install torch
pip install numpy
```

## Contents
- `alpha_beta_CROWN.py`: Implementation of CROWN, alpha-CROWN and beta-CROWN. Function about beta-CROWN including `Bab`, `general_split_robustness`, `domain_filter_robusness`, `optimized_beta_CROWN` and some related sub-function are defined inside `BoundedSequential` class.
- `linear.py`: Definition of `BoundLinear` class. 
- `relu_alpha_beta.py`: Definition of `BoundReLU` class. Most functions about ReLU are defined inside the  class.
- `model.py`: PyTorch model definition.  
- `models/relu_model.pth`: Pretrained model for debugging. It is only for the complex 2 ReLU model, and the parameter of simple 1 ReLU toy model is defined inside `model.py`. 

## Run
```
# For 2 ReLU example
python alpha_beta_CROWN.py data1.pth complex

# For 1 ReLU exmple, data path is not used here
python alpha_beta_CROWN.py data1.pth toy
```
## Object
Given an input `x` and an L_inf perturbation, we have the upper bound and the lower bound of the input, noted as `x_U` and `x_L` respectively. Our goal is to compute the lower (or upper) bound for the output of the network. And further for robustness verification, we want to ensure the lower bound for output related to the true label is greater than the upper bound for any other false labels. For our code, it will first try **incomplete verifier CROWN and alpha-CROWN**. If not verified, then try **complete verifier alpha-beta-CROWN**. If all non-linear layer(ReLU) is split and still unsafe, then it is verified to be unsafe. If the amount of domains exceed the pre-defined threshold, then it is said to be unknown. Otherwise, it is verified to be safe. In the following several sections, we will simply introduce the algorithm and structure we used.

## CROWN
### Algorithm
In CROWN, we relax the non-linear layer(ReLU) to linear by bounding it using a linear upper bound and linear lower bound. For non-linear layar, we will first calculate its intermediate bound using backward propagation starting from previous linear layer. After having the intermediate bound for all non-linear layer, we can get two relaxed bounding functions of lower bound and upper bound for each non-linear neuron. Then we can get the relaxed bound for the whole model. The detailed rules and algorithm can be seen in CROWN ([Zhang et al. 2018](https://arxiv.org/pdf/1811.00866))

### Implementation

## Alpha-CROWN
### Algorithm
Compared with CROWN, we make the bounding function flexible by using a trainable parameter `alpha` to replace the slope of bounding function for each nueron(For ReLU, we only need to replace the slope of lower bound). After optimizing the `alpha` towards the object, we can get tighter bound. The detailed rules and algorithm can be seen in alpha-CROWN ([Xu et al. 2021](https://arxiv.org/pdf/2011.13824))

### Implementation

## Beta-CROWN
### Algorithm

### Implementation
