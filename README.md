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

# For 1 ReLU example, data path is not used here
python alpha_beta_CROWN.py data1.pth toy
```
## Object
Given an input `x` and an L_inf perturbation, we have the upper bound and the lower bound of the input, noted as `x_U` and `x_L` respectively. Our goal is to compute the lower (or upper) bound for the output of the network. And further for robustness verification, we want to ensure the lower bound for output related to the true label is greater than the upper bound for any other false labels. For our code, it will first try **incomplete verifier CROWN and alpha-CROWN**. If not verified, then try **complete verifier alpha-beta-CROWN**. If all non-linear layers(ReLU) are split and still unsafe, then it is verified to be unsafe. If the amount of domains exceed the pre-defined threshold, then it is said to be unknown. Otherwise, it is verified to be safe. In the following several sections, we will simply introduce the algorithm and structure we used.

## CROWN
### Algorithm
In CROWN, we relax the non-linear layer(ReLU) to linear by bounding it using a linear upper bound and linear lower bound. For non-linear layar, we will first calculate its intermediate bound using backward propagation starting from previous linear layer. After having the intermediate bound for all non-linear layer, we can get two relaxed bounding functions of lower bound and upper bound for each non-linear neuron. Then we can get the relaxed bound for the whole model. The detailed rules and algorithm can be seen in CROWN ([Zhang et al. 2018](https://arxiv.org/pdf/1811.00866))

### Implementation
1. The whole model is converted to `BoundedSequential` object and each layer is converted to `BoundLinear` or `BoundReLU` based on its instance.
2. We will calculate the intermediate bound for each non-linear layer, by setting the `start node` to its previous layer and apply `boundpropogate`. The detailed steps of `boundpropogate` can be seen in CROWN ([Zhang et al. 2018](https://arxiv.org/pdf/1811.00866)). And you can also refer to the code of `boundpropogate` in `BoundReLU` and `BoundReLU`.
3. Having the intermediate bound for all non-linear layer, we then apply the `boundpropogate` from the last layer to get the bound for the whole model.
4. To notice that, as we are working on robustness verification, so we will use `C`, the initial coefficient matrix, to represent the output constraints. And we only care about the lower bound of the whole model. If we only want to get the bound, it should be an identity matrix.

## Alpha-CROWN
### Algorithm
Compared with CROWN, we make the bounding function flexible by using a trainable parameter `alpha` to replace the slope of bounding function for each nueron(For ReLU, we only need to replace the slope of lower bound). After optimizing the `alpha` towards the object, we can get tighter bound. The detailed rules and algorithm can be seen in alpha-CROWN ([Xu et al. 2021](https://arxiv.org/pdf/2011.13824))

### Implementation
1. Based on CROWN, we already have a `BoundedSequential` object. The initialization of `alpha` will be finalized when we run the CROWN. The `alpha` is initialized to the value of slope calculated in CROWN. And with different `start node`, we will have different `alpha`.
2.  During the iteration, will will first update the intermediate bound and then calculate the overall lower bound and upper bound. We will optimize the `alpha` by choosing `-lb.sum()` as loss function. Here we use Adam optimizer with a scheduler. Also, we will store the best `alpha` and tightest intermediate bound for later beta-CROWN. To notice that, if you want to find the bound, you should also optimize it towards minimizing the upper bound, by choosing `ub.sum()` as loss function.
## Beta-CROWN
### Algorithm
Compared with Alpha-CROWN, we use branch and bound to split unstable neurons and introduce a trainable parameter `beta` as a Lagarange multiplier to replace the constraint of split. With `beta` we are able to get tighter estimated bound when branch and bound is not complete, and also detect the conflict split during optimizing (notice that we also optimize `alpha` here). It is a complete verifier as we can finally split all unstable nuerons and get the true bound. The detailed rules and algorithm can be seen in beta-CROWN ([Wang et al. 2021](https://arxiv.org/pdf/2103.06624)).
### Implementation
1. We will first check the overall bound achieved from alpha-CROWN. A single lower bound, which we call instance here, is related to a single line in `C`. If it is larger than 0, then this instance is verified. Otherwise, we need further implement beta-CROWN on this instance.
2. We will first initialize the parameter, including using a temporary `C` related to this single instance, and modify the `alpha` and `beta` for this single instance. This step is done by a function `initialize_para` in `BoundedSequential`(all functions mentioned later are in this class). 
3. During the iteration, we will first split the domain by `general_split_robustness`. It will calculat babsr score which is introduced in [Branch and Bound for Piecewise Linear Neural Network
Verification(Bunel et al. 2020)](https://arxiv.org/pdf/2103.06624). We simply choose the neuron with largest score to split(if the `split depth` is not 1, choose topk scores). And the `split depth` will vary based on the size of remaining unverified domains. If the size is small, the `split depth` will be relevantly large. During this step, we will also modify the `alpha`, `beta`, `domains`, split logic `S` and `C`.
4. We will optimize `alpha` and `beta` towards minimizing the lower bound. Here we use Adam optimizer with a scheduler.
5. We will use `domain_filter_robusness` to remove the domains that are verified, which means its lower bound is larger than 0. In this step, we will also remove the parameters that are related to these domains. If it is not empty, which means there still exsists domains unverified, return to step 3 and repeat, until all domains are split(**unsafe**) or verified(**safe**). We will also end the loop when the size of domains exceed the threshold, with result **unknown**.
