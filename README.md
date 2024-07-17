# Alpha-Beta-CROWN-Robustness-Verification-Tutorial 
This repository contains a simple implementation of roboustness verification using CROWN, alpha-CROWN and beta-CROWN algorithms for fully connected sequential ReLU networks. 

## Prerequisites 
To run the tutorial, you need to install the `torch` and `numpy` library. You can install it using pip: 
```
pip install torch
pip install numpy
```

## Contents
- `alpha_beta_CROWN.py`: Implementation of CROWN, alpha-CROWN and beta-CROWN. Function about beta-CROWN including `Bab`, `general_split_robustness`, `domain_filter_robusness`, `optimized_beta_CROWN` and some related sub-function are defined inside `BoundedSequential` class. It will first try incomplete verifier CROWN and alpha-CROWN. If not verified, then try complete verifier beta-CROWN. If all split and still unsafe, then it is verified to be unsafe. If the split domain exceed the pre-defined threshold, then it is unknown. Otherwise, it is verified to be safe.
- `linear.py`: Definition of `BoundLinear` class. 
- `relu_alpha_beta.py`: Definition of `BoundReLU` class. Most functions about ReLU are defined inside the  class.
- `model.py`: PyTorch model definition.  
- `models/relu_model.pth`: Pretrained model for debugging. It is only for the relavent complex model, and the simple toy model is defined inside `model.py`. 

# Run
```
# For 2 ReLU example
python alpha_beta_CROWN.py data1.pth complex

# For 1 ReLU exmple, data path is not used here
python alpha_beta_CROWN.py data1.pth toy
```
# CROWN
## Algorithm

## Implementation

# Alpha-CROWN
## Algorithm

## Implementation

# Beta-CROWN
## Algorithm

## Implementation
