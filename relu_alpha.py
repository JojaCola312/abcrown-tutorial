import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class BoundReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super(BoundReLU, self).__init__(inplace)
        self.alpha_u = OrderedDict()
        self.alpha_l = OrderedDict()
        self.init_u = None
        self.init_l = None
    @staticmethod
    def convert(act_layer):
        r"""Convert a ReLU layer to BoundReLU layer

        Args:
            act_layer (nn.ReLU): The ReLU layer object to be converted.

        Returns:
            l (BoundReLU): The converted layer object.
        """
        l = BoundReLU(act_layer.inplace)
        return l

    def boundpropogate(self, last_uA, last_lA, start_node=None, optimize=False, out_features = 10):
        r"""Bound propagate through the ReLU layer.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation.

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.
        """
        # lb_r and ub_r are the bounds of input (pre-activation)
        # Here the clamping oepration ensures the results are correct for stable neurons.
        lb_r = self.lower_l.clamp(max=0)
        ub_r = self.upper_u.clamp(min=0)

        
        # avoid division by 0 when both lb_r and ub_r are 0
        ub_r = torch.max(ub_r, lb_r + 1e-8)

        # CROWN upper and lower linear bounds
        upper_d = ub_r / (ub_r - lb_r)  # slope
        upper_b = - lb_r * upper_d  # intercept
        upper_d = upper_d.unsqueeze(1)
        
        # Lower bound: 0 if |lb| < |ub|, 1 otherwise.
        # Equivalently we check whether the slope of the upper bound is > 0.5.
        lower_d = (upper_d > 0.5).float()

        if(optimize == False and self.init_l == None):
            self.init_l = lower_d.clone().detach()
            # self.alpha_l = nn.Parameter(self.alpha_l)
            # self.alpha_l.requires_grad_()

            self.init_u = lower_d.clone().detach()
            # self.alpha_u = nn.Parameter(self.alpha_u)
            # self.alpha_u.requires_grad_()

        if(start_node not in self.alpha_l):
            self.alpha_l[start_node] = self.init_l.repeat(1, out_features, 1)
            self.alpha_u[start_node] = self.init_u.repeat(1, out_features, 1)
            self.alpha_l[start_node] = nn.Parameter(self.alpha_l[start_node])
            self.alpha_l[start_node].requires_grad_()
            self.alpha_u[start_node] = nn.Parameter(self.alpha_u[start_node])
            self.alpha_u[start_node].requires_grad_()
        # print(self.alpha_l)
        # for i in range(len(lower_d[0,0])):
        lb_lower_d = self.alpha_l[start_node].clone().detach()
        ub_lower_d = self.alpha_u[start_node].clone().detach()
        lb_lower_d[0] = self.alpha_l[start_node][0]
        ub_lower_d[0] = self.alpha_u[start_node][0]
        # print(lb_lower_d.shape)
        # lower_d = self.alpha_l

        uA = lA = None
        ubias = lbias = 0
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            # Choose upper or lower bounds based on the sign of last_A
            # New linear bound coefficent.
            uA = upper_d * pos_uA + lb_lower_d * neg_uA
            # New bias term. Adjust shapes to use matmul (better way is to use einsum).
            mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            # Choose upper or lower bounds based on the sign of last_A
            # New linear bound coefficent.
            lA = upper_d * neg_lA + ub_lower_d * pos_lA
            # New bias term.
            mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)

        return uA, ubias, lA, lbias

    def clip_alpha(self):
        r"""Clip alphas after an single update.
        Alpha should be bewteen 0 and 1.
        """
        for v in self.alpha_l.values():
            v.data = torch.clamp(v.data, 0, 1)
        for v in self.alpha_u.values():
            v.data = torch.clamp(v.data, 0, 1)
