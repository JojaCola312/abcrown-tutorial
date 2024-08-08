import torch
import torch.nn as nn
from collections import OrderedDict
import copy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class BoundReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super(BoundReLU, self).__init__(inplace)
        self.alpha_u = OrderedDict()
        self.alpha_l = OrderedDict()
        self.beta_u = OrderedDict()
        self.beta_l = OrderedDict()
        self.init_u = None
        self.init_l = None
        self.init_beta_u = None
        self.init_beta_l = None

        self.S = None
        self.beta_l_list = OrderedDict()
        self.alpha_l_list = OrderedDict()
        self.beta_u_list = OrderedDict()
        self.alpha_u_list = OrderedDict()

        self.last_lA = None
        self.last_uA = None
        self.lb = None
        self.ub = None
        
        self.last_lA_list = OrderedDict()
        self.last_uA_list = OrderedDict()

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

            optimize (int): 0 for CROWN, 1 for alpha-CROWN, 2 for alpha-beta-CROWN

            out_features (int): the size of output

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
        upper_d = upper_d.unsqueeze(0)
        
        # Lower bound: 0 if |lb| < |ub|, 1 otherwise.
        # Equivalently we check whether the slope of the upper bound is > 0.5.
        lower_d = (upper_d > 0.5).float()

        if(optimize == 0 and self.init_l == None):
            self.init_l = lower_d.clone().detach()
            self.init_u = lower_d.clone().detach()
            self.init_beta_l = torch.zeros_like(lower_d)
            self.init_beta_u = torch.zeros_like(lower_d)

        if(start_node not in self.alpha_l):
            self.alpha_l[start_node] = self.init_l.repeat(1, out_features, 1)
            self.alpha_u[start_node] = self.init_u.repeat(1, out_features, 1)
            self.alpha_l[start_node] = nn.Parameter(self.alpha_l[start_node])
            self.alpha_l[start_node].requires_grad_()
            self.alpha_u[start_node] = nn.Parameter(self.alpha_u[start_node])
            self.alpha_u[start_node].requires_grad_()



            self.beta_l[start_node] = self.init_beta_l.repeat(1, out_features, 1)
            self.beta_u[start_node] = self.init_beta_u.repeat(1, out_features, 1)
            self.beta_l[start_node] = nn.Parameter(self.beta_l[start_node])
            self.beta_l[start_node].requires_grad_()
            self.beta_u[start_node] = nn.Parameter(self.beta_u[start_node])
            self.beta_u[start_node].requires_grad_()

        if(optimize == 2):
            S = self.S
        else:
            S = torch.zeros(self.beta_l[start_node].shape)

        lb_lower_d = self.alpha_l[start_node].clone().detach()
        ub_lower_d = self.alpha_u[start_node].clone().detach()

        self.inact_alpha(start_node)
        lb_lower_d[0] = self.alpha_l[start_node][0]
        ub_lower_d[0] = self.alpha_u[start_node][0]

        lb_beta = self.beta_l[start_node].clone().detach()
        ub_beta = self.beta_u[start_node].clone().detach()
        lb_beta[0] = self.beta_l[start_node][0]
        ub_beta[0] = self.beta_u[start_node][0]
        uA = lA = None
        ubias = lbias = 0
        S = S.to(device)
        

        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            # Choose upper or lower bounds based on the sign of last_A
            # New linear bound coefficent.
            uA = upper_d * pos_uA + ub_lower_d * neg_uA + ub_beta * S
            # New bias term. Adjust shapes to use matmul (better way is to use einsum).
            mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            # ubias = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            ubias = (mult_uA * upper_b).sum(dim=-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            # Choose upper or lower bounds based on the sign of last_A
            # New linear bound coefficent.
            lA = upper_d * neg_lA + lb_lower_d * pos_lA + lb_beta * S
            # New bias term.
            mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            # lbias = mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            lbias = (mult_lA * upper_b).sum(dim=-1)
        
        self.last_lA = last_lA
        self.last_uA = last_uA

        return uA, ubias, lA, lbias

    def clip_alpha(self):
        r"""Clip alphas after an single update.
        Alpha should be bewteen 0 and 1.
        """
        for v in self.alpha_l.values():
            v.data = torch.clamp(v.data, 0, 1)
        for v in self.alpha_u.values():
            v.data = torch.clamp(v.data, 0, 1)
        
    def clip_beta(self):
        r"""Clip betas after an single update.
        Beta should be non-negative.
        """
        for v in self.beta_l.values():
            v.data = torch.clamp(v.data, min=0)
        for v in self.beta_u.values():
            v.data = torch.clamp(v.data, min=0)

    def update(self, name, optimize, mask):
        r"""update the stored alpha, beta and last_lA list, usually store the best"""
        if(optimize == 2):
            self.last_lA_list[name] = self.last_lA.clone()
            self.last_uA_list[name] = self.last_uA.clone()
        else:
            if(name not in self.beta_l_list.keys()):
                self.beta_l_list[name] = copy.deepcopy(self.beta_l)
                self.alpha_l_list[name] = copy.deepcopy(self.alpha_l)
                self.beta_u_list[name] = copy.deepcopy(self.beta_u)
                self.alpha_u_list[name] = copy.deepcopy(self.alpha_u)
            else:
                for key in self.beta_l_list[name].keys():
                    if(key != self.final_start_node):
                        self.beta_l_list[name][key] = copy.deepcopy(self.beta_l[key])
                        self.alpha_l_list[name][key] = copy.deepcopy(self.alpha_l[key])

                        self.beta_u_list[name][key] = copy.deepcopy(self.beta_u[key])
                        self.alpha_u_list[name][key] = copy.deepcopy(self.alpha_u[key])
                    else:
                        with torch.no_grad():
                            self.beta_l_list[name][key][:,mask.squeeze(0),:] = self.beta_l[key][:,mask.squeeze(0),:].clone().detach()
                            self.alpha_l_list[name][key][:,mask.squeeze(0),:] = self.alpha_l[key][:,mask.squeeze(0),:].clone().detach()

                            self.beta_u_list[name][key][:,mask.squeeze(0),:] = self.beta_u[key][:,mask.squeeze(0),:].clone().detach()
                            self.alpha_u_list[name][key][:,mask.squeeze(0),:] = self.alpha_u[key][:,mask.squeeze(0),:].clone().detach()
                        

                self.last_lA_list[name] = self.last_lA.clone()
                self.last_uA_list[name] = self.last_uA.clone()



    def inact_alpha(self, start_node):
        r"""fix the alpha that should be 1 or 0"""
        lb = self.lower_l
        ub = self.upper_u
        mask_neg1 = (lb >= 0)
        mask_pos1 = (ub <= 0)

        with torch.no_grad():
            self.alpha_l[start_node][0][mask_neg1.expand_as(self.alpha_l[start_node][0])] = 1
            self.alpha_u[start_node][0][mask_neg1.expand_as(self.alpha_u[start_node][0])] = 1
            self.alpha_l[start_node][0][mask_pos1.expand_as(self.alpha_l[start_node][0])] = 0
            self.alpha_u[start_node][0][mask_pos1.expand_as(self.alpha_u[start_node][0])] = 0
