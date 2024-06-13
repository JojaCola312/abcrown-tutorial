import torch
import torch.nn as nn


class BoundHardTanh(nn.Hardtanh):
    def __init__(self,min=-1,max=1,inplace=False):
        super(BoundHardTanh, self).__init__(min,max,inplace)
        self.alpha_u = None
        self.alpha_l = None
        # self.alpha_u = torch.zeros_like(self.lower_u, requires_grad=True)
        # self.alpha_l = torch.zeros_like(self.lower_l, requires_grad=True)

    @staticmethod
    def convert(act_layer):
        r"""Convert a HardTanh layer to BoundHardTanh layer

        Args:
            act_layer (nn.HardTanh): The HardTanh layer object to be converted.

        Returns:
            l (BoundHardTanh): The converted layer object.
        """
        # TODO: Return the converted HardTanH
        l=BoundHardTanh(min=act_layer.min_val, max=act_layer.max_val, inplace=act_layer.inplace)
        return l

    def boundpropogate(self, last_uA, last_lA, start_node=None, optimize=False):
        """
        Propagate upper and lower linear bounds through the HardTanh activation function
        based on pre-activation bounds.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.

        """
        # These are preactivation bounds that will be used for form the linear relaxation.
        preact_lb = self.lower_l
        preact_ub = self.upper_u

        """
         Hints: 
         1. Have a look at the section 3.2 of the CROWN paper [1] (Case Studies) as to how segments are made for multiple activation functions
         2. Look at the HardTanH graph, and see multiple places where the pre activation bounds could be located
         3. Refer the ReLu example in the class and the diagonals to compute the slopes/intercepts
         4. The paper talks about 3 segments S+, S- and S+- for sigmoid and tanh. You should figure your own segments based on preactivation bounds for hardtanh.
         [1] https://arxiv.org/pdf/1811.00866.pdf
        """

        # You should return the linear lower and upper bounds after propagating through this layer.
        # Upper bound: uA is the coefficients, ubias is the bias.
        # Lower bound: lA is the coefficients, lbias is the bias.
        # upper_d = torch.zeros_like(preact_ub)
        # upper_b = torch.zeros_like(preact_ub)
        # lower_b = torch.zeros_like(preact_lb)
        # lower_d = torch.zeros_like(preact_lb)
        upper_d = preact_ub.clone()
        upper_b = preact_ub.clone()
        lower_b = preact_lb.clone()
        lower_d = preact_lb.clone()
        if(optimize == False and self.alpha_u == None):
            self.alpha_u = torch.zeros_like(self.upper_u)
            self.alpha_l = torch.zeros_like(self.lower_l)
            for i in range(len(preact_lb[0])):
                l, u = preact_lb[0,i], preact_ub[0,i]
                # print(l,u)
                if(u<=-1):
                    self.alpha_l[0,i], self.alpha_u[0,i] = 0, 0
                elif(l>=1):
                    self.alpha_l[0,i], self.alpha_u[0,i] = 0, 0
                elif(l>=-1 and u<=1):
                    self.alpha_l[0,i], self.alpha_u[0,i] = 1, 1
                elif(l<=-1 and u<=1 and u>=-1):
                    self.alpha_l[0,i], self.alpha_u[0,i] = (u+1)/(u-l), (u+1)/(u-l)
                elif(l>=-1 and u>=1 and l<=1):
                    self.alpha_l[0,i], self.alpha_u[0,i] = (1-l)/(u-l), (1-l)/(u-l)
                elif(l<=-1 and u>=1):
                    self.alpha_l[0,i], self.alpha_u[0,i] = 2/(u+1), 2/(1-l)
            self.alpha_l = nn.Parameter(self.alpha_l)
            self.alpha_u = nn.Parameter(self.alpha_u)
            self.alpha_u.requires_grad_()
            self.alpha_l.requires_grad_()
        for i in range(len(preact_lb[0])):
            l, u = preact_lb[0,i], preact_ub[0,i]
            # print(l,u)
            case=0
            if(u<=-1):
                case=1
                lower_b[0,i], upper_b[0,i] = -1, -1
                lower_d[0,i], upper_d[0,i] = 0, 0
            elif(l>=1):
                case=2
                lower_b[0,i], upper_b[0,i] = 1, 1
                lower_d[0,i], upper_d[0,i] = 0, 0
            elif(l>=-1 and u<=1):
                case=3
                lower_b[0,i], upper_b[0,i] = 0, 0
                lower_d[0,i], upper_d[0,i] = 1, 1
            elif(l<=-1 and u<=1 and u>=-1):
                case=4
                lower_b[0,i], upper_b[0,i] = (1+l)/(u-l), (-u-u*l)/(u-l)
                lower_d[0,i], upper_d[0,i] = (u+1)/(u-l), (u+1)/(u-l)

                lower_b[0,i], upper_b[0,i] = max(min(self.alpha_l[0,i],torch.tensor(1)),torch.tensor(0))-1, (-u-u*l)/(u-l)
                lower_d[0,i], upper_d[0,i] = max(min(self.alpha_l[0,i],torch.tensor(1)),torch.tensor(0)), (u+1)/(u-l)

            elif(l>=-1 and u>=1 and l<=1):
                case=5
                lower_b[0,i], upper_b[0,i] = (u*l-l)/(u-l), (u-1)/(u-l)
                lower_d[0,i], upper_d[0,i] = (1-l)/(u-l), (1-l)/(u-l)

                lower_b[0,i], upper_b[0,i] = (u*l-l)/(u-l), 1-max(min(self.alpha_u[0,i],torch.tensor(1)),torch.tensor(0))
                lower_d[0,i], upper_d[0,i] = (1-l)/(u-l), max(min(self.alpha_u[0,i],torch.tensor(1)),torch.tensor(0))

            elif(l<=-1 and u>=1):
                case=6
                lower_b[0,i], upper_b[0,i] = (1-u)/(u+1), (-l-1)/(1-l)
                lower_d[0,i], upper_d[0,i] = 2/(u+1), 2/(1-l)

                lower_b[0,i], upper_b[0,i] = max(min(self.alpha_l[0,i],2/(u+1)),torch.tensor(0))-1, 1-max(min(self.alpha_u[0,i],2/(1-l)),torch.tensor(0))
                lower_d[0,i], upper_d[0,i] = max(min(self.alpha_l[0,i],2/(u+1)),torch.tensor(0)), max(min(self.alpha_u[0,i],2/(1-l)),torch.tensor(0))


            # if(lower_d[0,i]*l+lower_b[0,i]>upper_d[0,i]*l+upper_b[0,i] or lower_d[0,i]*u+lower_b[0,i]>upper_d[0,i]*u+upper_b[0,i]):
            #     print(case,max(min(self.alpha_u[0,i],1),0))
            #     print(lower_d[0,i]*l+lower_b[0,i],upper_d[0,i]*l+upper_b[0,i])
            #     print(lower_d[0,i]*u+lower_b[0,i],upper_d[0,i]*u+upper_b[0,i])

#just copied from relu and add the operation on lower_d which is missing in relu
        uA = lA = None
        ubias = lbias = 0

        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            # Choose upper or lower bounds based on the sign of last_A
            # New linear bound coefficent.
            uA = upper_d * pos_uA + lower_d * neg_uA
            # New bias term. Adjust shapes to use matmul (better way is to use einsum).
            mult_uA_pos = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            mult_uA_neg = neg_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_uA_pos.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)+mult_uA_neg.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            # Choose upper or lower bounds based on the sign of last_A
            # New linear bound coefficent.
            lA = upper_d * neg_lA + lower_d * pos_lA
            # New bias term.
            mult_lA_neg = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            mult_lA_pos = pos_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_lA_neg.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)+mult_lA_pos.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)

        return uA, ubias, lA, lbias

