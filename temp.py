import torch
import torch.nn as nn
import numpy as np
from model import SimpleNNRelu, SimpleNNHardTanh, two_relu_toy_model
from linear import BoundLinear
# from relu import BoundReLU
from relu_alpha_beta import BoundReLU
from hardTanh_question_alpha import BoundHardTanh
import time
import argparse
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def deep_copy_structure(structure):
    if isinstance(structure, torch.Tensor):
        return structure.clone().detach()
    elif isinstance(structure, list):
        return [deep_copy_structure(item) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(deep_copy_structure(item) for item in structure)
    else:
        return structure

def zero_copy_structure(structure):
    if isinstance(structure, torch.Tensor):
        return torch.zeros_like(structure)
    elif isinstance(structure, list):
        return [zero_copy_structure(item) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(zero_copy_structure(item) for item in structure)
    else:
        return structure

class BoundedSequential(nn.Sequential):
    r"""This class wraps the PyTorch nn.Sequential object with bound computation."""

    def __init__(self, *args):
        super(BoundedSequential, self).__init__(*args)

    @staticmethod
    def convert(seq_model):
        r"""Convert a Pytorch model to a model with bounds.
        Args:
            seq_model: An nn.Sequential module.

        Returns:
            The converted BoundedSequential module.
        """
        layers = []
        for l in seq_model:
            if isinstance(l, nn.Linear):
                layers.append(BoundLinear.convert(l))
            elif isinstance(l, nn.ReLU):
                layers.append(BoundReLU.convert(l))
            elif isinstance(l, nn.Hardtanh):
                layers.append(BoundHardTanh.convert(l))
        return BoundedSequential(*layers)

    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=0):
        r"""Main function for computing bounds.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            optimize (bool): Whether we optimize alpha.

        Returns:
            ub (tensor): The upper bound of the final output.

            lb (tensor): The lower bound of the final output.
        """
        ub = lb = None
        ub, lb = self.full_boundpropogation(x_U=x_U, x_L=x_L, upper=upper, lower=lower, optimize=optimize)
        return ub, lb

    def pick_out(self, P, batch_size, split_bool_list):
        return P[min(len(P),batch_size):], P[:min(len(P),batch_size)], split_bool_list[min(len(P),batch_size):], split_bool_list[:min(len(P),batch_size)]

    def split(self, C_list, split_lists):
        C_split = []
        split_bool_list = []
        name_list = []
        def is_valid_diagonal_matrix(matrix):
            n = matrix.shape[0]
            diag_mask = torch.eye(n, dtype=bool)
            diag_elements = matrix[diag_mask]
            off_diag_elements = matrix[~diag_mask]
            if not torch.all(off_diag_elements == 0):
                return False
            if not torch.all((diag_elements == 1) | (diag_elements == -1)):
                return False
            return True
        for P, split_list in zip(C_list, split_lists):
            all_valid = all(is_valid_diagonal_matrix(matrix) for matrix in split_list)
            if(all_valid):
                continue
            _, _, C, name = P
            num_relu = len(C)
            C_1, C_2 = deep_copy_structure(C), deep_copy_structure(C)
            split_bool_1, split_bool_2 = deep_copy_structure(split_list), deep_copy_structure(split_list)
            rank_list = []
            i = 0
            #each C is the bound for 
            for single_C in C:
                lb, ub = single_C[0].detach().cpu().numpy(), single_C[1].detach().cpu().numpy()
                mask = (lb < 0) & (ub > 0)
                diff = ub - lb
                masked_diff = np.where(mask, diff, -np.inf)
                max_index = np.unravel_index(np.argmax(masked_diff), masked_diff.shape)
                max_diff = masked_diff[max_index]
                rank_list.append((max_diff,max_index,i))
                i += 1
            sorted_rank = sorted(rank_list, key=lambda x: x[0], reverse=True)
            max_range_term = sorted_rank[0]
            C_1[max_range_term[2]][1][max_range_term[1]] = 0
            C_2[max_range_term[2]][0][max_range_term[1]] = 0

            # split_bool_1[max_range_term[2]][1][max_range_term[1]] = 1
            # split_bool_1[max_range_term[2]][0][max_range_term[1]] = 1
            # split_bool_2[max_range_term[2]][1][max_range_term[1]] = 1
            # split_bool_2[max_range_term[2]][0][max_range_term[1]] = 1
            # numpy_list = [tensor.cpu().numpy() for tensor in split_bool_1]
            # nonzero_counts = [np.count_nonzero(arr) for arr in numpy_list]
            # print('before',nonzero_counts)
            # print(max_range_term[1][1])
            split_bool_1[max_range_term[2]][max_range_term[1][1],max_range_term[1][1]] = 1
            split_bool_2[max_range_term[2]][max_range_term[1][1],max_range_term[1][1]] = -1

            # numpy_list = [tensor.cpu().numpy() for tensor in split_bool_1]
            # nonzero_counts = [np.count_nonzero(arr) for arr in numpy_list]
            # print('after',nonzero_counts)
            C_split.append(C_1)
            C_split.append(C_2)
            split_bool_list.append(split_bool_1)
            split_bool_list.append(split_bool_2)
            name_list.append(name*2+1)
            name_list.append(name*2+2)

        return C_split, split_bool_list, name_list

    def compute_bound(self, P, b, method):
        d = 'OK'
        if(len(P)==0):
            print('empty P')
            d = 'END'
        

        if(method == "lower"):
            lower_bound = b.clone().detach()
            for i in range(len(lower_bound[0])):
                for j in range(len(P)):
                    if(P[j][0][0][i] > lower_bound[0][i]):
                        lower_bound[0][i] = P[j][0][0][i]
            return lower_bound, d
        elif(method == "upper"):
            upper_bound = b.clone().detach()
            for i in range(len(upper_bound[0])):
                for j in range(len(P)):
                    if(P[j][1][0][i] < upper_bound[0][i]):
                        upper_bound[0][i] = P[j][1][0][i]
            return upper_bound, d

    def optimized_beta_CROWN(self, C, split, x_U=None, x_L=None, upper=True, lower=True, optimize=1, name=0):
        print('name:', name)
        modules = list(self._modules.values())
        # CROWN propagation for all layers
        ind = 0
        # for module in modules:
        #     if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
        #         print(module.beta_l.values())
        #         print(module.beta_u.values())
        for k, node in enumerate(modules):
            if isinstance(node, BoundReLU):
                node.initialize_beta(name)
                node.initialize_alpha(name)
        # for module in modules:
        #     if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
        #         print(module.beta_l.values())
        #         print(module.beta_u.values())
        # numpy_list = [tensor.cpu().numpy() for tensor in split]
        # nonzero_counts = [np.count_nonzero(arr) for arr in numpy_list]
        # print(nonzero_counts)

        for i in range(len(modules)):
            # We only need the bounds before a ReLU/HardTanh layer
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                modules[i].upper_u = C[ind][1]
                modules[i].lower_l = C[ind][0]
                modules[i].S = split[ind]

                nonzero_counts = np.count_nonzero(modules[i].S.cpu())
                # print(i, nonzero_counts)
                ind += 1
        # Get the final layer bound
        ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                              C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper,
                                              lower=lower, start_node=i, optimize=0)
        if(optimize==0):
            print('using CROWN')
            return ub, lb
        elif(optimize==1):
            print('using alpha-CROWN')
            opt = False
            for module in modules:
                if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
                    lr = 0.1
                    if(opt == False):
                        optimizer = optim.Adam(list(module.alpha_l.values()), lr=lr)
                        optimizer.add_param_group({'params': list(module.alpha_u.values()),'lr': lr})
                        opt = True
                    else:
                        optimizer.add_param_group({'params': list(module.alpha_l.values()),'lr': lr})
                        optimizer.add_param_group({'params': list(module.alpha_u.values()),'lr': lr})
            iter = 20
            best_lb, best_ub = lb, ub
            best_loss = np.inf
            for j in range(iter):
                # print(lb)
                best_lb = torch.max(best_lb , lb)
                best_ub = torch.min(best_ub , ub)

                for i in range(len(modules)):
                    # We only need the bounds before a ReLU/HardTanh layer
                    if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                        if isinstance(modules[i - 1], BoundLinear):
                            # add a batch dimension
                            newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                            # Use CROWN to compute pre-activation bounds
                            # starting from layer i-1
                            ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                                    start_node=i - 1, optimize=0, out_features = modules[i - 1].out_features)
                        # Set pre-activation bounds for layer i (the ReLU layer)
                        modules[i].upper_u = ub
                        modules[i].lower_l = lb
                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper,
                                                lower=lower, start_node=i, optimize=True, out_features = modules[i].out_features)
                optimizer.zero_grad(set_to_none=True)
                loss = -lb.sum() 
                if(loss < best_loss):
                    for module in modules:
                        if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
                            module.update_l(name)
                    best_loss = loss
                loss.backward(retain_graph=True)
                optimizer.step()
                for k, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        node.clip_alpha()
                
            best_loss = np.inf
            for j in range(iter):
                # print(lb)
                best_lb = torch.max(best_lb , lb)
                best_ub = torch.min(best_ub , ub)
                for i in range(len(modules)):
                    # We only need the bounds before a ReLU/HardTanh layer
                    if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                        if isinstance(modules[i - 1], BoundLinear):
                            # add a batch dimension
                            newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                            # Use CROWN to compute pre-activation bounds
                            # starting from layer i-1
                            ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                                    start_node=i - 1, optimize=0, out_features = modules[i - 1].out_features)
                        # Set pre-activation bounds for layer i (the ReLU layer)
                        modules[i].upper_u = ub
                        modules[i].lower_l = lb
                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper,
                                                lower=lower, start_node=i, optimize=True, out_features = modules[i].out_features)
                optimizer.zero_grad(set_to_none=True)
                loss = ub.sum() 
                if(loss < best_loss):
                    for module in modules:
                        if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
                            module.update_u(name)
                    best_loss = loss
                loss.backward(retain_graph=True)
                optimizer.step()
                for k, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        node.clip_alpha()
                
            return best_ub, best_lb
        else:
            print('')
            print('')
            opt = False
            lr_a = 0.1
            lr_b = 0.5
            # lr_b = 0.5
            for module in modules:
                if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
                    if(opt == False):
                        alpha_optimizer = optim.Adam(list(module.alpha_l.values()), lr=lr_a)
                        alpha_optimizer.add_param_group({'params': list(module.alpha_u.values()),'lr': lr_a})
                        alpha_optimizer.add_param_group({'params': list(module.beta_l.values()),'lr': lr_b})
                        # beta_optimizer = optim.Adam(list(module.beta_l.values()), lr=lr_b)

                        alpha_optimizer.add_param_group({'params': list(module.beta_u.values()),'lr': lr_b})
                        # beta_optimizer.add_param_group({'params': list(module.beta_u.values()),'lr': lr_b})
                        opt = True
                    else:
                        alpha_optimizer.add_param_group({'params': list(module.alpha_u.values()),'lr': lr_a})
                        alpha_optimizer.add_param_group({'params': list(module.alpha_l.values()),'lr': lr_a})
                        alpha_optimizer.add_param_group({'params': list(module.beta_u.values()),'lr': lr_b})
                        alpha_optimizer.add_param_group({'params': list(module.beta_l.values()),'lr': lr_b})
                        # beta_optimizer.add_param_group({'params': list(module.beta_u.values()),'lr': lr_b})
                        # beta_optimizer.add_param_group({'params': list(module.beta_l.values()),'lr': lr_b})

            # print("Parameters in alpha_optimizer:")
            # for param_group in alpha_optimizer.param_groups:
            #     for param in param_group['params']:
            #         print(param, param.requires_grad)

            # print("Parameters in beta_optimizer:")
            # for param_group in beta_optimizer.param_groups:
            #     for param in param_group['params']:
            #         print(param.size(), param.requires_grad)
            
            iter = 150
            best_lb, best_ub = lb, ub
            best_loss = np.inf
            for j in range(iter):
                # if((ub-lb).sum() < 0):
                best_lb = torch.max(best_lb , lb)
                best_ub = torch.min(best_ub , ub)
                if(torch.any(best_lb > best_ub)):
                    return best_ub, best_lb
            
                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper,
                                                lower=lower, start_node=i, optimize=True, out_features = modules[i].out_features)

                # beta_optimizer.zero_grad()
                # beta_loss = -lb.sum() 
                # # beta_loss = ub.sum() 
                # beta_loss.backward(retain_graph=True)
                # beta_optimizer.step()

                # for k, node in enumerate(modules):
                #     if isinstance(node, BoundReLU):
                #         node.clip_beta()

                alpha_optimizer.zero_grad()
                alpha_loss = -lb.sum() 
                if(alpha_loss < best_loss):
                    for module in modules:
                        if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
                            module.update_l(name)
                            module.update_u(name)
                    best_loss = alpha_loss
                alpha_loss.backward()
                alpha_optimizer.step()

                # if(alpha_loss > temp_loss+10 or torch.any(lb > ub)):
                # if(torch.any(lb > ub)):
                #     return lb-10, lb
                # if(torch.abs(temp_loss-alpha_loss) < 0.0001):
                #     break
                
                # temp_loss = beta_loss
                print("alpha loss:",alpha_loss)
                for k, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        node.clip_alpha()
                for k, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        node.clip_beta()
                
                # print(beta_loss)
            print('best_lb:',best_lb)
            return best_ub, best_lb

    def domain_filter(self, bounds, P_old, C_list, split_bool_list_global, C_splits, split_bool_list_loop, name_list):
        P = P_old
        split_list = split_bool_list_global
        # std = torch.tensor([[-2.1788,-4.8912,-0.2099,0.3329,-6.5422,-1.6018,-11.8794,9.7545,-2.4076,-1.1615]]).to(device)
        for bound, C, split, name in zip(bounds, C_splits, split_bool_list_loop, name_list):
            lb, ub = bound[1], bound[0]
            if(torch.any(lb > ub)):
                continue
            # elif(torch.any(lb > std)):
            #     print('wrong!!!wrong!!!wrong!!!wrong!!!wrong!!!')
            #     print(lb)
            P.append((lb,ub,C,name))
            split_list.append(split)
        return P, split_list

    def BaB(self, x_U = None, x_L = None, delta = 100, n = 8, batch_size = 1, optimize = 0):
        modules = list(self._modules.values())
        C = []
        for i in range(len(modules)):
            # We only need the bounds before a ReLU/HardTanh layer
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                if isinstance(modules[i - 1], BoundLinear):
                    # add a batch dimension
                    newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                    # Use CROWN to compute pre-activation bounds
                    # starting from layer i-1
                    ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                            start_node=i-1, optimize=0, out_features = modules[i - 1].out_features)
                # Set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
                C.append((lb,ub))
        def get_s(C,device):
            S = []
            # print(C[1][1].shape)
            for C_single in C:                
                len_b = len(C_single[0][0])
                zero_matrix = torch.zeros((len_b, len_b))
                S.append(zero_matrix.to(device))
            return S
        # split_bool_list_global = [zero_copy_structure(C)]
        split_bool_list_global = [get_s(C,device)]
        if(optimize == 0):
            ub, lb = self.optimized_beta_CROWN(C, split_bool_list_global[0], x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=0, name=0)
            return lb, ub
        if(optimize == 1):
            ub, lb = self.optimized_beta_CROWN(C, split_bool_list_global[0], x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=1, name=0)
            return lb, ub
        print('the num of relu:',len(C))
        print('using beta-CROWN')
        # ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
        #                                     C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=True,
        #                                     lower=True, start_node=i, optimize=0, out_features = modules[i].out_features)
        ub, lb = self.optimized_beta_CROWN(C, split_bool_list_global[0], x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=1, name=0)
        # return lb, ub
        # print(C)
        P = [(lb, ub, C, 0)]
        interations = 10
        inter = 0
        print(((ub - lb).sum() > delta))
        print(len(P) < n)
        print(inter <= interations)
        while ((ub - lb).sum() > delta) and len(P) < n and inter <= interations:

            P, C_list, split_bool_list_global, split_bool_list_loop = self.pick_out(P, batch_size, split_bool_list_global)

            assert len(C_list) == len(split_bool_list_loop)
            assert len(P) == len(split_bool_list_global)

            #SINGLE C: (lb,ub)
            C_split, split_bool_list_loop, name_list = self.split(C_list, split_bool_list_loop)
            # print(C_split)
            # print(split_bool_list_loop)
            # print(len(C_list),len(C_split))
            bounds = [self.optimized_beta_CROWN(C_i, split_i, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=2, name=name) for C_i, split_i, name in zip(C_split,split_bool_list_loop,name_list)]
            
            assert len(bounds) == len(C_split)
            assert len(bounds) == len(split_bool_list_loop)

            P, split_bool_list_global = self.domain_filter(bounds, P, C_list, split_bool_list_global, C_split, split_bool_list_loop, name_list)

            lb,d = self.compute_bound(P, lb, "lower")
            ub,d = self.compute_bound(P, ub, "upper")
            if(d == 'END'):
                return lb, ub

            inter+=1
            # print(((ub - lb).sum() > delta))
            # print(len(P) < n)
            # print(inter <= interations)
            
            print(inter,len(P))
            # print(P)
        return lb, ub

    def boundpropogate_from_layer(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None, optimize=False, out_features = 10):
        r"""The bound propagation starting from a given layer. Can be used to compute intermediate bounds or the final bound.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            C (tensor): The initial coefficient matrix. Can be used to represent the output constraints.
            But we don't have any constraints here. So it's just an identity matrix.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            start_node (int): The start node of this propagation. It should be a linear layer.
        Returns:
            ub (tensor): The upper bound of the output of start_node.
            lb (tensor): The lower bound of the output of start_node.
        """
        modules = list(self._modules.values()) if start_node is None else list(self._modules.values())[:start_node + 1]
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0])
        # def omega(modules, k, i, device):
        #     assert(i>k)
        #     if(i == k):
        #         return torch.eye(modules[(i-1)*2].out_features).unsqueeze(0).to(device)
        #     elif:
        #         o = omega(modules, k-1, i, device)

        for i, module in enumerate(reversed(modules)):
            upper_A, upper_b, lower_A, lower_b = module.boundpropogate(upper_A, lower_A, start_node, optimize=optimize, out_features = modules[start_node].out_features)
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b
            # print(module,lower_A,lower_b)

        # sign = +1: upper bound, sign = -1: lower bound
        def _get_concrete_bound(A, sum_b, sign=-1):
            if A is None:
                return None
            A = A.view(A.size(0), A.size(1), -1)
            # A has shape (batch, specification_size, flattened_input_size)
            x_ub = x_U.view(x_U.size(0), -1, 1)
            x_lb = x_L.view(x_L.size(0), -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            bound = bound.squeeze(-1) + sum_b
            return bound

        lb = _get_concrete_bound(lower_A, lower_sum_b, sign=-1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign=+1)
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf])
        return ub, lb




def main():
    # Create the parser
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--activation', default='relu', choices=['relu', 'hardtanh'],
                        type=str, help='Activation Function')
    parser.add_argument('data_file', type=str, help='input data, a tensor saved as a .pth file.')
    parser.add_argument('optimize', type=int, help='whether to use alpha crown, 1 for alpha crown and 0 for crown')
    # Parse the command line arguments
    args = parser.parse_args()

    x_test, label = torch.load(args.data_file)
    if args.activation == 'relu':
        print('use ReLU model')
        model = SimpleNNRelu().to(device)
        model.load_state_dict(torch.load('models/relu_model.pth'))
        # model = two_relu_toy_model(in_dim=2, out_dim=2).to(device)
    else:
        print('use HardTanh model')
        model = SimpleNNHardTanh().to(device)
        model.load_state_dict(torch.load('models/hardtanh_model.pth'))

    batch_size = x_test.size(0)
    x_test = x_test.reshape(batch_size, -1).to(device)
    output = model(x_test)
    y_size = output.size(1)

    # x_test = torch.tensor([[0., 0.]]).float().to(device)
    # batch_size = x_test.size(0)
    # output = model(x_test)
    # y_size = output.size(1)
    print("Network prediction: {}".format(output))
    eps = 0.01
    # eps = 1
    x_u = x_test + eps
    x_l = x_test - eps

    print(f"Verifiying Pertubation - {eps}")
    start_time = time.time()
    boundedmodel = BoundedSequential.convert(model)
    lb, ub = boundedmodel.BaB(x_U=x_u, x_L=x_l, delta=0, n=100, batch_size=100, optimize=args.optimize)
    # print(C)
    # ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True, optimize=args.optimize)
    for i in range(batch_size):
        for j in range(y_size):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))

from contextlib import redirect_stdout
with open('try.txt', 'w') as f:
    with redirect_stdout(f):
        main()