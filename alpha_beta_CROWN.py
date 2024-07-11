import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model import SimpleNNRelu, SimpleNNHardTanh, two_relu_toy_model
from linear import BoundLinear
# from relu import BoundReLU
from relu_alpha_beta import BoundReLU
from hardTanh_question_alpha import BoundHardTanh
import time
import argparse
import torch.optim as optim
import itertools

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

    # def pick_out(self, P, batch_size, split_bool_list):
    #     return P[min(len(P),batch_size):], P[:min(len(P),batch_size)], split_bool_list[min(len(P),batch_size):], split_bool_list[:min(len(P),batch_size)]



    # def general_split(self, C_list, split_lists, split_depth):
    #     C_split = []
    #     split_bool_list = []
    #     name_list = []
    #     relu_list = []
    #     linear_list = []
    #     modules = list(self._modules.values())
    #     for i, module in enumerate(modules):
    #         if isinstance(module, BoundReLU):
    #             relu_list.append(module)
    #             linear_list.append(modules[i-1])
    #     def is_valid_diagonal_matrix(matrix):
    #         n = matrix.shape[0]
    #         diag_mask = torch.eye(n, dtype=bool)
    #         diag_elements = matrix[diag_mask]
    #         off_diag_elements = matrix[~diag_mask]
    #         if not torch.all(off_diag_elements == 0):
    #             return False
    #         if not torch.all((diag_elements == 1) | (diag_elements == -1)):
    #             return False
    #         return True
    #     for P, split_list in zip(C_list, split_lists):
    #         #if all valid, then all splitted
    #         all_valid = all(is_valid_diagonal_matrix(matrix) for matrix in split_list)
    #         if(all_valid):
    #             continue
    #         _, _, C, name = P
    #         num_relu = len(C)
    #         C_1, C_2 = deep_copy_structure(C), deep_copy_structure(C)
    #         split_bool_1, split_bool_2 = deep_copy_structure(split_list), deep_copy_structure(split_list)
    #         rank_list = []
    #         i = 0
    #         #calculate the score of all possible split for each C, here just ub-lb
    #         for single_C, module, linear in zip(C, relu_list, linear_list):
    #             lb, ub = single_C[0].clone().detach(), single_C[1].clone().detach()
    #             # lb, ub = module.lower_l, module.upper_u
                
    #             mask = (lb < 0) & (ub > 0)
    #             score = self.babsr_score(module, linear, name, lb, ub)
    #             score = score.mean(1)
    #             score = torch.where(mask, score, torch.tensor(0, device=device))

    #             # max_index = torch.unravel_index(torch.argmax(score), score.shape)
    #             # max_score = score[max_index]
    #             # #i for index of the relu in the whole model, max_index for the lb index in a certain relu, max_diff is the score
    #             # rank_list.append((max_score,max_index,i))

    #             topk_scores, topk_indices = torch.topk(score.view(-1), split_depth)
    #             max_indices = torch.unravel_index(topk_indices, score.shape)
    #             for idx, max_score in enumerate(topk_scores): 
    #                 max_index = (max_indices[0][idx], max_indices[1][idx])
    #                 rank_list.append((max_score, max_index, i))
    #             i += 1
    #         # if(len(rank_list)==0):
    #         #     continue
    #         sorted_rank = sorted(rank_list, key=lambda x: x[0], reverse=True)
    #         # max_range_term = sorted_rank[0]
    #         max_range_terms = sorted_rank[:split_depth]

    #         #change ub
    #         for max_range_term in max_range_terms:
    #             C_1[max_range_term[2]][1][max_range_term[1]] = 0
    #             #change lb
    #             C_2[max_range_term[2]][0][max_range_term[1]] = 0

    #             split_bool_1[max_range_term[2]][max_range_term[1][1],max_range_term[1][1]] = 1
    #             split_bool_2[max_range_term[2]][max_range_term[1][1],max_range_term[1][1]] = -1

    #             C_split.append(C_1)
    #             C_split.append(C_2)
    #             split_bool_list.append(split_bool_1)
    #             split_bool_list.append(split_bool_2)
    #             name_list.append(name*2+1)
    #             name_list.append(name*2+2)

    #     return C_split, split_bool_list, name_list

    def babsr_score(self, module, linear, name, lb, ub):
        def compute_ratio(lb, ub):
            lower_temp = lb.clamp(max=0)
            upper_temp = F.relu(ub)
            slope_ratio = upper_temp / (upper_temp - lower_temp)
            intercept = -1 * lower_temp * slope_ratio
            return slope_ratio, intercept
        ratio = module.last_lA_list[name]
        ratio_temp_0, ratio_temp_1 = compute_ratio(lb, ub)
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1.unsqueeze(1)
        b_temp = linear.bias

        # print('btemp',b_temp.shape,b_temp)
        # print('ratio',ratio.shape,ratio)
        # print('ratio_temp_0',ratio_temp_0.shape,ratio_temp_0)
        # print('ratio_temp_1',ratio_temp_1.shape,ratio_temp_1)
        b_temp = b_temp * ratio

        ratio_temp_0 = ratio_temp_0.unsqueeze(1)
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = torch.min(bias_candidate_1, bias_candidate_2)  # max for babsr by default
        score_candidate = (bias_candidate + intercept_candidate).abs()
        return score_candidate


    def get_C(self, old_C, labels):

        batch_size, out_features, _ = old_C.shape
        
        new_C = torch.zeros((batch_size, out_features - 1, out_features), device=old_C.device)
        # new_C = torch.zeros((batch_size, out_features, out_features), device=old_C.device)
        
        for b in range(batch_size):
            label = labels[b]
            row_indices = [i for i in range(out_features) if i != label]

            # row_indices = [i for i in range(out_features)]
            
            for i, row in enumerate(row_indices):
                new_C[b, i, row] = -1
            new_C[b, :, label] = 1
            # new_C[b, label, label] = 0
        return new_C

    # def compute_bound(self, P, b, method):
    #     d = 'OK'
    #     if(len(P)==0):
    #         d = 'END'
        

    #     if(method == "lower"):
    #         lower_bound = b.clone().detach()
    #         for i in range(len(lower_bound[0])):
    #             for j in range(len(P)):
    #                 if(P[j][0][0][i] > lower_bound[0][i]):
    #                     lower_bound[0][i] = P[j][0][0][i]
    #         return lower_bound, d
    #     elif(method == "upper"):
    #         upper_bound = b.clone().detach()
    #         for i in range(len(upper_bound[0])):
    #             for j in range(len(P)):
    #                 if(P[j][1][0][i] < upper_bound[0][i]):
    #                     upper_bound[0][i] = P[j][1][0][i]
    #         return upper_bound, d

    def optimized_beta_CROWN(self, C, C_matrix, split, x_U=None, x_L=None, upper=True, lower=True,optimize=1, name=0):
        modules = list(self._modules.values())
        # CROWN propagation for all layers
        ind = 0
        # for module in modules:
        #     if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
        #         print(module.beta_l.values())
        #         print(module.beta_u.values())
        # for k, node in enumerate(modules):
        #     if isinstance(node, BoundReLU):
        #         node.initialize_beta(name)
        #         node.initialize_alpha(name)
        # numpy_list = [tensor.cpu().numpy() for tensor in split]
        # nonzero_counts = [np.count_nonzero(arr) for arr in numpy_list]
        # print(nonzero_counts)

        for i in range(len(modules)):
            # We only need the bounds before a ReLU/HardTanh layer
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                modules[i].upper_u = C[ind][1]
                modules[i].lower_l = C[ind][0]
                modules[i].S = split[ind]
                ind += 1
        # Get the final layer bound
        

        # ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
        #                                       C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper,
        #                                       lower=lower, start_node=i, optimize=0)

        ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                              C=C_matrix, upper=upper,
                                              lower=lower, start_node=i, optimize=0, out_features = C_matrix.shape[1])

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
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
            iter = 100
            best_lb, best_ub = lb, ub
            best_loss = np.inf
            #compute lower bound
            for j in range(iter):
                #update intermediate bound

                for i in range(len(modules)):
                    if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                        if isinstance(modules[i - 1], BoundLinear):
                            newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                            ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                                    start_node=i - 1, optimize=0, out_features = modules[i - 1].out_features)
                        modules[i].upper_u = ub
                        modules[i].lower_l = lb


                # ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                #                                 C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper,
                #                                 lower=lower, start_node=i, optimize=True, out_features = modules[i].out_features)

                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=C_matrix, upper=upper,
                                                lower=lower, start_node=i, optimize=True, out_features = C_matrix.shape[1])
                optimizer.zero_grad()
                loss = -lb.sum()
                print('ret:',lb) 
                if(loss < best_loss):
                    for i in range(len(modules)):
                        if isinstance(modules[i], BoundHardTanh) or isinstance(modules[i], BoundReLU):
                            modules[i].update_l(name)
                            modules[i].update_u(name)
                            if isinstance(modules[i - 1], BoundLinear):
                                newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                                ubt, lbt = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                                        start_node=i - 1, optimize=0, out_features = modules[i - 1].out_features)
                            modules[i].ub = ubt
                            modules[i].lb = lbt

                            
                    best_loss = loss
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
                for k, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        node.clip_alpha()
                best_lb = torch.max(best_lb , lb)
                best_ub = torch.min(best_ub , ub)
                #store the last
                # for i in range(len(modules)):
                #     if isinstance(modules[i], BoundHardTanh) or isinstance(modules[i], BoundReLU):
                #         print('alpha:',modules[i].alpha_l)
                #         print('list:',modules[i].alpha_l_list)
            best_lbb = best_lb.clone()
            #compute lower bound
            return best_ub, best_lb
        else:
            # for k, node in enumerate(modules):
            #     if isinstance(node, BoundReLU):
            #         node.clip_alpha()
            #         node.clip_beta()

            print('C', C)
            print('split:', split)
            # print('name:', name)
            opt = False
            lr_a = 0.01
            lr_b = 0.05

            alpha_params = []
            beta_params = []
            alpha_schedulers = []
            beta_schedulers = []

            for module in modules:
                if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
                    alpha_params.extend(list(module.alpha_l.values()))
                    alpha_params.extend(list(module.alpha_u.values()))
                    beta_params.extend(list(module.beta_l.values()))
                    beta_params.extend(list(module.beta_u.values()))

            alpha_optimizer = optim.Adam([
                {'params': alpha_params, 'lr': lr_a},
                {'params': beta_params, 'lr': lr_b}
            ])
            
            alpha_scheduler = optim.lr_scheduler.ExponentialLR(alpha_optimizer, 0.98)
            iter = 15
            best_lb, best_ub = lb, ub
            best_loss = np.inf
            # print('labels:', self.labels)
            for j in range(iter):
                
                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=C_matrix, upper=upper,
                                                lower=lower, start_node=i, optimize=2, out_features = C_matrix.shape[1])

                # for module in modules:
                #     if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
                #         print('alpha_l')
                #         print(module.alpha_l.values())
                #         print('beta_l')
                #         print(module.beta_l.values())
                alpha_optimizer.zero_grad()
                alpha_loss = -lb.sum()                 
                print('ret:', lb)


                if(alpha_loss < best_loss):
                    for module in modules:
                        if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
                            module.update_l(name)
                            module.update_u(name)
                    best_loss = alpha_loss
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha_scheduler.step()

                
                for k, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        node.clip_alpha()
                        node.clip_beta()
                        # node.inact_alpha(i)
                best_lb = torch.max(best_lb , lb)
                best_ub = torch.min(best_ub , ub)
                if(torch.all(best_lb > 0)):
                    return best_ub, best_lb
            # for module in modules:
            #     if isinstance(module, BoundHardTanh) or isinstance(module, BoundReLU):
            #         print('Stored List')
            #         print('Alpha',module.alpha_l_list)
            #         print('Beta',module.beta_l_list)
            print('best_lb:',best_lb)
            return best_ub, best_lb

    # def domain_filter(self, bounds, P_old, C_list, split_bool_list_global, C_splits, split_bool_list_loop, name_list):
    #     P = P_old
    #     split_list = split_bool_list_global
    #     # std = torch.tensor([[-2.1788,-4.8912,-0.2099,0.3329,-6.5422,-1.6018,-11.8794,9.7545,-2.4076,-1.1615]]).to(device)
    #     for bound, C, split, name in zip(bounds, C_splits, split_bool_list_loop, name_list):
    #         lb, ub = bound[1], bound[0]
    #         if(torch.any(lb > ub)):
    #             continue
    #         P.append((lb,ub,C,name))
    #         split_list.append(split)
    #     return P, split_list

    #sub-function for robustness
    def initialize_para(self, modules, C_domain, C_new, S, name, start_node):
        #initialize upper_d and upper_b in each relu module using the new domain, only called at the first time
        assert len(C_domain)==len(S)
        ind = 0
        for i in range(len(modules)):
            # We only need the bounds before a ReLU/HardTanh layer
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                upper_u = C_domain[ind][1]
                lower_l = C_domain[ind][0]
                

                lb_r = lower_l.clamp(max=0)
                ub_r = upper_u.clamp(min=0)

                # avoid division by 0 when both lb_r and ub_r are 0
                ub_r = torch.max(ub_r, lb_r + 1e-8)

                # CROWN upper and lower linear bounds
                upper_d = ub_r / (ub_r - lb_r)  # slope
                upper_b = - lb_r * upper_d  # intercept
                upper_d = upper_d.unsqueeze(1)
                modules[i].upper_d = upper_d
                modules[i].upper_b = upper_b
                modules[i].S = S[ind]
                modules[i].alpha_l[start_node] = modules[i].alpha_l_list[0][start_node][:,name-1,:].unsqueeze(1)
                modules[i].alpha_u[start_node] = modules[i].alpha_u_list[0][start_node][:,name-1,:].unsqueeze(1)
                modules[i].beta_l[start_node] = modules[i].beta_l_list[0][start_node][:,name-1,:].unsqueeze(1)
                modules[i].beta_u[start_node] = modules[i].beta_u_list[0][start_node][:,name-1,:].unsqueeze(1)
                modules[i].last_lA_list[name] = modules[i].last_lA_list[0][:,name-1,:].unsqueeze(1)
                ind += 1
    

    def mark_topk_indices(self, scores, k):
        #return a mask which has the same size with the lb,ub and split. 1 for split decision point

        batch_size = scores[0].size(0)

        assert all(s.size(0) == batch_size for s in scores)
        
        flattened_scores = [s.view(batch_size, -1) for s in scores]
        
        concatenated_scores = torch.cat(flattened_scores, dim=1)
        
        topk_values, topk_indices = torch.topk(concatenated_scores, k=k, dim=1, largest=True)
        
        masks = [torch.zeros_like(s) for s in scores]
        concatenated_masks = torch.cat([m.view(batch_size, -1) for m in masks], dim=1)

        flattened_indices = topk_indices + (torch.arange(batch_size) * concatenated_scores.size(1)).unsqueeze(1).to(topk_indices.device)
        concatenated_masks.view(-1)[flattened_indices.view(-1)] = 1

        split_sizes = [s.size(1) for s in scores]
        split_masks = torch.split(concatenated_masks, split_sizes, dim=1)
        final_masks = [sm.view_as(s) for sm, s in zip(split_masks, scores)]
        
        return final_masks

    def generate_combinations_with_masks(self, C, masks, split_bool, C_matrix, modules, split_depth):
        batch_size = len(C)
        num_combinations = 2 ** split_depth
        new_C = []
        new_split_bool = []
        combinations = torch.tensor(list(itertools.product([0, 1], repeat=split_depth)), device=C[0][0].device)
        C_matrix_new = C_matrix.repeat(1, num_combinations, 1)
        final_start_node = len(modules) - 1
        ind = 0
        for i in range(len(modules)):
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                modules[i].alpha_l[final_start_node] = modules[i].alpha_l[final_start_node].repeat(1, num_combinations, 1)
                modules[i].alpha_u[final_start_node] = modules[i].alpha_u[final_start_node].repeat(1, num_combinations, 1)
                modules[i].beta_l[final_start_node] = modules[i].beta_l[final_start_node].repeat(1, num_combinations, 1)
                modules[i].beta_u[final_start_node] = modules[i].beta_u[final_start_node].repeat(1, num_combinations, 1)

                modules[i].alpha_l[final_start_node] = nn.Parameter(modules[i].alpha_l[final_start_node])
                modules[i].alpha_l[final_start_node].requires_grad_()
                modules[i].alpha_u[final_start_node] = nn.Parameter(modules[i].alpha_u[final_start_node])
                modules[i].alpha_u[final_start_node].requires_grad_()
                modules[i].beta_l[final_start_node] = nn.Parameter(modules[i].beta_l[final_start_node])
                modules[i].beta_l[final_start_node].requires_grad_()
                modules[i].beta_u[final_start_node] = nn.Parameter(modules[i].beta_u[final_start_node])
                modules[i].beta_u[final_start_node].requires_grad_()
                ind += 1

        # Concatenate all lb, ub, masks, and split_bool into one large tensor
        all_lb = torch.cat([c[0] for c in C], dim=1)
        all_ub = torch.cat([c[1] for c in C], dim=1)
        all_masks = torch.cat([m for m in masks], dim=1)
        all_split = torch.cat([s for s in split_bool], dim=1)
        
        num_elements = all_lb.size(1)
        batch_size = all_lb.size(0)

        # Repeat to match the number of combinations
        repeated_lb = all_lb.repeat(num_combinations, 1)
        repeated_ub = all_ub.repeat(num_combinations, 1)
        repeated_split = all_split.repeat(num_combinations, 1)
        mask_indices = all_masks.nonzero(as_tuple=True)[1].view(batch_size,-1)
        
        for i, combination in enumerate(combinations):
            combo_values = combination.float()
            for mask_indice in mask_indices:
                for idx, combo in zip(mask_indice, combo_values):
                    if combo == 0:
                        repeated_lb[i*batch_size:(i+1)*batch_size:, idx] = 0
                        repeated_split[i*batch_size:(i+1)*batch_size:, idx] = -1
                    else:
                        repeated_ub[i*batch_size:(i+1)*batch_size:, idx] = 0
                        repeated_split[i*batch_size:(i+1)*batch_size:, idx] = 1
        
        #reshape to the original
        split_sizes = [s.size(1) for s in split_bool]
        split_split = torch.split(repeated_split, split_sizes, dim=1)
        new_split_bool = [ss.view(-1,split_size) for ss, split_size in zip(split_split,split_sizes)]

        new_C = []

        lb_split = torch.split(repeated_lb, split_sizes, dim=1)
        lb_final = [ss.view(-1,split_size) for ss, split_size in zip(lb_split,split_sizes)]

        ub_split = torch.split(repeated_ub, split_sizes, dim=1)
        ub_final = [ss.view(-1,split_size) for ss, split_size in zip(ub_split,split_sizes)]

        for lb, ub in zip(lb_final, ub_final):
            new_C.append((lb,ub))

        return new_C, new_split_bool, C_matrix_new
        

    def general_split_robustness(self, C, split_bool, C_matrix, modules, name, split_depth = 1):
        relu_list = []
        linear_list = []
        for i, module in enumerate(modules):
            if isinstance(module, BoundReLU):
                relu_list.append(module)
                linear_list.append(modules[i-1])


        rank_list = []
        scores = []
        i = 0
        #calculate the score of all possible split for each C, here just ub-lb
        for single_C, module, linear in zip(C, relu_list, linear_list):
            lb, ub = single_C[0].clone().detach(), single_C[1].clone().detach()
            
            mask = (lb < 0) & (ub > 0)
            score = self.babsr_score(module, linear, name, lb, ub)
            score = score.mean(1)
            # score[0][22]+=100
            score = torch.where(mask, score, torch.tensor(0, device=device))
            # print('using babsr:',score.shape,score)
            scores.append(score)
            i += 1
        masks = self.mark_topk_indices(scores, split_depth)
        new_C, new_split_bool, C_matrix_new = self.generate_combinations_with_masks(C, masks, split_bool, C_matrix, modules, split_depth)


        return new_C, new_split_bool, C_matrix_new

    def domain_filter_robusness(self, lb, C, split_bool_sub, C_new, modules, name):
        mask = lb < 0
        #all lbs are smaller than 0, no domain change
        if(torch.all(mask)):
            return C, split_bool_sub, C_new, 'remain'
        #all lbs are larger than 0, then verified
        if(torch.all(~mask)):
            return C, split_bool_sub, C_new, 'verified'

        C_filtered = []
        split_filtered = []
        for C_batch, split_batch in zip(C, split_bool_sub):
            C_filtered.append((C_batch[0][mask.squeeze(0)],C_batch[1][mask.squeeze(0)]))
            split_filtered.append(split_batch[mask.squeeze(0), :])

        C_new = C_new[:, mask.squeeze(0), :]
        final_start_node = len(modules) - 1
        for i in range(len(modules)):
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                modules[i].alpha_l[final_start_node] = modules[i].alpha_l[final_start_node][:, mask.squeeze(0), :]
                modules[i].alpha_u[final_start_node] = modules[i].alpha_u[final_start_node][:, mask.squeeze(0), :]
                modules[i].beta_l[final_start_node] = modules[i].beta_l[final_start_node][:, mask.squeeze(0), :]
                modules[i].beta_u[final_start_node] = modules[i].beta_u[final_start_node][:, mask.squeeze(0), :]

                modules[i].alpha_l[final_start_node] = nn.Parameter(modules[i].alpha_l[final_start_node])
                modules[i].alpha_l[final_start_node].requires_grad_()
                modules[i].alpha_u[final_start_node] = nn.Parameter(modules[i].alpha_u[final_start_node])
                modules[i].alpha_u[final_start_node].requires_grad_()
                modules[i].beta_l[final_start_node] = nn.Parameter(modules[i].beta_l[final_start_node])
                modules[i].beta_l[final_start_node].requires_grad_()
                modules[i].beta_u[final_start_node] = nn.Parameter(modules[i].beta_u[final_start_node])
                modules[i].beta_u[final_start_node].requires_grad_()

                modules[i].last_lA_list[name] = modules[i].last_lA_list[name][:, mask.squeeze(0), :]


        return C_filtered, split_filtered, C_new, 'changed'
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
        C_matrix = self.get_C(torch.eye(modules[i].out_features).unsqueeze(0).to(x_U),self.labels)
        
        def get_s(C,device):
            S = []
            # print(C[1][1].shape)
            for C_single in C:                
                zero_matrix = torch.zeros(C_single[0].shape)
                S.append(zero_matrix.to(device))
            return S
        # split_bool_list_global = [zero_copy_structure(C)]
        split_bool = get_s(C,device)
        if(optimize == 0):
            ub, lb = self.optimized_beta_CROWN(C, C_matrix, split_bool, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=0, name=0)
            return lb, ub
        if(optimize == 1):
            ub, lb = self.optimized_beta_CROWN(C, C_matrix, split_bool, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=1, name=0)
            return lb, ub
        print('using beta-CROWN')
        # ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
        #                                     C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=True,
        #                                     lower=True, start_node=i, optimize=0, out_features = modules[i].out_features)

        ub, lb = self.optimized_beta_CROWN(C, C_matrix, split_bool, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=0, name=0)
        print('result from crown is:',lb)
        if(torch.all(lb>0)):
            print('verified using crown')
            return lb, ub

        ub, alpha_lb = self.optimized_beta_CROWN(C, C_matrix, split_bool, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=1, name=0)
        print('result from alpha crown is:',alpha_lb)
        if(torch.all(lb>0)):
            print('verified using alpha crown')
            return alpha_lb, ub

        C = []
        unstable_size = 0
        for i in range(len(modules)):
            # We only need the bounds before a ReLU/HardTanh layer
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                C.append((modules[i].lb, modules[i].ub))
                mask = (modules[i].lb < 0) & (modules[i].ub > 0)
                unstable_size += mask.sum().item()
        print('unstable:',unstable_size)
        num_out_features = C_matrix.shape[1]
        final_start_node = len(modules) - 1
        for i in range(num_out_features):
            unstable_remain = unstable_size
            C_new = C_matrix[:,i,:].unsqueeze(1)
            C_sub = deep_copy_structure(C)
            split_bool_sub = deep_copy_structure(split_bool)
            ret = alpha_lb[:,i]
            name = i+1
            if(torch.all(ret > 0)):
                print('instance', i, 'verified with lb', ret, 'and C', C_new)
                print('')
            else:
                print('')
                print('instance', i, 'have not been verified with lb', ret, 'and C', C_new)

                self.initialize_para(modules,C_sub,C_new,split_bool_sub,name,final_start_node)

                while(torch.any(ret<0) and C_new.shape[1] <= n):
                    # assert C[0][0] == len(split_bool_list_global)
                    # assert len(C_list) == C_new.shape[1]
                    split_depth = 7
                    real_ss = min(split_depth,unstable_remain)

                    C_sub, split_bool_sub, C_new = self.general_split_robustness(C_sub, split_bool_sub, C_new, modules, name, split_depth = real_ss)
                    unstable_remain -= real_ss

                    _, ret = self.optimized_beta_CROWN(C_sub, C_new, split_bool_sub, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=2, name=name)

                    C_sub, split_bool_sub, C_new, verified = self.domain_filter_robusness(ret, C_sub, split_bool_sub, C_new, modules, name)

                    if(verified == 'verified'):
                        #one batch verified
                        print('verified!!!!')
                        break
                    elif(unstable_remain == 0):
                        #one batch unsafe, so just end the process
                        print('unsafe!!!!')
                        return alpha_lb, ub, 'unsafe'
                if(verified != 'verified'):
                    #one batch is not verified in fixed size of domains
                    print('Unknown!!!! Out of size')
                    return alpha_lb, ub, 'unknown'
        return alpha_lb, ub, 'safe'







        # P = [(alpha_lb, ub, C, 0)]
        # interations = 0
        # inter = 0

        # while (torch.any(lb < 0)) and len(P) < n and inter <= interations:

        #     P, C_list, split_bool_list_global, split_bool_list_loop = self.pick_out(P, batch_size, split_bool_list_global)

        #     assert len(C_list) == len(split_bool_list_loop)
        #     assert len(P) == len(split_bool_list_global)

        #     #SINGLE C: (lb,ub)
        #     C_split, split_bool_list_loop, name_list = self.general_split(C_list, split_bool_list_loop, split_depth = 1)
        #     # print(C_split)
        #     # print(split_bool_list_loop)
        #     # print(len(C_list),len(C_split))
        #     bounds = [self.optimized_beta_CROWN(C_i, C_matrix, split_i, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=2, name=name) for C_i, split_i, name in zip(C_split,split_bool_list_loop,name_list)]
            
        #     assert len(bounds) == len(C_split)
        #     assert len(bounds) == len(split_bool_list_loop)

        #     P, split_bool_list_global = self.domain_filter(bounds, P, C_list, split_bool_list_global, C_split, split_bool_list_loop, name_list)

        #     lb,d = self.compute_bound(P, lb, "lower")
        #     ub,d = self.compute_bound(P, ub, "upper")
        #     if(d == 'END'):
        #         return lb, ub

        #     inter+=1
        #     # print(((ub - lb).sum() > delta))
        #     # print(len(P) < n)
        #     # print(inter <= interations)
            
        #     print(inter,len(P))
        #     # print(P)
        # return lb, ub

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
            upper_A, upper_b, lower_A, lower_b = module.boundpropogate(upper_A, lower_A, start_node, optimize=optimize, out_features = out_features)
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
    parser.add_argument('data', type=str, help='toy tor box, complex for complex')
    # Parse the command line arguments
    args = parser.parse_args()

    x_test, labels = torch.load(args.data_file)
    if args.activation == 'relu':
        print('use ReLU model')
        if(args.data == 'complex'):
            model = SimpleNNRelu().to(device)
            model.load_state_dict(torch.load('models/relu_model.pth'))
        else:
            model = two_relu_toy_model(in_dim=2, out_dim=2).to(device)
    else:
        print('use HardTanh model')
        model = SimpleNNHardTanh().to(device)
        model.load_state_dict(torch.load('models/hardtanh_model.pth'))

    if(args.data == 'complex'):
        batch_size = x_test.size(0)
        x_test = x_test.reshape(batch_size, -1).to(device)
        labels = torch.tensor([labels-1]).long().to(device)
        output = model(x_test)
        y_size = output.size(1) - 1
    else:
        x_test = torch.tensor([[0., 0.]]).float().to(device)
        labels = torch.tensor([0]).long().to(device)
        batch_size = x_test.size(0)
        output = model(x_test)
        y_size = output.size(1) - 1

    print("Network prediction: {}".format(output))
    if(args.data == 'complex'):
        eps = 0.023
        # eps = 0.025
    else:
        eps = 1
    x_u = x_test + eps
    x_l = x_test - eps

    print(f"Verifiying Pertubation - {eps}")
    start_time = time.time()
    boundedmodel = BoundedSequential.convert(model)
    boundedmodel.labels = labels
    lb, ub, verify = boundedmodel.BaB(x_U=x_u, x_L=x_l, delta=-1, n=1000, batch_size=100, optimize=args.optimize)

    if(verify == 'safe'):
        print('')
        print('all batches are verified!')
    # # print(C)
    # # ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True, optimize=args.optimize)
    # for i in range(batch_size):
    #     for j in range(y_size):
    #         print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
    #             j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))

from contextlib import redirect_stdout
with open('try.txt', 'w') as f:
    with redirect_stdout(f):
        main()