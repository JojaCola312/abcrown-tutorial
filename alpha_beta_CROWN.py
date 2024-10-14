import torch
import torch.nn as nn
import numpy as np
from model import SimpleNNRelu, two_relu_toy_model
from linear import BoundLinear
from relu_alpha_beta import BoundReLU
from split import general_split_robustness
import time
import argparse
import torch.optim as optim


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class final:
    def __init__(self):
        self.lower_A = None
        self.lower_sum_b = None
        self.upper_A = None
        self.upper_sum_b = None
        self.lower_A_actual = None
        self.lower_sum_b_actual = None
        self.upper_A_actual = None
        self.upper_sum_b_actual = None

final_para = final()
def deep_copy_structure(structure):
    r"""As the sturcture of domains contains tuple and list, using this function to copy them."""
    if isinstance(structure, torch.Tensor):
        return structure.clone().detach()
    elif isinstance(structure, list):
        return [deep_copy_structure(item) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(deep_copy_structure(item) for item in structure)
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

    


    def get_C(self, old_C, labels):
        r"""Get the initial coefficient matrix for robustness verification

        Args:
            old_C (tensor): The initial coefficient matrix, which should be an identical matrix.
                            Shape should be (1, out_features, out_features)

            labels (list): The list of true labels

        Return:
            new_C (tensor): The initial coefficient matrix for robustness verification.
                            Shape should be (1, out_features - 1, out_features)

        """
        batch_size, out_features, _ = old_C.shape      
        new_C = torch.zeros((batch_size, out_features - 1, out_features), device=old_C.device)      
        for b in range(batch_size):
            label = labels[b]
            row_indices = [i for i in range(out_features) if i != label]    
            for i, row in enumerate(row_indices):
                new_C[b, i, row] = -1
            new_C[b, :, label] = 1
        return new_C


    def optimized_beta_CROWN(self, domains, C_matrix, split, x_U=None, x_L=None, upper=True, lower=True, optimize=1, name=0):
        """Main function to get the optimized bound

        Args:
            domains (list): Bounds for different pre-activation layers.
            C_matrix (tensor): The initial coefficient matrix
            split (list): [split_1, split_2, ...], each split has the same shape with lower bound and the length is the same with domains,
                            indicates the split logic, 0 for no split, -1 for active, 1 for inactive
            x_U (tensor): The upper bound of x.
            x_L (tensor): The lower bound of x.
            upper (bool): Whether we want upper bound.
            lower (bool): Whether we want lower bound.
            optimize (int): 0 for CROWN, 1 for alpha-CROWN, 2 for alpha-beta-CROWN.
            name (int): id of the instance, 0 for CROWN and alpha-CROWN.
        
        Return:
            best_ub (tensor): The final output upper bound.
            best_lb (tensor): The final output lower bound.
        """
        modules = list(self._modules.values())
        ind = 0
        for i in range(len(modules)):
            # initialize the pre-activation bounds
            if isinstance(modules[i], BoundReLU):
                modules[i].upper_u = domains[ind][1]
                modules[i].lower_l = domains[ind][0]
                modules[i].S = split[ind]
                modules[i].final_start_node = len(modules) - 1
                ind += 1
        # Get the final layer bound
        ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                              C=C_matrix, upper=upper,
                                              lower=lower, start_node=i, optimize=optimize, out_features = C_matrix.shape[1], save = 3)

        if(optimize==0):
            print('using CROWN')
            return ub, lb
        
        elif(optimize==4):
            ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                              C=C_matrix, upper=upper,
                                              lower=lower, start_node=i, optimize=4, out_features = C_matrix.shape[1], save = 3)
            return ub, lb

        elif(optimize==3):
            lr_a = 0.01
            lr_b = 0.05

            alpha_params = []
            beta_params = []

            for module in modules:
                if isinstance(module, BoundReLU):
                    alpha_params.extend(list(module.alpha_l.values()))
                    alpha_params.extend(list(module.alpha_u.values()))
                    beta_params.extend(list(module.beta_l.values()))
                    beta_params.extend(list(module.beta_u.values()))

            optimizer = optim.Adam([
                {'params': alpha_params, 'lr': lr_a},
                {'params': beta_params, 'lr': lr_b}
            ])
            
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
            iter = 15
            best_lb, best_ub = lb, ub

            for j in range(iter):
                
                for i in range(len(modules)):
                    if isinstance(modules[i], BoundReLU):
                        if isinstance(modules[i - 1], BoundLinear):
                            newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                            ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                                    start_node=i - 1, optimize=2, out_features = modules[i - 1].out_features, save = 4)
                        modules[i].upper_u = ub
                        modules[i].lower_l = lb
                ind = 0
                for i in range(len(modules)):
                    # initialize the pre-activation bounds
                    if isinstance(modules[i], BoundReLU):
                        s = split[ind]
                        pos = (s == 1)
                        lower_mask = (s == 1)
                        upper_mask = (s == -1)
                        upper_u_clone = modules[i].upper_u.clone()
                        lower_l_clone = modules[i].lower_l.clone()

                        upper_u_clone[lower_mask] = 0
                        lower_l_clone[upper_mask] = 0

                        modules[i].upper_u = upper_u_clone
                        modules[i].lower_l = lower_l_clone
                        # modules[i].upper_u = domains[ind][1]
                        # modules[i].lower_l = domains[ind][0]
                        # modules[i].S = split[ind]
                        # print('have a look:', s, domains[ind][1], domains[ind][0], modules[i].upper_u, modules[i].lower_l)

                        ind += 1

                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=C_matrix, upper=upper,
                                                lower=lower, start_node=i, optimize=2, out_features = C_matrix.shape[1], save = 3)

                optimizer.zero_grad()
                loss = -lb.sum()                 

                if(torch.any(lb > best_lb)):
                    for module in modules:
                        if isinstance(module, BoundReLU):
                            module.update(name, 2, lb>best_lb)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # for module in modules:
                #     if isinstance(module, BoundReLU):
                #         print('module.beta')
                #         for key, value in module.beta_l.items():
                #             print(f"Key: {key}, Value: {value}, S:{module.S}")

                for k, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        node.clip_alpha()
                        node.clip_beta()
                best_lb = torch.max(best_lb , lb)
                best_ub = torch.min(best_ub , ub)
                
                if(torch.all(best_lb > 0)):
                    return best_ub, best_lb

            return best_ub, best_lb

        
        elif(optimize==1):
            print('using alpha-CROWN')
            alpha_params = []
            lr = 0.1
            for module in modules:
                if isinstance(module, BoundReLU):
                    alpha_params.extend(list(module.alpha_l.values()))
                    alpha_params.extend(list(module.alpha_u.values()))
            optimizer = optim.Adam([
                {'params': alpha_params, 'lr': lr}
            ])
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

            iter = 55
            best_lb, best_ub = lb, ub
            #compute lower bound
            for j in range(iter):
                #update intermediate bound

                for i in range(len(modules)):
                    if isinstance(modules[i], BoundReLU):
                        if isinstance(modules[i - 1], BoundLinear):
                            newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                            ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                                    start_node=i - 1, optimize=1, out_features = modules[i - 1].out_features, save = 0)
                        
                        modules[i].upper_u = ub
                        modules[i].lower_l = lb
                        

                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=C_matrix, upper=upper,
                                                lower=lower, start_node=i, optimize=1, out_features = C_matrix.shape[1], save = 2)
                optimizer.zero_grad()
                loss = -lb.sum()
                # update the pre-activate bound to the best, and update relevant parameter
                if(torch.any(lb >= best_lb)):
                    for i in range(len(modules)):
                        if isinstance(modules[i], BoundReLU):
                            modules[i].update(name, 1, lb>best_lb)
                            if isinstance(modules[i - 1], BoundLinear):
                                newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                                ubt, lbt = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                                        start_node=i - 1, optimize=0, out_features = modules[i - 1].out_features, save = 0)
                                if(modules[i].lb != None):
                                    modules[i].ub = torch.min(ubt,modules[i].ub)
                                    modules[i].lb = torch.max(lbt,modules[i].lb)
                                else:
                                    modules[i].ub = ubt
                                    modules[i].lb = lbt
                            
                    
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
                for k, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        node.clip_alpha()
                best_lb = torch.max(best_lb , lb)
                best_ub = torch.min(best_ub , ub)
            return best_ub, best_lb
        else:
            lr_a = 0.01
            lr_b = 0.05

            alpha_params = []
            beta_params = []

            for module in modules:
                if isinstance(module, BoundReLU):
                    alpha_params.extend(list(module.alpha_l.values()))
                    alpha_params.extend(list(module.alpha_u.values()))
                    beta_params.extend(list(module.beta_l.values()))
                    beta_params.extend(list(module.beta_u.values()))

            optimizer = optim.Adam([
                {'params': alpha_params, 'lr': lr_a},
                {'params': beta_params, 'lr': lr_b}
            ])
            
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
            # iter = 15
            iter = 15
            best_lb, best_ub = lb, ub

            for j in range(iter):
                
                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=C_matrix, upper=upper,
                                                lower=lower, start_node=i, optimize=2, out_features = C_matrix.shape[1], save = 3)

                optimizer.zero_grad()
                loss = -lb.sum()                 

                if(torch.any(lb > best_lb)):
                    for module in modules:
                        if isinstance(module, BoundReLU):
                            module.update(name, 2, lb>best_lb)
                loss.backward()
                optimizer.step()
                scheduler.step()

                
                for k, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        node.clip_alpha()
                        node.clip_beta()
                best_lb = torch.max(best_lb , lb)
                best_ub = torch.min(best_ub , ub)
                if(torch.all(best_lb > 0)):
                    return best_ub, best_lb

            return best_ub, best_lb


    def initialize_para(self, modules, name, start_node):
        """initialize the parameter, only be called at the begining of each instance"""
        if start_node == 'all':
            start_node = len(modules) - 1
            for i in range(len(modules)):
                if isinstance(modules[i], BoundReLU):
                    for key in modules[i].alpha_l.keys():
                        modules[i].alpha_l[key] = modules[i].alpha_l_list[0][key].clone().detach()
                        modules[i].alpha_u[key] = modules[i].alpha_u_list[0][key].clone().detach()
                        modules[i].beta_l[key] = modules[i].beta_l_list[0][key].clone().detach()
                        modules[i].beta_u[key] = modules[i].beta_u_list[0][key].clone().detach()

                        modules[i].alpha_l[key] = nn.Parameter(modules[i].alpha_l[key])
                        modules[i].alpha_l[key].requires_grad_()

                        modules[i].alpha_u[key] = nn.Parameter(modules[i].alpha_u[key])
                        modules[i].alpha_u[key].requires_grad_()

                        modules[i].beta_l[key] = nn.Parameter(modules[i].beta_l[key])
                        modules[i].beta_l[key].requires_grad_()

                        modules[i].beta_u[key] = nn.Parameter(modules[i].beta_u[key])
                        modules[i].beta_u[key].requires_grad_()
                    modules[i].alpha_l[start_node] = modules[i].alpha_l_list[0][start_node][:,name-1,:].unsqueeze(1).clone().detach()
                    modules[i].alpha_u[start_node] = modules[i].alpha_u_list[0][start_node][:,name-1,:].unsqueeze(1).clone().detach()
                    modules[i].beta_l[start_node] = modules[i].beta_l_list[0][start_node][:,name-1,:].unsqueeze(1).clone().detach()
                    modules[i].beta_u[start_node] = modules[i].beta_u_list[0][start_node][:,name-1,:].unsqueeze(1).clone().detach()

                    modules[i].alpha_l[start_node] = nn.Parameter(modules[i].alpha_l[start_node])
                    modules[i].alpha_l[start_node].requires_grad_()
                    modules[i].alpha_u[start_node] = nn.Parameter(modules[i].alpha_u[start_node])
                    modules[i].alpha_u[start_node].requires_grad_()
                    modules[i].beta_l[start_node] = nn.Parameter(modules[i].beta_l[start_node])
                    modules[i].beta_l[start_node].requires_grad_()
                    modules[i].beta_u[start_node] = nn.Parameter(modules[i].beta_u[start_node])
                    modules[i].beta_u[start_node].requires_grad_()
                    modules[i].last_lA_list[name] = modules[i].last_lA_list[0][:,name-1,:].unsqueeze(1).clone().detach()

        else:
            for i in range(len(modules)):
                if isinstance(modules[i], BoundReLU):
                    
                    modules[i].alpha_l[start_node] = modules[i].alpha_l_list[0][start_node][:,name-1,:].unsqueeze(1).clone().detach()
                    modules[i].alpha_u[start_node] = modules[i].alpha_u_list[0][start_node][:,name-1,:].unsqueeze(1).clone().detach()
                    modules[i].beta_l[start_node] = modules[i].beta_l_list[0][start_node][:,name-1,:].unsqueeze(1).clone().detach()
                    modules[i].beta_u[start_node] = modules[i].beta_u_list[0][start_node][:,name-1,:].unsqueeze(1).clone().detach()
                    modules[i].last_lA_list[name] = modules[i].last_lA_list[0][:,name-1,:].unsqueeze(1).clone().detach()

                    modules[i].alpha_l[start_node] = nn.Parameter(modules[i].alpha_l[start_node])
                    modules[i].alpha_l[start_node].requires_grad_()
                    modules[i].alpha_u[start_node] = nn.Parameter(modules[i].alpha_u[start_node])
                    modules[i].alpha_u[start_node].requires_grad_()
                    modules[i].beta_l[start_node] = nn.Parameter(modules[i].beta_l[start_node])
                    modules[i].beta_l[start_node].requires_grad_()
                    modules[i].beta_u[start_node] = nn.Parameter(modules[i].beta_u[start_node])
                    modules[i].beta_u[start_node].requires_grad_()
    
    

    def domain_filter_robusness(self, lb, domains, split_bool, C, modules, name):
        """Domain filter after each iteration

        Args:
            lb (tensor): The lower bound of output
            domains (list): Bounds for different pre-activation layers.
            split_bool (list): [split_1, split_2, ...], each split has the same shape with lower bound and the length is the same with domains,
                            indicates the split logic, 0 for no split, -1 for active, 1 for inactive
            C (tensor): The initial coefficient matrix
            modules (list): [module1, module2, ...], the list of all ReLU layers.
            name (int): id of the instance
        
        Return:
            domains (list): Bounds for different pre-activation layers after filter, the domains related to the verified instance are removed.
            split_bool (list): The split logic after filter, the split logic related to the verified instance is removed.
            verify_status (string): 'remain' for unchanges, 'verified' for verified, 'changed' for changing. Only 'verified' is meaningful.
            lb (tensor): The remaining lower bound of inverified instance.
        """
        mask = lb < 0
        #all lbs are smaller than 0, no domain change
        if(torch.all(mask)):
            return domains, split_bool, C, 'remain', lb[mask]
        #all lbs are larger than 0, then verified
        if(torch.all(~mask)):
            return domains, split_bool, C, 'verified', lb[mask]
        
        final_para.lower_A_actual = final_para.lower_A_ing[:, mask.squeeze(0), :]
        final_para.lower_sum_b_actual = final_para.lower_sum_b_ing[:, mask.squeeze(0)]
        final_para.upper_A_actual = final_para.upper_A_ing[:, mask.squeeze(0), :]
        final_para.upper_sum_b_actual = final_para.upper_sum_b_ing[:, mask.squeeze(0)]

        domains_filtered = []
        split_filtered = []
        for domains_batch, split_batch in zip(domains, split_bool):
            domains_filtered.append((domains_batch[0][mask.squeeze(0)],domains_batch[1][mask.squeeze(0)]))
            split_filtered.append(split_batch[mask.squeeze(0), :])

        C = C[:, mask.squeeze(0), :]
        final_start_node = len(modules) - 1
        for i in range(len(modules)):
            if isinstance(modules[i], BoundReLU):
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



        return domains_filtered, split_filtered, C, 'changed', lb[mask]

    def get_split_depth(self, batch_size, min_batch_size):
        # Here we check the length of current domain list.
        # If the domain list is small, we can split more layers.
        if batch_size < min_batch_size:
            # Split multiple levels, to obtain at least min_batch_size domains in this batch.
            return max(1, int(
                np.log(min_batch_size / max(1, batch_size)) / np.log(2)))
        else:
            return 1

    def BaB(self, x_U = None, x_L = None, n = 2048):
        """Main function, first try CROWN and alpha-CROWN, then alpha-beta-CROWN
        
        Args:
            x_U (tensor): The upper bound of x.
            x_L (tensor): The lower bound of x.
            n (int): The threshold of maximum domain size of 'unknown'
        
        Return:
            lb (tensor): The final output lower bound. Not necessary if finally using alpha-beta-CROWN.
            ub (tensor): The final output upper bound. Not necessary if finally using alpha-beta-CROWN.
            verified_status (string): 'safe' if all instances are verified. 'unknown' if it is not verified within 
                                      pre-determined domain size or all node split.
        """
        modules = list(self._modules.values())
        domains = []
        for i in range(len(modules)):
            # We only need the bounds before a ReLU layer
            if isinstance(modules[i], BoundReLU):
                if isinstance(modules[i - 1], BoundLinear):
                    # add a batch dimension
                    print('CROWN pre-activation')
                    newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                    # Use CROWN to compute pre-activation bounds
                    # starting from layer i-1
                    ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                            start_node=i-1, optimize=0, out_features = modules[i - 1].out_features, save = 1)
                # Set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
                domains.append((lb,ub))
        C_matrix = self.get_C(torch.eye(modules[i].out_features).unsqueeze(0).to(x_U),self.labels)
        
        #get the initial split logic, which should be all 0
        def get_s(domains,device):
            S = []
            for domains_single in domains:            
                zero_matrix = torch.zeros(domains_single[0].shape)
                S.append(zero_matrix.to(device))
            return S
        split_bool = get_s(domains,device)

        #try CROWN
        ub, lb = self.optimized_beta_CROWN(domains, C_matrix, split_bool, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=0, name=0)
        print('result from CROWN is:',lb)
        if(torch.all(lb>0)):
            print('verified using CROWN')
            return lb, ub, 'safe'

        #try alpha CROWN
        ub, alpha_lb = self.optimized_beta_CROWN(domains, C_matrix, split_bool, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=1, name=0)
        print('result from alpha CROWN is:',alpha_lb)
        if(torch.all(lb>0)):
            print('verified using alpha CROWN')
            return alpha_lb, ub, 'safe'

        domains = []
        unstable_size = 0
        #get the initial domain and unstable size
        for i in range(len(modules)):
            if isinstance(modules[i], BoundReLU):
                domains.append((modules[i].lb, modules[i].ub))
                mask = (modules[i].lb < 0) & (modules[i].ub > 0)
                unstable_size += mask.sum().item()

        print('using alpha-beta-CROWN')
        print('size of unstable nuerons:',unstable_size)
        num_out_features = C_matrix.shape[1]
        final_start_node = len(modules) - 1

        for i in range(num_out_features):
            unstable_remain = unstable_size
            C_new = C_matrix[:,i,:].unsqueeze(1)
            domains_sub = deep_copy_structure(domains)
            split_bool_sub = deep_copy_structure(split_bool)
            ret = alpha_lb[:,i]
            final_para.lower_A_actual = final_para.lower_A[:,i,:].unsqueeze(1)
            final_para.upper_A_actual = final_para.upper_A[:,i,:].unsqueeze(1)
            final_para.lower_sum_b_actual = final_para.lower_sum_b[:,i].unsqueeze(1)
            final_para.upper_sum_b_actual = final_para.upper_sum_b[:,i].unsqueeze(1)
            name = i+1
            if(i != 1):
                continue
            if(torch.all(ret > 0)):
                print('')
                print('instance', i, 'is verified with lb', ret, 'and C', C_new)
            else:
                print('')
                print('instance', i, 'is not verified with lb', ret, 'and C', C_new)


                def try_all_split_t():
                    split_sizes = [s.size(1) for s in split_bool_sub]
                    all_lb = torch.cat([c[0] for c in domains_sub], dim=1)
                    all_ub = torch.cat([c[1] for c in domains_sub], dim=1)
                    all_split = torch.cat([s for s in split_bool_sub], dim=1)
                    all_lb = all_lb.repeat(all_lb.shape[1]*2, 1)
                    all_ub = all_ub.repeat(all_ub.shape[1]*2, 1)
                    all_split = all_split.repeat(all_split.shape[1]*2, 1)
                    C_matrix_new = C_new.repeat(1, all_split.shape[1]*2, 1)
                    final_start_node = len(modules) - 1
                    self.initialize_para(modules, name, final_start_node)
                    for i in range(len(modules)):
                        if isinstance(modules[i], BoundReLU):
                            modules[i].alpha_l[final_start_node] = modules[i].alpha_l[final_start_node].repeat(1, all_split.shape[1]*2, 1)
                            modules[i].alpha_u[final_start_node] = modules[i].alpha_u[final_start_node].repeat(1, all_split.shape[1]*2, 1)
                            modules[i].beta_l[final_start_node] = modules[i].beta_l[final_start_node].repeat(1, all_split.shape[1]*2, 1)
                            modules[i].beta_u[final_start_node] = modules[i].beta_u[final_start_node].repeat(1, all_split.shape[1]*2, 1)
                            
                            modules[i].alpha_l[final_start_node] = nn.Parameter(modules[i].alpha_l[final_start_node])
                            modules[i].alpha_l[final_start_node].requires_grad_()
                            modules[i].alpha_u[final_start_node] = nn.Parameter(modules[i].alpha_u[final_start_node])
                            modules[i].alpha_u[final_start_node].requires_grad_()
                            modules[i].beta_l[final_start_node] = nn.Parameter(modules[i].beta_l[final_start_node])
                            modules[i].beta_l[final_start_node].requires_grad_()
                            modules[i].beta_u[final_start_node] = nn.Parameter(modules[i].beta_u[final_start_node])
                            modules[i].beta_u[final_start_node].requires_grad_()

                            # for key in modules[i].alpha_l.keys():
                            #     if key == final_start_node:
                            #         continue
                            #     modules[i].alpha_l[key] = modules[i].alpha_l[key].repeat(all_split.shape[1]*2, 1, 1)
                            #     modules[i].alpha_u[key] = modules[i].alpha_u[key].repeat(all_split.shape[1]*2, 1, 1)
                            #     modules[i].beta_l[key] = modules[i].beta_l[key].repeat(all_split.shape[1]*2, 1, 1)
                            #     modules[i].beta_u[key] = modules[i].beta_u[key].repeat(all_split.shape[1]*2, 1, 1)

                            #     modules[i].alpha_l[key] = nn.Parameter(modules[i].alpha_l[key])
                            #     modules[i].alpha_l[key].requires_grad_()
                            #     modules[i].alpha_u[key] = nn.Parameter(modules[i].alpha_u[key])
                            #     modules[i].alpha_u[key].requires_grad_()
                            #     modules[i].beta_l[key] = nn.Parameter(modules[i].beta_l[key])
                            #     modules[i].beta_l[key].requires_grad_()
                            #     modules[i].beta_u[key] = nn.Parameter(modules[i].beta_u[key])
                            #     modules[i].beta_u[key].requires_grad_()
                    score_try = torch.zeros(all_lb.shape)
                    for i in range(len(modules)):
                        if isinstance(modules[i], BoundReLU):
                            for key, value in modules[i].alpha_l.items():
                                print(f"alpha Key: {key}, Value shape: {value.shape}")
                            for key, value in modules[i].beta_l.items():
                                print(f"beta Key: {key}, Value shape: {value.shape}")
                            print('domain shape:', modules[i].lower_l.shape)
                            print('split shape:', modules[i].S.shape)
                            print('')
                            # print(modules[i].alpha_l, modules[i].beta_l)

                    for domain_ in domains_sub:
                        lb, ub = domain_
                        print((lb < 0) & (ub > 0))
                    _, orig = self.optimized_beta_CROWN(domains_sub, C_new, split_bool_sub, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=4, name=name)
                    # print('original:', orig)
                    for i in range(all_lb.shape[1]):
                        if all_lb[0][i] >= 0 or all_ub[0][i] <= 0:
                            continue

                        all_ub[i][i] = 0
                        all_split[i][i] = 1
                        
                        all_lb[i+all_lb.shape[1]][i] = 0
                        all_split[i+all_lb.shape[1]][i] = -1

                    split_split = torch.split(all_split, split_sizes, dim=1)
                    new_split_bool = [ss.view(-1,split_size) for ss, split_size in zip(split_split,split_sizes)]

                    ub_split = torch.split(all_ub, split_sizes, dim=1)
                    new_ub = [ss.view(-1,split_size) for ss, split_size in zip(ub_split,split_sizes)]

                    lb_split = torch.split(all_lb, split_sizes, dim=1)
                    new_lb = [ss.view(-1,split_size) for ss, split_size in zip(lb_split,split_sizes)]

                    new_domains = []
                    for lbt, ubt in zip(new_lb, new_ub):
                        new_domains.append((lbt,ubt))
                    
                    _, ret = self.optimized_beta_CROWN(new_domains, C_matrix_new, new_split_bool, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=2, name=name)
                    imp = ret-orig
                    split_sizes = [s.shape[1] for s in split_bool_sub]*2
                    imp_split = torch.split(imp, split_sizes, dim=1)
                    new_imp = [ss.view(-1,split_size) for ss, split_size in zip(imp_split,split_sizes)]
                    print(new_imp)

                def try_all_beta():
                    split_sizes = [s.size(1) for s in split_bool_sub]
                    all_lb = torch.cat([c[0] for c in domains_sub], dim=1)
                    all_ub = torch.cat([c[1] for c in domains_sub], dim=1)
                    all_split = torch.cat([s for s in split_bool_sub], dim=1)
                    all_lb = all_lb.repeat(all_lb.shape[1]*2, 1)
                    all_ub = all_ub.repeat(all_ub.shape[1]*2, 1)
                    all_split = all_split.repeat(all_split.shape[1]*2, 1)
                    C_matrix_new = C_new.repeat(1, all_split.shape[1]*2, 1)
                    final_start_node = len(modules) - 1
                    for i in range(all_lb.shape[1]):
                        if all_lb[0][i] >= 0 or all_ub[0][i] <= 0:
                            continue
                        all_ub[i][i] = 0
                        all_split[i][i] = 1
                        
                        all_lb[i+all_lb.shape[1]][i] = 0
                        all_split[i+all_lb.shape[1]][i] = -1

                    split_split = torch.split(all_split, split_sizes, dim=1)
                    new_split_bool = [ss.view(-1,split_size) for ss, split_size in zip(split_split,split_sizes)]

                    ub_split = torch.split(all_ub, split_sizes, dim=1)
                    new_ub = [ss.view(-1,split_size) for ss, split_size in zip(ub_split,split_sizes)]

                    lb_split = torch.split(all_lb, split_sizes, dim=1)
                    new_lb = [ss.view(-1,split_size) for ss, split_size in zip(lb_split,split_sizes)]

                    new_domains = []
                    for lbt, ubt in zip(new_lb, new_ub):
                        new_domains.append((lbt,ubt))


                    sub_domains = []
                    sub_s = []

                    for i in range(new_split_bool[0].shape[0]):
                        sub_domain_i = []
                        sub_s_i = []
                        for domain, s in zip(new_domains, new_split_bool):
                            sub_domain_0 = domain[0][i].unsqueeze(0)
                            sub_domain_1 = domain[1][i].unsqueeze(0)
                            
                            sub_s_ii = s[i].unsqueeze(0)
                            
                            sub_domain_i.append((sub_domain_0, sub_domain_1))
                            sub_s_i.append(sub_s_ii)
                        
                        sub_domains.append(sub_domain_i)
                        sub_s.append(sub_s_i)
                    sc = torch.zeros(1, all_lb.shape[0]).to(x_U)
                    self.initialize_para(modules, name, final_start_node)
                    _, orig = self.optimized_beta_CROWN(domains_sub, C_new, split_bool_sub, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=4, name=name)
                    print(orig)
                    for i, (d, s) in enumerate(zip(sub_domains, sub_s)):
                        t = 160
                        print(i,t)
                        if i!=t and i!=t+all_lb.shape[1]:
                            continue
                        print(i,d,s)
                        self.initialize_para(modules, name, 'all')
                        _, ret = self.optimized_beta_CROWN(d, C_new, s, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=2, name=name)
                        sc[0,i] = ret[0,0]
                        print(sc[0,i])

                    imp = sc-orig
                    print(imp)
                    split_sizes = [s.shape[1] for s in split_bool_sub]*2
                    imp_split = torch.split(imp, split_sizes, dim=1)
                    new_imp = [ss.view(-1,split_size) for ss, split_size in zip(imp_split,split_sizes)]
                    print(new_imp)
                    
                try_all_split_t()
                #try_all_beta()
                break
                self.initialize_para(modules, name, final_start_node)
                d = 0
                ind = 1
                # only 1 iteration
                leng = []
                while(torch.any(ret<0) and C_new.shape[1] <= n):
                    print('')
                    print('Bab round', ind)
                    print('batch:', len(ret))
                    ind+=1
                    min_batch_size = 204.8 #2048*0.1
                    split_depth = self.get_split_depth(C_new.shape[1], min_batch_size)
                    real_ss = min(split_depth,unstable_remain)
                    d += real_ss
                    print('split depth step:', real_ss)
                    print('split depth total:', d)

                    domains_sub, split_bool_sub, C_new = general_split_robustness(domains_sub, split_bool_sub, C_new, modules, name, split_depth = real_ss, final_para = final_para)
                    break
                    unstable_remain -= real_ss

                    _, ret = self.optimized_beta_CROWN(domains_sub, C_new, split_bool_sub, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=2, name=name)

                    domains_sub, split_bool_sub, C_new, verified, ret = self.domain_filter_robusness(ret, domains_sub, split_bool_sub, C_new, modules, name)
                    
                    if(verified == 'verified'):
                        #one batch verified
                        print('this instance is verified!!!!')
                        print('domains size change:', leng)
                        print('')
                        break
                    elif(unstable_remain == 0):
                        #all node split, so it is unknown and need further LP solver to determin, just end the process
                        print('unknown!!!! All nodes split')
                        print('domains size change:', leng)
                        return alpha_lb, ub, 'unknown'
                    print('length of domains:',len(ret))
                    leng.append(len(ret))
                    print('domain remains:', ret)
                if(verified != 'verified'):
                    #one batch is not verified in fixed size of domains
                    print('Unknown!!!! Out of size')
                    print('domains size change:', leng)
                    return alpha_lb, ub, 'unknown'
        return alpha_lb, ub, 'safe'

    

    def boundpropogate_from_layer(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None, optimize=False, out_features = 10, save = 0):
        r"""The bound propagation starting from a given layer. Can be used to compute intermediate bounds or the final bound.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            C (tensor): The initial coefficient matrix. Can be used to represent the output constraints.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            start_node (int): The start node of this propagation. It should be a linear layer.
        Returns:
            ub (tensor): The upper bound of the output of start_node.
            lb (tensor): The lower bound of the output of start_node.
        """
        modules = list(self._modules.values()) if start_node is None else list(self._modules.values())[:start_node + 1]
        # if(optimize == 5):
        #     print('lllllllllll')
        #     # upper_A = C if upper else None
        #     # lower_A = C if lower else None
        #     # upper_sum_b = lower_sum_b = x_U.new([0])
        #     upper_A = C.repeat(404, 1, 1) if upper else None
        #     lower_A = C.repeat(404, 1, 1) if lower else None
        #     upper_sum_b = lower_sum_b = x_U.new([0])
        # else:
        
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0])
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

            batch_size = A.shape[0]
            center = center.expand(batch_size, center.shape[1], center.shape[2])
            diff = diff.expand(batch_size, diff.shape[1], diff.shape[2])

            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            bound = bound.squeeze(-1) + sum_b
            return bound        
        if save == 1:
            modules[-1].lower_A = lower_A.clone().detach()
            modules[-1].lower_sum_b = lower_sum_b.clone().detach()
            modules[-1].upper_A = upper_A.clone().detach()
            modules[-1].upper_sum_b = upper_sum_b.clone().detach()
            modules[-1].x_L = x_L.clone().detach()
            modules[-1].x_U = x_U.clone().detach()

        elif save == 4:
            #in domain update beta
            S = list(self._modules.values())[start_node + 1].S
            lower_mask = (S == 1)
            modules[-1].lower_A[lower_mask] = lower_A[lower_mask]
            modules[-1].lower_sum_b[lower_mask] = lower_sum_b[lower_mask]

            upper_mask = (S == -1)
            modules[-1].upper_A[lower_mask] = upper_A[lower_mask]
            modules[-1].upper_sum_b[lower_mask] = upper_sum_b[lower_mask]


        elif save == 2:
            final_para.lower_A = lower_A.clone().detach()
            final_para.lower_sum_b = lower_sum_b.clone().detach()
            final_para.upper_A = upper_A.clone().detach()
            final_para.upper_sum_b = upper_sum_b.clone().detach()
        elif save == 3:
            final_para.lower_A_ing = lower_A.clone().detach()
            final_para.lower_sum_b_ing = lower_sum_b.clone().detach()
            final_para.upper_A_ing = upper_A.clone().detach()
            final_para.upper_sum_b_ing = upper_sum_b.clone().detach()


        lb = _get_concrete_bound(lower_A, lower_sum_b, sign=-1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign=+1)
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf])
        return ub, lb




def main():
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, help='input data, a tensor saved as a .pth file.')
    parser.add_argument('data', type=str, help='toy tor toy, complex for complex, it is example demo')
    # Parse the command line arguments
    args = parser.parse_args()

    print('use ReLU model')
    if(args.data == 'complex'):
        model = SimpleNNRelu().to(device)
        model.load_state_dict(torch.load('models/relu_model.pth'), strict=False)
        with torch.no_grad():
            weight_values = torch.tensor([
                [ 0.0012, -0.0011,  0.0009, -0.0008,  0.0010, -0.0007,  0.0005, -0.0012,  0.0006, -0.0009],
                [-0.0005,  0.0010, -0.0008,  0.0007, -0.0011,  0.0009, -0.0006,  0.0012, -0.0007,  0.0008],
                [ 0.0009, -0.0006,  0.0011, -0.0010,  0.0007, -0.0008,  0.0012, -0.0005,  0.0009, -0.0007],
                [-0.0008,  0.0007, -0.0012,  0.0009, -0.0006,  0.0011, -0.0009,  0.0005, -0.0010,  0.0006],
                [ 0.0007, -0.0010,  0.0005, -0.0009,  0.0012, -0.0006,  0.0008, -0.0007,  0.0011, -0.0005]
            ]).to(device)

            bias_values = torch.tensor([0.0001, -0.0001, 0.0002, -0.0002, 0.0001]).to(device)

            model[6].weight.copy_(weight_values)
            model[6].bias.copy_(bias_values)
    else:
        model = two_relu_toy_model(in_dim=2, out_dim=2).to(device)

    if(args.data == 'complex'):
        x_test, labels = torch.load(args.data_file)
        batch_size = x_test.size(0)
        x_test = x_test.reshape(batch_size, -1).to(device)
        labels = torch.tensor([1]).long().to(device)
        output = model(x_test)
        y_size = output.size(1) - 1
    else:
        x_test = torch.tensor([[0., 0.]]).float().to(device)
        labels = torch.tensor([0]).long().to(device)
        output = model(x_test)
        y_size = output.size(1) - 1

    print("Network prediction: {}".format(output))
    if(args.data == 'complex'):
        eps = 0.025
    else:
        eps = 1
    x_u = x_test + eps
    x_l = x_test - eps

    print(f"Verifiying Pertubation - {eps}")
    start_time = time.time()
    boundedmodel = BoundedSequential.convert(model)
    boundedmodel.labels = labels
    lb, ub, verify = boundedmodel.BaB(x_U=x_u, x_L=x_l, n=2048)

    if(verify == 'safe'):
        print('')
        print('all batches are verified!')

#print the result in 'try.txt'
from contextlib import redirect_stdout
with open('try.txt', 'w') as f:
    with redirect_stdout(f):
        main()