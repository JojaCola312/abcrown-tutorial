import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model import SimpleNNRelu, two_relu_toy_model
from linear import BoundLinear
from relu_alpha_beta import BoundReLU
from split import general_split_robustness
import time
import argparse
import torch.optim as optim


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
        '''Get the initial coefficient matrix for robustness verification

        Args:
            old_C (tensor): The initial coefficient matrix, which should be an identical matrix.
                            Shape should be (1, out_features, out_features)
            labels (list): The list of true labels

        Return:
            new_C (tensor): The initial coefficient matrix for robustness verification.
                            Shape should be (1, out_features - 1, out_features)

        '''
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
        '''Main function to get the optimized bound

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
        '''
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
                                              lower=lower, start_node=i, optimize=0, out_features = C_matrix.shape[1])

        if(optimize==0):
            print('using CROWN')
            return ub, lb
        elif(optimize==1):
            print('using alpha-CROWN')
            opt = False
            for module in modules:
                if isinstance(module, BoundReLU):
                    lr = 0.1
                    if(opt == False):
                        optimizer = optim.Adam(list(module.alpha_l.values()), lr=lr)
                        optimizer.add_param_group({'params': list(module.alpha_u.values()),'lr': lr})
                        opt = True
                    else:
                        optimizer.add_param_group({'params': list(module.alpha_l.values()),'lr': lr})
                        optimizer.add_param_group({'params': list(module.alpha_u.values()),'lr': lr})
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
                                                                    start_node=i - 1, optimize=0, out_features = modules[i - 1].out_features)
                        modules[i].upper_u = ub
                        modules[i].lower_l = lb
                        

                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=C_matrix, upper=upper,
                                                lower=lower, start_node=i, optimize=True, out_features = C_matrix.shape[1])
                optimizer.zero_grad()
                loss = -lb.sum()
                # print('ret:',lb) 
                if(torch.any(lb > best_lb)):
                    for i in range(len(modules)):
                        if isinstance(modules[i], BoundReLU):
                            modules[i].update(name, 1, lb>best_lb)
                            if isinstance(modules[i - 1], BoundLinear):
                                newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                                ubt, lbt = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                                        start_node=i - 1, optimize=0, out_features = modules[i - 1].out_features)
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
            
            # print('split:', split)
            # print('name:', name)
            opt = False
            lr_a = 0.01
            lr_b = 0.05

            alpha_params = []
            beta_params = []
            alpha_schedulers = []
            beta_schedulers = []

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
                
                ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                                C=C_matrix, upper=upper,
                                                lower=lower, start_node=i, optimize=2, out_features = C_matrix.shape[1])

                optimizer.zero_grad()
                loss = -lb.sum()                 
                # print('ret:', lb)

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

            print('final_ret:',best_lb)
            return best_ub, best_lb


    def initialize_para(self, modules, domains, C_new, S, name, start_node):
        '''initialize the parameter, only be called at the begining of each instance'''

        for i in range(len(modules)):
            if isinstance(modules[i], BoundReLU):
                modules[i].alpha_l[start_node] = modules[i].alpha_l_list[0][start_node][:,name-1,:].unsqueeze(1)
                modules[i].alpha_u[start_node] = modules[i].alpha_u_list[0][start_node][:,name-1,:].unsqueeze(1)
                modules[i].beta_l[start_node] = modules[i].beta_l_list[0][start_node][:,name-1,:].unsqueeze(1)
                modules[i].beta_u[start_node] = modules[i].beta_u_list[0][start_node][:,name-1,:].unsqueeze(1)
                modules[i].last_lA_list[name] = modules[i].last_lA_list[0][:,name-1,:].unsqueeze(1)
    
    

    def domain_filter_robusness(self, lb, domains, split_bool, C, modules, name):
        '''Domain filter after each iteration

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
        '''
        mask = lb < 0
        #all lbs are smaller than 0, no domain change
        if(torch.all(mask)):
            return domains, split_bool, C, 'remain', lb[mask]
        #all lbs are larger than 0, then verified
        if(torch.all(~mask)):
            return domains, split_bool, C, 'verified', lb[mask]

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
        '''Main function, first try CROWN and alpha-CROWN, then alpha-beta-CROWN
        
        Args:
            x_U (tensor): The upper bound of x.
            x_L (tensor): The lower bound of x.
            n (int): The threshold of maximum domain size of 'unknown'
        
        Return:
            lb (tensor): The final output lower bound. Not necessary if finally using alpha-beta-CROWN.
            ub (tensor): The final output upper bound. Not necessary if finally using alpha-beta-CROWN.
            verified_status (string): 'safe' if all instances are verified. 'unsafe' if there exists instance unsafe. 'unknown'
                                      if it is not verified within pre-determined domain size.
        '''
        modules = list(self._modules.values())
        domains = []
        for i in range(len(modules)):
            # We only need the bounds before a ReLU layer
            if isinstance(modules[i], BoundReLU):
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
                domains.append((lb,ub))
        C_matrix = self.get_C(torch.eye(modules[i].out_features).unsqueeze(0).to(x_U),self.labels)
        
        def get_s(domains,device):
            S = []
            for domains_single in domains:                
                zero_matrix = torch.zeros(domains_single[0].shape)
                S.append(zero_matrix.to(device))
            return S
        split_bool = get_s(domains,device)


        ub, lb = self.optimized_beta_CROWN(domains, C_matrix, split_bool, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=0, name=0)
        print('result from crown is:',lb)
        if(torch.all(lb>0)):
            print('verified using crown')
            return lb, ub, 'safe'

        ub, alpha_lb = self.optimized_beta_CROWN(domains, C_matrix, split_bool, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=1, name=0)
        print('result from alpha crown is:',alpha_lb)
        if(torch.all(lb>0)):
            print('verified using alpha crown')
            return alpha_lb, ub, 'safe'

        domains = []
        unstable_size = 0
        for i in range(len(modules)):
            # We only need the bounds before a ReLU/HardTanh layer
            if isinstance(modules[i], BoundReLU):
                domains.append((modules[i].lb, modules[i].ub))
                mask = (modules[i].lb < 0) & (modules[i].ub > 0)
                unstable_size += mask.sum().item()
        print('using alpha-beta-crown')
        print('size of unstable nuerons:',unstable_size)
        num_out_features = C_matrix.shape[1]
        final_start_node = len(modules) - 1

        for i in range(num_out_features):
            # if(i == 1):
            #     break
            # i = 3
            unstable_remain = unstable_size
            C_new = C_matrix[:,i,:].unsqueeze(1)
            domains_sub = deep_copy_structure(domains)
            split_bool_sub = deep_copy_structure(split_bool)
            ret = alpha_lb[:,i]
            name = i+1
            if(torch.all(ret > 0)):
                print('')
                print('instance', i, 'is verified with lb', ret, 'and C', C_new)
            else:
                print('')
                print('instance', i, 'is not verified with lb', ret, 'and C', C_new)

                self.initialize_para(modules,domains_sub,C_new,split_bool_sub,name,final_start_node)
                d = 0
                ind = 1
                while(torch.any(ret<0) and C_new.shape[1] <= n):
                    print('')
                    print('Bab round', ind)
                    print('batch:', len(ret))
                    ind+=1
                    # assert C[0][0] == len(split_bool_list_global)
                    # assert len(C_list) == C_new.shape[1]
                    min_batch_size = 204.8
                    split_depth = self.get_split_depth(C_new.shape[1], min_batch_size)
                    real_ss = min(split_depth,unstable_remain)
                    d += real_ss
                    print('split depth step:', real_ss)
                    print('split depth total:', d)

                    
                    domains_sub, split_bool_sub, C_new = general_split_robustness(domains_sub, split_bool_sub, C_new, modules, name, split_depth = real_ss)
                    unstable_remain -= real_ss

                    _, ret = self.optimized_beta_CROWN(domains_sub, C_new, split_bool_sub, x_U=x_U, x_L=x_L, upper=True, lower=True, optimize=2, name=name)

                    domains_sub, split_bool_sub, C_new, verified, ret = self.domain_filter_robusness(ret, domains_sub, split_bool_sub, C_new, modules, name)

                    if(verified == 'verified'):
                        #one batch verified
                        print('this instance is verified!!!!')
                        print('')
                        break
                    elif(unstable_remain == 0):
                        #one batch unsafe, so just end the process
                        print('unsafe!!!!')
                        return alpha_lb, ub, 'unsafe'
                    print('length of domains:',len(ret))
                    print('domain remains:', ret)
                if(verified != 'verified'):
                    #one batch is not verified in fixed size of domains
                    print('Unknown!!!! Out of size')
                    return alpha_lb, ub, 'unknown'
        return alpha_lb, ub, 'safe'


    def boundpropogate_from_layer(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None, optimize=False, out_features = 10):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, help='input data, a tensor saved as a .pth file.')
    parser.add_argument('data', type=str, help='toy tor toy, complex for complex, it is example demo')
    # Parse the command line arguments
    args = parser.parse_args()

    print('use ReLU model')
    if(args.data == 'complex'):
        model = SimpleNNRelu().to(device)
        model.load_state_dict(torch.load('models/relu_model.pth'))
    else:
        model = two_relu_toy_model(in_dim=2, out_dim=2).to(device)

    if(args.data == 'complex'):
        x_test, labels = torch.load(args.data_file)
        batch_size = x_test.size(0)
        x_test = x_test.reshape(batch_size, -1).to(device)
        labels = torch.tensor([labels]).long().to(device)
        output = model(x_test)
        y_size = output.size(1) - 1
    else:
        x_test = torch.tensor([[0., 0.]]).float().to(device)
        labels = torch.tensor([0]).long().to(device)
        output = model(x_test)
        y_size = output.size(1) - 1

    print("Network prediction: {}".format(output))
    if(args.data == 'complex'):
        eps = 0.0265
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

from contextlib import redirect_stdout
with open('try.txt', 'w') as f:
    with redirect_stdout(f):
        main()