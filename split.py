import torch
import torch.nn as nn
from torch.nn import functional as F
from relu_alpha_beta import BoundReLU
import itertools

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_ratio(lb, ub):
    lower_temp = lb.clamp(max=0)
    upper_temp = F.relu(ub)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio
    return slope_ratio, intercept

def babsr_score(module, linear, name, lb, ub):
    
    r"""Compute branching scores.

    Args:
        module (BoundReLU): the module for the ReLU layer.

        linear (BoundLinear): the module for the linear layer before ReLU layer, to get the bias.

        name (int): the id of the instance.

        lb (tensor): lower bounds for one pre-activation layer.

        ub (tensor): upper bounds for one pre-activation layer.

    return
        score_candidate (tensor): same structure as lb, indicating the score for this neuron.
    """
    shape = module.last_lA_list[name].shape
    ratio = module.last_lA_list[name].view(shape[1],-1,shape[2])
    ratio_temp_0, ratio_temp_1 = compute_ratio(lb, ub)
    intercept_temp = torch.clamp(ratio, max=0)
    intercept_candidate = intercept_temp * ratio_temp_1.unsqueeze(1)
    b_temp = linear.bias
    b_temp = b_temp * ratio

    ratio_temp_0 = ratio_temp_0.unsqueeze(1)
    bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
    bias_candidate_2 = b_temp * ratio_temp_0
    bias_candidate = torch.min(bias_candidate_1, bias_candidate_2)  # max for babsr by default
    score_candidate = (bias_candidate + intercept_candidate).abs()
    return score_candidate

def babsr_score_considernp(module, linear, name, lb, ub):
    
    r"""Compute branching scores.
    module: the module for the ReLU layer.
    linear: the module for the linear layer before ReLU layer, to get the bias.
    name: the id of the instance
    lb: lower bounds for one pre-activation layer.
    ub: upper bounds for one pre-activation layer.

    return
    score_candidate: same structure as lb, indicating the score for this neuron.
    """
    shape = module.last_lA_list[name].shape
    ratio = module.last_lA_list[name].view(shape[1],-1,shape[2])
    ratio_temp_0, ratio_temp_1 = compute_ratio(lb, ub)
    intercept_temp = torch.clamp(ratio, max=0)
    intercept_candidate = intercept_temp * ratio_temp_1.unsqueeze(1)
    b_temp = linear.bias

    ratio_temp_0 = ratio_temp_0.unsqueeze(1)
    # print('aaaaaaa',ratio)
    # original = ratio_temp_0 * torch.clamp(ratio, max=0) + ratio_temp_0 * torch.clamp(ratio, min=0)
    # original = ratio_temp_0 * torch.clamp(ratio, min=0) - ratio_temp_0 * torch.clamp(ratio, max=0)
    # original = ratio_temp_0 * torch.clamp(ratio, max=0) + module.alpha_l[module.final_start_node].transpose(0, 1) * torch.clamp(ratio, min=0)
    # original = ratio_temp_0 * torch.clamp(ratio, min=0) + module.alpha_l[module.final_start_node].transpose(0, 1) * torch.clamp(ratio, max=0)
    bias_candidate_1 = b_temp * original - b_temp * ratio
    bias_candidate_2 = b_temp * original
    bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)  # max for babsr by default
    score_candidate = (bias_candidate + intercept_candidate).abs()
    return score_candidate


def generate_combinations_with_indices(domains, mask_indices, split_bool, C_matrix, modules, split_depth):
    """split the domians and modify the parameter based on the topk score
    domains: list, bounds for different pre-activation layers.
    mask_indeces: Indices of the elements to be split, it is determined by topk score in each batch.
    split_bool: list, [split_1, split_2, ...], each split has the same shape with lower bound and the length is the same with domains,
                indicates the split logic, 0 for no split, -1 for active, 1 for inactive
    C_matrix: the initial coefficient matrix, the shape is (1, num_batch, num_output)
    modules: list, [module1, module2, ...], the list of all ReLU layers.
    split_depth: ind, the split depth.

    return
    new_domains: list, new bounds for different pre-activation layers
                after split. Length will change based on the split depth, it should be len(domains) * 2 ** split_depth
    new_split_bool: list, [split_1, split_2, ...], indicates the new split logic after split, 
                    Length will change based on the split depth, it should be len(domains) * 2 ** split_depth
    C_matrix_new: the initial coefficient matrix after split, the shape is (1, num_batch * 2 ** split_depth, num_output)
    """
    batch_size = len(domains)
    num_combinations = 2 ** split_depth
    new_domains = []
    new_split_bool = []
    combinations = torch.tensor(list(itertools.product([0, 1], repeat=split_depth)), device=domains[0][0].device)
    C_matrix_new = C_matrix.repeat(1, num_combinations, 1)
    final_start_node = len(modules) - 1
    ind = 0
    for i in range(len(modules)):
        if isinstance(modules[i], BoundReLU):
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
    all_lb = torch.cat([c[0] for c in domains], dim=1)
    all_ub = torch.cat([c[1] for c in domains], dim=1)
    # all_masks = torch.cat([m for m in masks], dim=1)
    all_split = torch.cat([s for s in split_bool], dim=1)
    
    num_elements = all_lb.size(1)
    batch_size = all_lb.size(0)

    # Repeat to match the number of combinations
    
    repeated_lb = all_lb.repeat(num_combinations, 1)
    repeated_ub = all_ub.repeat(num_combinations, 1)
    repeated_split = all_split.repeat(num_combinations, 1)
    
    for i, combination in enumerate(combinations):
        combo_values = combination.float()
        j = 0
        for mask_indice in mask_indices:
            for idx, combo in zip(mask_indice, combo_values):
                if combo == 0:
                    repeated_lb[i*batch_size+j, idx] = 0
                    repeated_split[i*batch_size+j, idx] = -1
                else:
                    repeated_ub[i*batch_size+j, idx] = 0
                    repeated_split[i*batch_size+j, idx] = 1
                
            j += 1
    
    #reshape to the original
    split_sizes = [s.size(1) for s in split_bool]
    split_split = torch.split(repeated_split, split_sizes, dim=1)
    new_split_bool = [ss.view(-1,split_size) for ss, split_size in zip(split_split,split_sizes)]

    new_domains = []

    lb_split = torch.split(repeated_lb, split_sizes, dim=1)
    lb_final = [ss.view(-1,split_size) for ss, split_size in zip(lb_split,split_sizes)]

    ub_split = torch.split(repeated_ub, split_sizes, dim=1)
    ub_final = [ss.view(-1,split_size) for ss, split_size in zip(ub_split,split_sizes)]

    for lb, ub in zip(lb_final, ub_final):
        new_domains.append((lb,ub))

    return new_domains, new_split_bool, C_matrix_new
        

def general_split_robustness(domains, split_bool, C_matrix, modules, name, split_depth = 1):
    """the main function of split for branching
    domains: list, bounds for different pre-activation layers.
    split_bool: list, [split_1, split_2, ...], each split has the same shape with lower bound and the length is the same with domains,
                indicates the split logic, 0 for no split, -1 for active, 1 for inactive
    C_matrix: the initial coefficient matrix, the shape is (1, num_batch, num_output)
    modules: list, [module1, module2, ...], the list of all ReLU layers.
    name: the id of the instance
    split_depth: ind, the split depth.

    return
    new_domains: list, new bounds for different pre-activation layers
                after split. Length will change based on the split depth, it should be len(domains) * 2 ** split_depth
    new_split_bool: list, [split_1, split_2, ...], indicates the new split logic after split, 
                    Length will change based on the split depth, it should be len(domains) * 2 ** split_depth
    C_matrix_new: the initial coefficient matrix after split, the shape is (1, num_batch * 2 ** split_depth, num_output)
    """
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
    for single_domains, module, linear in zip(domains, relu_list, linear_list):
        lb, ub = single_domains[0].clone().detach(), single_domains[1].clone().detach()
        mask = (lb < 0) & (ub > 0)
        # score = babsr_score(module, linear, name, lb, ub)
        score = babsr_score_considernp(module, linear, name, lb, ub)
        score = score.mean(1)
        score = torch.where(mask, score, torch.tensor(0, device=device))
        scores.append(score)
        i += 1
    # print('using babsr,', scores)
    def mark_topk_indices(scores, k):
        batch_size = scores[0].size(0)
        flattened_scores = [s.view(batch_size, -1) for s in scores]
        concatenated_scores = torch.cat(flattened_scores, dim=1)
        topk_values, topk_indices = torch.topk(concatenated_scores, k=k, dim=1, largest=True)
        return topk_indices
    masks = mark_topk_indices(scores, split_depth)
    new_domains, new_split_bool, C_matrix_new = generate_combinations_with_indices(domains, masks, split_bool, C_matrix, modules, split_depth)

    return new_domains, new_split_bool, C_matrix_new