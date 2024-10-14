import torch
import torch.nn as nn
from torch.nn import functional as F
from relu_alpha_beta import BoundReLU
import itertools

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
global input_score
input_score = {}
global input_score_lower_global
input_score_lower_global = {}
global input_score_upper_global
input_score_upper_global = {}
global mask
def compute_ratio(lb, ub):
    lower_temp = lb.clamp(max=0)
    upper_temp = F.relu(ub)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio
    return slope_ratio, intercept

def babsr_score_func(module, linear, name, lb, ub):
    
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
    # original = ratio_temp_0 * torch.clamp(ratio, max=0) + ratio_temp_0 * torch.clamp(ratio, min=0)
    # original = ratio_temp_0 * torch.clamp(ratio, max=0) + module.init_l * torch.clamp(ratio, min=0)
    original = ratio_temp_0 * torch.clamp(ratio, max=0) + module.alpha_l[module.final_start_node].transpose(0, 1) * torch.clamp(ratio, min=0)
    # bias_candidate_1 = b_temp * original - b_temp * ratio
    # bias_candidate_2 = b_temp * original
    # bias_candidate = torch.min(bias_candidate_1, bias_candidate_2)  # max for babsr by default
    # score_candidate = (bias_candidate + intercept_candidate).abs()

    bias_candidate_1 = b_temp * original - b_temp * ratio + intercept_candidate
    bias_candidate_2 = b_temp * original + intercept_candidate
    bias_candidate = torch.min(bias_candidate_1, bias_candidate_2)  # max for babsr by default
    score_candidate = bias_candidate.abs()
    return score_candidate

def normalize_to_01(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def babsr_score_considernp_input(module, linear, name, lb, ub, input_lower, input_upper):
    
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
    # original = ratio_temp_0 * torch.clamp(ratio, max=0) + ratio_temp_0 * torch.clamp(ratio, min=0)
    # original = ratio_temp_0 * torch.clamp(ratio, max=0) + module.init_l * torch.clamp(ratio, min=0)
    original = ratio_temp_0 * torch.clamp(ratio, max=0) + module.alpha_l[module.final_start_node].transpose(0, 1) * torch.clamp(ratio, min=0)

    bias_candidate_1 = b_temp * original - b_temp * ratio + intercept_candidate
    bias_candidate_2 = b_temp * original + intercept_candidate
    # bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)  # max for babsr by default
    print('babsr lower score:', bias_candidate_2)
    print('babsr upper score:', bias_candidate_1)
    input_upper_norm = normalize_to_01(input_upper)
    input_lower_norm = normalize_to_01(input_lower)
    bias_candidate_1_norm = normalize_to_01(bias_candidate_1.abs())
    bias_candidate_2_norm = normalize_to_01(bias_candidate_2.abs())

    if bias_candidate_1.shape[0]==1:
        bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)
        # bias_candidate = normalize_to_01(torch.max(bias_candidate_1_norm * input_upper_norm, bias_candidate_2_norm * input_lower_norm))  # max for babsr by default
        # bias_candidate = normalize_to_01(torch.max(bias_candidate_1.abs() * input_upper_norm, bias_candidate_2.abs() * input_lower_norm))  # max for babsr by default
        # bias_candidate = normalize_to_01(torch.max(bias_candidate_1, bias_candidate_2).abs()*torch.max(input_upper_norm, input_lower_norm))  # max for babsr by default
        # bias_candidate = normalize_to_01(torch.max(bias_candidate_1.abs(), bias_candidate_2.abs()))  # max for babsr by default
        # bias_candidate = normalize_to_01(torch.max(bias_candidate_1 * input_upper_norm, bias_candidate_2 * input_lower_norm))  # max for babsr by default
    else:
        bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)  # max for babsr by default
        # bias_candidate = normalize_to_01(torch.max(bias_candidate_1.abs() * input_upper_norm, bias_candidate_2.abs() * input_lower_norm))
    score_candidate = bias_candidate.abs()
    return score_candidate

def center_distance(center, A, b):
    numerator = torch.abs(torch.matmul(A, center.unsqueeze(-1)).squeeze(-1) + b)
    denominator = torch.sqrt(torch.sum(A ** 2, dim=-1))
    # print('num:', numerator)
    # print('denominator:', denominator)
    distances = numerator / denominator
    return distances

def fast_solve(a, c, d, epsilon):
    def f(beta):
        return - (epsilon * (a.unsqueeze(0) * beta.unsqueeze(-1) + c)).abs().sum(-1) + d * beta
    q = -c / a
    order_idx = torch.argsort(q)  # Dominates time complexity.
    sorted_a = (a * epsilon)[order_idx]   # Scale "a" by epsilon.
    sum_a_neg = -sorted_a.abs().cumsum(0)
    # print(sum_a_neg.shape)
    total_a = -sum_a_neg[-1]  # sorted_a.abs().sum()
    sum_a_pos = total_a + sum_a_neg
    # Supergradient at the i-th crossing-zero point is in range [super_gradients[i-1], super_gradients[i]]
    # For i = 0, super_gradients[-1] = total_a * epsilon but we don't need to explicitly compute it - we compare to f(0) below.
    super_gradients = (sum_a_pos + sum_a_neg) + d
    # Search where the supergradient contain 0, which is the point of maximum.
    best_idx = torch.searchsorted(-super_gradients, 0, right=True)
    if best_idx >= a.size(0):
        # This should not happen in our case, if our constraints are from unstable neurons.
        # print('Objective is unbounded.')
        return -float("inf"), -float("inf")
    else:
        # If the best solution is with beta < 0, we have to clamp it.
        best_beta = q[order_idx[best_idx]].clamp(min=0)
        best_obj = f(best_beta)
        # We still need to compare to f(0), which is an additional end point.
        f0 = (-epsilon * c.abs()).sum()
        if best_obj < f0:
            best_obj = f0
            best_beta = torch.tensor(0.)
        # print(f'best obj is {best_obj.item()}, best beta is {best_beta}, idx {best_idx}')
        return best_obj, best_beta

def _get_concrete_bound(A, sum_b, sign, x_U, x_L):
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
def input_split_score(module, linear, name, lb, ub, final_para, index, mask):
    global input_score
    # print('when spliting:')
    # print('lower_A',linear.lower_A.shape)
    # print('lower_sum_b',linear.lower_sum_b.shape)
    # print('upper_A',linear.upper_A.shape)
    # print('upper_sum_b',linear.upper_sum_b.shape)
    # print('x', linear.x_L, linear.x_U)
    # print('final_para_lower_A',final_para.lower_A_actual.shape)
    # print('final_para_lower_sum_b',final_para.lower_sum_b_actual.shape)
    # print('final_para_upper_A',final_para.upper_A_actual.shape)
    # print('final_para_upper_sum_b',final_para.upper_sum_b_actual.shape)
    lower_A, lower_sum_b = linear.lower_A, linear.lower_sum_b
    upper_A, upper_sum_b = linear.upper_A, linear.upper_sum_b
    final_para_lower_A, final_para_lower_sum_b = final_para.lower_A_actual, final_para.lower_sum_b_actual
    final_para_upper_A, final_para_upper_sum_b = final_para.upper_A_actual, final_para.upper_sum_b_actual
    score = torch.full((final_para_lower_A.shape[1], lower_A.shape[1]), float('-inf'))
    if final_para_lower_A.shape[1] != 1:
        return input_score[index]
    
    x_L, x_U = linear.x_L, linear.x_U
    center = (x_L + x_U) / 2.0
    diff =  (x_U - x_L) / 2.0
    # lower_distance = center_distance(center, lower_A, lower_sum_b)
    # upper_distance = center_distance(center, upper_A, upper_sum_b)
    # score = torch.min(lower_distance,upper_distance)
    original_bound = _get_concrete_bound(final_para_lower_A, final_para_lower_sum_b, -1 , x_U, x_L)
    # print('original bound:', original_bound)
    for i in range(final_para_lower_A.shape[1]):
        current_final_para_lower_A, current_final_para_lower_sum_b = final_para_lower_A[:, i, :].view(-1), final_para_lower_sum_b[:, i].view(-1)
        for j in range(lower_A.shape[1]):
            if mask[0, j]==False:
                continue
            # only consider lower constraint, inactive
            current_lower_A, current_lower_sum_b = lower_A[:, j, :].view(-1), lower_sum_b[:, j].view(-1)
            d = current_lower_A @ center.view(-1) + current_lower_sum_b[0]
            extra = current_final_para_lower_A @ center.view(-1) + current_final_para_lower_sum_b[0]
            obj_lower, beta = fast_solve(current_lower_A, current_final_para_lower_A, d, diff.view(-1))
            score_candidate_lower = obj_lower + extra

            # only consider upper constraint, active
            current_upper_A, current_upper_sum_b = -upper_A[:, j, :].view(-1), -upper_sum_b[:, j].view(-1)
            d = current_upper_A @ center.view(-1) + current_upper_sum_b[0]
            obj_upper, beta = fast_solve(current_upper_A, current_final_para_lower_A, d, diff.view(-1))
            score_candidate_upper = obj_upper + extra
            print('score_candidate:', score_candidate_lower, score_candidate_upper)

            score[i, j] = torch.max(score_candidate_lower, score_candidate_upper)

    # print('current_final_para_lower_A',current_final_para_lower_A.shape)
    # print('current_final_para_lower_sum_b',current_final_para_lower_sum_b.shape)
    # print('current_lower_A',current_lower_A.shape)
    # print('current_lower_sum_b',current_lower_sum_b.shape)
    # print('x0 shape:', center.shape)
    # print('epsilon shape:,', diff.shape)
    # print('d:', d)
    # print('score:', score)
    input_score[index] = score.to(device)
    return score


def input_split_score_sub(module, linear, name, lb, ub, final_para, index, mask):
    global input_score_lower_global
    global input_score_upper_global
    # print('when spliting:')
    # print('lower_A',linear.lower_A.shape)
    # print('lower_sum_b',linear.lower_sum_b.shape)
    # print('upper_A',linear.upper_A.shape)
    # print('upper_sum_b',linear.upper_sum_b.shape)
    # print('x', linear.x_L, linear.x_U)
    # print('final_para_lower_A',final_para.lower_A_actual.shape)
    # print('final_para_lower_sum_b',final_para.lower_sum_b_actual.shape)
    # print('final_para_upper_A',final_para.upper_A_actual.shape)
    # print('final_para_upper_sum_b',final_para.upper_sum_b_actual.shape)
    lower_A, lower_sum_b = linear.lower_A, linear.lower_sum_b
    upper_A, upper_sum_b = linear.upper_A, linear.upper_sum_b
    final_para_lower_A, final_para_lower_sum_b = final_para.lower_A_actual, final_para.lower_sum_b_actual
    final_para_upper_A, final_para_upper_sum_b = final_para.upper_A_actual, final_para.upper_sum_b_actual
    # score_lower = torch.full((final_para_lower_A.shape[1], lower_A.shape[1]), float('-inf'))
    # score_upper = torch.full((final_para_lower_A.shape[1], lower_A.shape[1]), float('-inf'))
    score_lower = torch.zeros((final_para_lower_A.shape[1], lower_A.shape[1]))
    score_upper = torch.zeros((final_para_lower_A.shape[1], lower_A.shape[1]))
    if final_para_lower_A.shape[1] != 1:
        return input_score_lower_global[index], input_score_upper_global[index]
    # print(mask) 
    x_L, x_U = linear.x_L, linear.x_U
    center = (x_L + x_U) / 2.0
    diff =  (x_U - x_L) / 2.0
    # lower_distance = center_distance(center, lower_A, lower_sum_b)
    # upper_distance = center_distance(center, upper_A, upper_sum_b)
    # score = torch.min(lower_distance,upper_distance)
    original_bound = _get_concrete_bound(final_para_lower_A, final_para_lower_sum_b, -1 , x_U, x_L)
    # print('original bound:', original_bound)
    
    for i in range(final_para_lower_A.shape[1]):
        current_final_para_lower_A, current_final_para_lower_sum_b = final_para_lower_A[:, i, :].view(-1), final_para_lower_sum_b[:, i].view(-1)
        for j in range(lower_A.shape[1]):
            if mask[0, j]==False:
                continue
            # only consider lower constraint, inactive
            current_lower_A, current_lower_sum_b = lower_A[:, j, :].view(-1), lower_sum_b[:, j].view(-1)
            # current_lower_A, current_lower_sum_b = upper_A[:, j, :].view(-1), upper_sum_b[:, j].view(-1)
            d = current_lower_A @ center.view(-1) + current_lower_sum_b[0]
            extra = current_final_para_lower_A @ center.view(-1) + current_final_para_lower_sum_b[0]
            obj_lower, beta = fast_solve(current_lower_A, current_final_para_lower_A, d, diff.view(-1))
            score_candidate_lower = obj_lower + extra
            score_lower[i,j] = torch.max(torch.tensor(0), score_candidate_lower - original_bound)
            # if (score_candidate_lower - original_bound)>torch.tensor(3.0):
            #     print('a',current_lower_A)
            #     print('c',current_final_para_lower_A)
            #     print('d',d)
            #     print('epsilon',diff.view(-1))

            # only consider upper constraint, active
            current_upper_A, current_upper_sum_b = -upper_A[:, j, :].view(-1), -upper_sum_b[:, j].view(-1)
            # current_upper_A, current_upper_sum_b = -lower_A[:, j, :].view(-1), -lower_sum_b[:, j].view(-1)
            d = current_upper_A @ center.view(-1) + current_upper_sum_b[0]
            obj_upper, beta = fast_solve(current_upper_A, current_final_para_lower_A, d, diff.view(-1))
            score_candidate_upper = obj_upper + extra
            score_upper[i,j] = torch.max(torch.tensor(0), score_candidate_upper - original_bound)
            print('score_candidate:', score_candidate_lower, score_candidate_upper)
            

    # print('current_final_para_lower_A',current_final_para_lower_A.shape)
    # print('current_final_para_lower_sum_b',current_final_para_lower_sum_b.shape)
    # print('current_lower_A',current_lower_A.shape)
    # print('current_lower_sum_b',current_lower_sum_b.shape)
    # print('x0 shape:', center.shape)
    # print('epsilon shape:,', diff.shape)
    # print('d:', d)
    # print('score:', score)
    input_score_lower_global[index] = score_lower.to(device)
    input_score_upper_global[index] = score_upper.to(device)
    return input_score_lower_global[index], input_score_upper_global[index]
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
        

def general_split_robustness(domains, split_bool, C_matrix, modules, name, split_depth = 1, final_para = None):
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
    global input_score
    global input_score_lower_global
    global input_score_upper_global
    global mask
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
    for single_domains, module, linear, split in zip(domains, relu_list, linear_list, split_bool):
        lb, ub = single_domains[0].clone().detach(), single_domains[1].clone().detach()
        mask = (lb < 0) & (ub > 0)
        # score = babsr_score(module, linear, name, lb, ub)
        input_split_score_sub(module, linear, name, lb, ub, final_para, i, mask)
        # input_score_user = input_score[i].detach().clone()
        # input_score_mask = torch.where(mask, input_score_user, torch.tensor(float('-inf'), device=device))

        # babsr_score = babsr_score_considernp(module, linear, name, lb, ub)
        # babsr_score = babsr_score_func(module, linear, name, lb, ub)
        print('input split lower bound:', input_score_lower_global[i])
        print('input split upper bound:', input_score_upper_global[i])
        babsr_score = babsr_score_considernp_input(module, linear, name, lb, ub, input_score_lower_global[i], input_score_upper_global[i])
        babsr_score = babsr_score.mean(1)
        babsr_score = torch.where(mask, babsr_score, torch.tensor(0, device=device))
        
        # score = babsr_score + input_score_mask
        score = babsr_score
        # score = input_score_mask
        # print('innnnnnnnn:', input_score_mask)
        # print('nnnnnnm:', normalized_input_score)
        # print('sssssssssssssss:', score)

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