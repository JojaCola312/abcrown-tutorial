def general_split(self, C_list, split_lists, split_depth):
        C_split = []
        split_bool_list = []
        name_list = []
        relu_list = []
        linear_list = []
        modules = list(self._modules.values())
        for i, module in enumerate(modules):
            if isinstance(module, BoundReLU):
                relu_list.append(module)
                linear_list.append(modules[i-1])
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
            #if all valid, then all splitted
            all_valid = all(is_valid_diagonal_matrix(matrix) for matrix in split_list)
            if(all_valid):
                continue
            _, _, C, name = P
            num_relu = len(C)
            C_1, C_2 = deep_copy_structure(C), deep_copy_structure(C)
            split_bool_1, split_bool_2 = deep_copy_structure(split_list), deep_copy_structure(split_list)
            rank_list = []
            i = 0
            #calculate the score of all possible split for each C, here just ub-lb
            for single_C, module, linear in zip(C, relu_list, linear_list):
                lb, ub = single_C[0].clone().detach(), single_C[1].clone().detach()
                # lb, ub = module.lower_l, module.upper_u
                
                mask = (lb < 0) & (ub > 0)
                score = self.babsr_score(module, linear, name, lb, ub)
                score = score.mean(1)
                score = torch.where(mask, score, torch.tensor(-float('inf'), device=device))
                # print('score:',score)

                # max_index = torch.unravel_index(torch.argmax(score), score.shape)
                # max_score = score[max_index]
                # #i for index of the relu in the whole model, max_index for the lb index in a certain relu, max_diff is the score
                # rank_list.append((max_score,max_index,i))

                topk_scores, topk_indices = torch.topk(score.view(-1), split_depth)
                max_indices = torch.unravel_index(topk_indices, score.shape)
                for idx, max_score in enumerate(topk_scores): 
                    max_index = (max_indices[0][idx], max_indices[1][idx])
                    rank_list.append((max_score, max_index, i))
                i += 1
            # if(len(rank_list)==0):
            #     continue
            sorted_rank = sorted(rank_list, key=lambda x: x[0], reverse=True)
            # max_range_term = sorted_rank[0]
            max_range_terms = sorted_rank[:split_depth]

            #change ub
            for max_range_term in max_range_terms:
                C_1[max_range_term[2]][1][max_range_term[1]] = 0
                #change lb
                C_2[max_range_term[2]][0][max_range_term[1]] = 0

                split_bool_1[max_range_term[2]][max_range_term[1][1],max_range_term[1][1]] = 1
                split_bool_2[max_range_term[2]][max_range_term[1][1],max_range_term[1][1]] = -1

                C_split.append(C_1)
                C_split.append(C_2)
                split_bool_list.append(split_bool_1)
                split_bool_list.append(split_bool_2)
                name_list.append(name*2+1)
                name_list.append(name*2+2)

        return C_split, split_bool_list, name_list