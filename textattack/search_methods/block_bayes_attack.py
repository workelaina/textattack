"""
Search via Bayesian Optimization
===============

"""
from collections import defaultdict
import numpy as np
import torch
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
import random

from textattack.BayesOpt.acquisition.algorithm import kmeans_pp, refine_inputs, acquisition_maximization_with_indices, acquisition_maximization_with_indices_v2
from textattack.BayesOpt.historyboard import HistoryBoardFixed
from textattack.BayesOpt.surrogate_model.gp_model import MyGPModel

import copy
import time
from copy import deepcopy
import gc

use = UniversalSentenceEncoder()

def list_by_ind(l, ind):
    return [l[i] for i in ind]

class BayesAttack(SearchMethod):
    """An attack based on Bayesian Optimization

    Args:
        kernel_name : kernel name. one of ['linear_reg', 'inner', 'matern', 'discrete_inner', 'discrete_matern', 'discrete_categorical', 'discrete_diffusion']
        dpp_type : dpp type. one of ['no','dpp_posterior]
    """

    def __init__(self, kernel_name='categorical', block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, block_policy='straight', dpp_type='dpp_posterior', max_loop=5, niter=20):
        
        self.kernel_name = kernel_name
        self.block_size = block_size
        self.batch_size = batch_size
        self.update_step = update_step
        self.max_patience = max_patience
        self.post_opt = post_opt
        self.use_sod = use_sod
        self.block_policy = block_policy
        self.dpp_type = dpp_type
        self.max_loop = max_loop
        self.batch_size = batch_size
        self.niter = niter

        self.reg_coef = 0.0
        self.memory_count = 0
        self.one_at_once = False
        self.no_terminal = False
    
    def check_query_const(self):
        if self.goal_function.num_queries == self.goal_function.query_budget:
            return True
        else:
            return False

    def init_before_loop(self):
        # Bayesian Optimization Loop
        # set best evaluated text as initial text.
        best_ind = self.hb.best_in_history()[3][0]
        initial_text = self.hb.eval_texts[best_ind]
        eff_len = len(self.hb.target_indices)
        
        D_0 = []
        index_order = self.get_index_order_for_block_decomposition(self.block_policy)

        self.NB_INIT = int(np.ceil(eff_len / self.block_size))
        self.INDEX_DICT = defaultdict(list)
        self.HISTORY_DICT = defaultdict(list)
        self.BLOCK_QUEUE = [(0,int(i)) for i in range(self.NB_INIT)]

        center_text = initial_text
        center_ind = 0

        # init INDEX_DICT, HISTORY_DICT
        ALL_IND = index_order
        for KEY in self.BLOCK_QUEUE:
            self.INDEX_DICT[KEY] = deepcopy(ALL_IND[self.block_size*KEY[1]:self.block_size*(KEY[1]+1)])

        LOCAL_OPTIMUM = defaultdict(list)
        stage = -1

        return eff_len, D_0, center_text, center_ind, LOCAL_OPTIMUM, stage

    def perform_search(self, initial_result):
        print("query budget is ", self.goal_function.query_budget)
        self.time_cache = {}
        self.time_cache['init'] = time.time()
        self.orig_text = initial_result.attacked_text

        self.hb = HistoryBoardFixed(orig_text=self.orig_text, transformer = self.get_transformations, enc_model=use)

        # set greedy(del) query constraint (For BAE, TF query const)
        if 'gc' in self.post_opt:
            self.goal_function.query_budget = sum(self.hb.n_vertices)

        #print("query budget is ", self.goal_function.query_budget)
    
        if self.check_query_const() or len(self.hb.target_indices)==0:
            setattr(initial_result,'info_',[-1,-1,-1,'self',self.hb.reduced_n_vertices,self.hb.n_vertices,self.time_cache,self.hb.eval_Y,self.hb.hamming_with_orig])
            return initial_result

        self.hb.add_datum(self.orig_text, initial_result)
        #print("orig score : ",self.hb.eval_Y[0])
        
        # Initialize surrogate model wrapper.
        self.surrogate_model = MyGPModel(self.kernel_name, niter=self.niter)

        eff_len, D_0, center_text, center_ind, LOCAL_OPTIMUM, stage = self.init_before_loop()

        while self.BLOCK_QUEUE:
            self.clean_memory_cache()
            if self.BLOCK_QUEUE[0][0] != stage:
                self.BLOCK_QUEUE = self.update_queue(self.BLOCK_QUEUE, self.INDEX_DICT)
                stage += 1
            if not self.BLOCK_QUEUE: break

            KEY = self.BLOCK_QUEUE.pop(0)

            opt_indices = deepcopy(self.INDEX_DICT[KEY])
            fix_indices = list( set(list(range(eff_len))) - set(opt_indices) )

            self.HISTORY_DICT[KEY].append(int(center_ind))

            if not opt_indices: continue

            stage_init_ind = len(self.hb.eval_Y)
            stage_iter = sum([self.hb.reduced_n_vertices[ind]-1 for ind in opt_indices]) 
            ex_ball_size = 10000
            n_samples = int(stage_iter / len(opt_indices)) if len(opt_indices)<=3 else int(stage_iter / len(opt_indices)) * 2
            next_len = self.block_size                   

            #print("KEY : ", KEY, 'in', self.BLOCK_QUEUE)
            #print("opt_indices : ", opt_indices)
            #print("cur best")
            #print(self.hb.best_in_history()[1][0][0].item(), self.hb.best_in_history()[2][0])

            # Exploration.
            prev_qr = len(self.hb.eval_Y)
            stage_call = 0 
            if self.check_query_const(): break

            time0 = time.time()
            stage_call, fresult, result = self.exploration_ball_with_indices(center_text=center_text,n_samples=n_samples,ball_size=ex_ball_size,stage_call=stage_call, opt_indices=opt_indices, KEY=KEY, stage_init_ind=stage_init_ind)

            time1 = time.time()

            if stage_call == -1:
                setattr(result,'info_',[KEY,self.INDEX_DICT,self.block_size,fresult,self.hb.reduced_n_vertices,self.hb.n_vertices,self.time_cache,self.hb.eval_Y,self.hb.hamming_with_orig])
                return result
            if len(self.hb.eval_Y) == prev_qr:
                if KEY[0] < self.max_loop: 
                    new = (KEY[0]+1, KEY[1])
                    self.BLOCK_QUEUE.append(new)
                    self.INDEX_DICT[new] = deepcopy(opt_indices[:next_len])
                    self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_texts))])
                continue
            if self.check_query_const(): break

            time2 = time.time()

            # union parent histories and prev best ind (center ind)
            parent_history = set(D_0)
            for n in range(KEY[0]):
                key = (int(n),KEY[1])
                parent_history |= set(self.HISTORY_DICT[key])

                inds = self.HISTORY_DICT[key]
                loc_fix_indices = list( set(list(range(eff_len))) - set(deepcopy(self.INDEX_DICT[key])) )
                if loc_fix_indices:
                    history = self.hb.eval_X_num[inds]
                    uniq = torch.unique(history[:,loc_fix_indices],dim=0)
                    assert uniq.shape[0] == 1, f'{uniq.shape},{uniq[:,:5]}'
                    print(uniq[:,:5], fix_indices)

            parent_history.add(center_ind)
            parent_history = list(parent_history)

            if self.use_sod:
                parent_history = self.subset_of_dataset(parent_history, stage_iter)
                assert len(parent_history) <= stage_iter, f'something wrong {stage_iter}, {len(parent_history)}'
            # Exploitation.
            num_candids = 10
            
            init_cent_indiced = deepcopy(self.hb.numbering_text(center_text))

            time3 = time.time()

            print("before loop")
            print(time1-time0, time2-time1, time3-time2)
            count = 0

            prev_size = len(self.hb.eval_Y)
            iter_patience = 5
            while stage_call < stage_iter and iter_patience:
                self.clean_memory_cache()
                t0 = time.time()
                #print(self.goal_function.num_queries)
                if prev_size == len(self.hb.eval_Y):
                    iter_patience -= 1
                else:
                    iter_patience = 10
                    prev_size = len(self.hb.eval_Y)
                self.surrogate_model.fit_partial(self.hb, list(range(eff_len)), stage_init_ind, prev_indices=parent_history) 
                t1 = time.time()

                if count  % self.update_step == 0:
                    ttt0 = time.time()
                    # TODO 0.006
                    best_inds = self.hb.topk_in_history_with_fixed_indices(len(self.hb.eval_Y), init_cent_indiced, fix_indices)[3]          
                    ttt1 = time.time()
                    for best_ind in best_inds:
                        if not (best_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                            break

                    best_val = self.hb.eval_Y[best_inds[0]][0].item()
                    best_text = [self.hb.eval_texts[best_ind]]
                    reference = best_val - self.reg_coef * torch.count_nonzero(self.hb.eval_X_num[best_inds[0]])
                    #print(f"best of current KEY {KEY}, {count} :", best_val, self.hb.eval_Y[best_ind][0].item(), self.hb._hamming(self.orig_text, best_text[0]), best_ind, LOCAL_OPTIMUM[tuple(opt_indices)])
                    best_indiced = self.hb.numbering_texts(best_text)
                    ttt2 = time.time()
                    
                    #best_texts = self.find_greedy_init_with_indices(cand_indices=best_indiced, max_radius=eff_len, num_candids=num_candids, reference=reference)
                    best_texts = self.find_greedy_init_with_indices_v2(cand_indices=best_indiced, max_radius=eff_len, num_candids=num_candids, reference=reference)
                    ttt3 = time.time()
                    #print("    update    ",ttt1-ttt0,ttt2-ttt1,ttt3-ttt2)
                t12 = time.time()
                # TODO 0.05
                best_candidates = acquisition_maximization_with_indices_v2(best_texts, opt_indices=opt_indices, batch_size=self.batch_size, stage=eff_len, hb=self.hb, surrogate_model=self.surrogate_model, kernel_name=self.kernel_name, reference=reference, dpp_type=self.dpp_type, acq_with_opt_indices=False, reg_coef=self.reg_coef)

                t2 = time.time()
                if type(best_candidates) == type(None):
                    LOCAL_OPTIMUM[tuple(opt_indices)].append(best_ind)
                    rand_indices = self.hb.nbd_sampler(best_indiced, self.batch_size, 2, 1, fix_indices=fix_indices)
                    best_candidates = [self.hb.text_by_indices_v2(inds) for inds in random.sample(list(rand_indices), self.batch_size)]                        

                if stage_call >= stage_iter or self.check_query_const(): break
                
                t23 = time.time()

                if self.one_at_once:
                    for best_candidate in best_candidates:
                        if self.eval_and_add_datum(best_candidate):
                            fresult = self.hb.eval_results[-1]
                            self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_texts))])
                            result = self.final_exploitation(best_candidate, len(self.hb.eval_Y)-1)
                            setattr(result,'info_',[KEY,self.INDEX_DICT,self.block_size,fresult,self.hb.reduced_n_vertices,self.hb.n_vertices,self.time_cache,self.hb.eval_Y,self.hb.hamming_with_orig])
                            return result
                        stage_call += 1
                        if stage_call >= stage_iter or self.check_query_const(): break
                else:
                    prev_len = len(self.hb.eval_Y) 
                    if self.eval_and_add_data(best_candidates):
                        fresult = self.hb.eval_results[-1]
                        self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_texts))])
                        result = self.final_exploitation(self.hb.eval_texts[-1], len(self.hb.eval_Y)-1)
                        setattr(result,'info_',[KEY,self.INDEX_DICT,self.block_size,fresult,self.hb.reduced_n_vertices,self.hb.n_vertices,self.time_cache,self.hb.eval_Y,self.hb.hamming_with_orig])
                        return result
                    stage_call += len(self.hb.eval_Y) - prev_len
                    if stage_call >= stage_iter or self.check_query_const(): break
                t3 = time.time()
                #print(t1-t0, t12-t1, t2-t12, t23-t2, t3-t23)
                count += 1

            best_inds = self.hb.topk_in_history(len(self.hb.eval_Y))[3] 
            
            for center_ind in best_inds:
                if not (center_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                    center_ind = int(center_ind)
                    break
            center_text = self.hb.eval_texts[center_ind]
            if self.check_query_const(): break

            if KEY[0] < self.max_loop: 
                print("280 line.")
                print(KEY[0], self.max_loop)
                new = (KEY[0]+1, KEY[1])
                self.BLOCK_QUEUE.append(new)
                self.INDEX_DICT[new] = deepcopy(opt_indices[:next_len])
                self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_texts))])
            print(f"best of KEY {KEY} : ", self.hb.best_in_recent_history(len(self.hb.eval_texts)-stage_init_ind)[1][0][0].item())
            continue
        
        if (self.goal_function.num_queries < self.goal_function.query_budget) and (not 'nogreed' in self.post_opt) and (not self.no_terminal):
            # Greedy step.
            print("greedy step !!!")
            best_ind = self.hb.topk_in_history(1)[3][0]
            best_result = self.hb.eval_results[best_ind]
            best_score = self.hb.eval_Y[best_ind][0].item()
            self.fit_surrogate_model_by_block_history()
            index_order = self.get_index_order_for_block_decomposition('beta')
            print("before greedy step!!")
            for i in range(10):
                if i>0:
                    orig_indiced = self.hb.numbering_text(self.orig_text)
                    rand_indices = self.hb.nbd_sampler(orig_indiced, 1, eff_len, 1, fix_indices=[])
                    rand_texts = [self.hb.text_by_indices_v2(indices) for indices in rand_indices]
                    rand_results, search_over = self.get_goal_results(rand_texts)
                    if rand_results:
                        rand_results = sorted(rand_results, key=lambda x: -x.score)
                        if rand_results[0].score >= 0: return rand_results[0]
                        center_text = rand_results[0].attacked_text
                        search_over = False
                    else:
                        search_over = True
                else:
                    center_text = self.hb.eval_texts[best_ind]
                    search_over = False
                if not search_over:
                    best_result, is_nice = self.greedy_step(center_text, index_order)
                    print('best score in final greedy loop', best_result.score)
                    if is_nice:
                        print("veryveryverynice!!!!!")
                        setattr(best_result,'greedy',True)
                        return best_result
                else:
                    break
            print("after greedy step!!")
            
        if self.goal_function.query_budget < float('inf'):
            results = self.hb.eval_results[:self.goal_function.query_budget]
        else:
            results = self.hb.eval_results
        results = sorted(results, key=lambda x: -x.score)
        result = results[0]
        setattr(result,'info_',[KEY,self.INDEX_DICT,self.block_size,'self',self.hb.reduced_n_vertices,self.hb.n_vertices,self.time_cache,self.hb.eval_Y,self.hb.hamming_with_orig])
        return result
    
    def subset_of_dataset(self, history, num_samples):
        if len(history) <= num_samples:
            return history
        else:
            print(self.hb.eval_X_num)
            print(history)
            history_X_num = self.hb.eval_X_num[history].numpy()
            _, selected_indices_ = kmeans_pp(history_X_num, num_samples, dist='hamming')
            history = [history[ind] for ind in selected_indices_]
            return history

    def greedy_step(self, text, index_order, is_shuffle=True):
        best_text = text
        best_indiced = self.hb.numbering_text(text)
        best_result = self.get_goal_results([best_text])[0][0]
        best_score = best_result.score

        order = deepcopy(index_order)
        print('initial_score ', best_score)
        while True:
            prev_best_text = best_text
            if is_shuffle:
                random.shuffle(order)
            for ind in order:
                nv = self.hb.reduced_n_vertices[ind]
                candids = [deepcopy(best_indiced) for i in range(nv)] 
                for i, cand in enumerate(candids):
                    cand[ind] = i 
                candid_texts = [self.hb.text_by_indices_v2(cand) for cand in candids]
                candid_results, search_over = self.get_goal_results(candid_texts)
                if search_over and not candid_results: break
                candid_results = sorted(candid_results, key=lambda x: -x.score)
                if candid_results[0].score >= 0: return candid_results[0], True
                if best_score < candid_results[0].score:
                    best_text = candid_results[0].attacked_text
                    best_indiced = self.hb.numbering_text(best_text)
                    best_result = candid_results[0]
                    best_score = candid_results[0].score
                print(ind, best_score)
            if prev_best_text.words == best_text.words:
                break 
        return best_result, False

    def ind_score(self, ind, beta, stage):
        score = sum([float(beta[i]) for i in ind]) + 1e6 * stage
        return score

    def update_queue(self, Q, I):
        if Q[0][0] == 0:
            print("first stage.")
            if len(Q)==1:
                return Q
            else:
                order = self.get_block_order(Q,I)
                Q_ = [Q[ind] for ind in order]
                return Q_
        if len(Q)==1:
            return Q
        self.fit_surrogate_model_by_block_history()

        if self.kernel_name == 'categorical':
            beta = 1/(self.surrogate_model.model.covar_module.base_kernel.lengthscale[0].detach().cpu()+1e-6)
        elif self.kernel_name == 'categorical_horseshoe':
            beta = self.surrogate_model.model.covar_module.base_kernel.lengthscale[0].detach().cpu()
        for KEY in Q:
            if not I[KEY]:
                Q.remove(KEY)

        def f(KEY):
            return self.ind_score(I[KEY],beta,KEY[0])
        Q_ = sorted(Q,key=f)
        print(Q)
        print('->')
        print(Q_)
        return Q_

    def get_block_order(self, Q, I):
        leave_block_texts = []
        for KEY in Q:
            inds = I[KEY]
            start, end = self.hb.target_indices[inds[0]], self.hb.target_indices[inds[-1]]
            del_text = deepcopy(self.orig_text)
            for i in range(start,end+1):
                if i == start:
                    assert del_text.words[i] == self.orig_text.words[start], "?"
                elif i == end:
                    assert del_text.words[start] == self.orig_text.words[end], "??"
                del_text = del_text.delete_word_at_index(start)
            leave_block_texts.append(
                del_text
            )
        leave_block_results, search_over = self.get_goal_results(leave_block_texts)
        index_scores = np.array([abs(result.score-self.hb.eval_Y[0].item()) for result in leave_block_results])
        index_order = (-index_scores).argsort()
        return index_order

    def get_index_order_for_block_decomposition(self, block_policy):
        if 'rand' == block_policy:
            index_order = list(range(len(self.hb.target_indices)))
            random.shuffle(index_order)
        elif 'straight' in block_policy:
            index_order = list(range(len(self.hb.target_indices)))
        elif 'beta' == block_policy:
            if self.kernel_name == 'categorical':
                beta = 1/(self.surrogate_model.model.covar_module.base_kernel.lengthscale.detach().cpu()+1e-6)
            elif self.kernel_name == 'categorical_horseshoe':
                beta = self.surrogate_model.model.covar_module.base_kernel.lengthscale.detach().cpu()
            index_order = torch.argsort(beta)[0]
        index_order = [int(i) for i in index_order]
        return index_order

    def exploration_ball_with_indices(self, center_text, n_samples, ball_size, stage_call, opt_indices, KEY, stage_init_ind):
        if n_samples == 0:
            print(1)
            return stage_call, None, None
        fix_indices = list(set(list(range(len(self.hb.target_indices)))) - set(opt_indices))
        prev_len = self.hb.eval_Y.shape[0]
        rand_candidates = self.hb.sample_ball_candidates_from_text(center_text, n_samples=n_samples, ball_size=ball_size, fix_indices=fix_indices)

        if self.eval_and_add_data(rand_candidates):
            fresult = self.hb.eval_results[-1]
            best_candidate = self.hb.eval_texts[-1]
            return -1, fresult, self.final_exploitation(best_candidate, len(self.hb.eval_Y)-1)

        if self.check_query_const():
            stage_call += self.hb.eval_Y.shape[0] - prev_len
            return stage_call,None, None

        # If any query were not evaluated, hardly sample non orig examples.
        if self.hb.eval_Y.shape[0] == prev_len:
            center_indiced = self.hb.numbering_text(center_text)
            rand_indiced = copy.deepcopy(center_indiced)
            for ind in opt_indices:
                rand_indiced[ind] = int(random.sample(list(range(self.hb.reduced_n_vertices[ind]-1)), 1)[0] + 1)
            rand_candidate = self.hb.text_by_indices_v2(rand_indiced)
            if self.eval_and_add_datum(rand_candidate):
                self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_texts))])
                return -1, self.final_exploitation(rand_candidate, len(self.hb.eval_Y)-1)
        stage_call += self.hb.eval_Y.shape[0] - prev_len
        return stage_call, None, None

    def find_greedy_init_with_indices(self, cand_indices, max_radius, num_candids, reference=None):
        ### Before Greedy Ascent Step ###
        # calculate acquisition
        import time
        ttt0 = time.time()
        if reference is None:
            _, reference, best_ind = self.hb.best_of_hamming_orig(distance=max_radius)
            reference = reference - self.reg_coef *  torch.count_nonzero(self.hb.eval_X_num[best_ind])
        ttt1 = time.time()
        ei = self.surrogate_model.acquisition(cand_indices, bias=reference,reg_coef=self.reg_coef)
        ttt2 = time.time()
        topk_values, topk_indices = torch.topk(ei, min(len(ei),num_candids))
        ttt3 = time.time()
        center_candidates = [self.hb.text_by_indices_v2(cand_indices[idx]) for idx in topk_indices]
        ttt4 = time.time()
        print("greedy init      ",cand_indices.shape, len(ei),ttt1-ttt0,ttt2-ttt1,ttt3-ttt2,ttt4-ttt3)
        return center_candidates

    # CERT
    def find_greedy_init_with_indices_v2(self, cand_indices, max_radius, num_candids, reference=None):
        ### Before Greedy Ascent Step ###
        # calculate acquisition
        if reference is None:
            _, reference, best_ind = self.hb.best_of_hamming_orig(distance=max_radius)
            reference = reference - self.reg_coef *  torch.count_nonzero(self.hb.eval_X_num[best_ind])
        ei = self.surrogate_model.acquisition(cand_indices, bias=reference,reg_coef=self.reg_coef)
        topk_values, topk_indices = torch.topk(ei, min(len(ei),num_candids))
        center_indices_list = [cand_indices[idx].view(1,-1) for idx in topk_indices]
        return center_indices_list

    def eval_and_add_datum(self, text, return_=True):
        if not text.text in self.hb.eval_texts_str:
            result, _ = self.get_goal_results([text])
            self.time_cache[self.goal_function.num_queries] = time.time()
            self.hb.add_datum(text, result[0])
            if return_ and result[0].goal_status == GoalFunctionResultStatus.SUCCEEDED and not self.no_terminal:
                return 1
            else:
                return 0
        else:
            return 0

    def eval_and_add_data(self, texts, return_=True):
        results, _ = self.get_goal_results(texts)
        self.time_cache[self.goal_function.num_queries] = time.time()
        for text, result in zip(texts,results):
            if not text.text in self.hb.eval_texts_str:
                self.hb.add_datum(text, result)
            if return_ and result.goal_status == GoalFunctionResultStatus.SUCCEEDED and not self.no_terminal:
                return 1
        return 0

    def eval_and_add_data_best_ind(self, texts, cur_text, best_ind, tmp, tmp_modif, patience):
        results, _ = self.get_goal_results(texts)
        for text, result in zip(texts,results):
            if not text.text in self.hb.eval_texts_str:
                self.hb.add_datum(text, result)
                if result.goal_status == GoalFunctionResultStatus.SUCCEEDED and not self.no_terminal:
                    modif = self.hb._hamming(self.orig_text, text)
                    if tmp < self.hb.eval_Y[-1].item() or (tmp == self.hb.eval_Y[-1].item() and tmp_modif > modif):
                        tmp = self.hb.eval_Y[-1].item()
                        tmp_modif = modif
                        cur_text = text
                        best_ind = len(self.hb.eval_results) - 1 
            patience -= 1
        return cur_text, best_ind, patience

    def final_exploitation(self, text, ind):
        if 'v3' in self.post_opt:
            return self.final_exploitation_v3(text, ind)
        elif 'v2' in self.post_opt:
            return self.final_exploitation_v2(text, ind)[0]
        elif 'v1' in self.post_opt:
            ind = self.final_exploitation_v1(text, ind)[1]
            return self.hb.eval_results[ind]
        elif 'v4' in self.post_opt:
            return self.final_exploitation_v4(text, ind)
        else:
            return self.hb.eval_results[ind]

    def final_exploitation_v4(self, text, ind):
        init_ind = len(self.hb.eval_Y)
        v2_ind = self.final_exploitation_v2(text, ind)[1]
        v2_text = self.hb.eval_texts[v2_ind]

        forced_inds = [int(i) for i in range(init_ind, len(self.hb.eval_Y))]
        return self.final_exploitation_v3(v2_text, v2_ind, forced_inds)

    def final_exploitation_v1(self, text, ind):
        cur_text = text
        cur_ind = ind
        cur_score = self.hb.eval_Y[cur_ind].item()
        cur_indices = self.hb.numbering_text(cur_text)
        nonzero_indices = [ct for ct, ind in enumerate(cur_indices) if ind > 0]
        print("final exploitation", ind, len(nonzero_indices))

        if len(nonzero_indices)==1:
            print(0)
            cur_result = self.hb.eval_results[cur_ind]
            setattr(cur_result, 'expl_info',[None, None, None, cur_ind])
            return self.hb.eval_results[cur_ind], cur_ind

        imps = []
        indices = []
        scores = []
        results = []
        inds = []
        for ct, idx in enumerate(nonzero_indices):
            new_indices = copy.deepcopy(cur_indices)
            new_indices[idx] = 0
            new_text = self.hb.text_by_indices_v2(new_indices)
            if self.check_query_const(): break
            self.eval_and_add_datum(new_text)
            if new_text.text in self.hb.eval_texts_str:
                new_ind = self.hb.eval_texts_str.index(new_text.text)
                new_score = self.hb.eval_Y[new_ind].item()
                new_result = self.hb.eval_results[new_ind]   
            else:
                new_ind = len(self.hb.eval_texts_str) - 1
                new_score = self.hb.eval_Y[-1].item()  
                new_result = self.hb.eval_results[-1]   
            imps.append(cur_score - new_score)
            scores.append(new_score)
            inds.append(new_ind)
        order = np.argsort(imps)
        if len(order)==0:
            print(1)
            cur_result = self.hb.eval_results[cur_ind]
            setattr(cur_result, 'expl_info',[order, imps, scores, cur_ind])
            return cur_result, cur_ind
        first_idx = order[0]
        prev_score = scores[first_idx]
        prev_idx = inds[first_idx]
        if prev_score < 0:
            print(2)
            cur_result = self.hb.eval_results[cur_ind]
            setattr(cur_result, 'expl_info',[order, imps, scores, cur_ind])
            return self.hb.eval_results[cur_ind], cur_ind
        else:
            print(3)
            return self.final_exploitation_v1(self.hb.eval_texts[prev_idx], prev_idx)

    def final_exploitation_v2(self, text, ind):
        print("final exploitation")
        cur_text = text
        cur_ind = ind
        cur_score = self.hb.eval_Y[cur_ind].item()
        cur_indices = self.hb.numbering_text(cur_text)
        nonzero_indices = [ct for ct, ind in enumerate(cur_indices) if ind > 0]

        if len(nonzero_indices)==1:
            print(0)
            cur_result = self.hb.eval_results[cur_ind]
            setattr(cur_result, 'expl_info',[None, None, None, cur_ind])
            return self.hb.eval_results[cur_ind], cur_ind

        imps = []
        indices = []
        scores = []
        results = []
        for ct, idx in enumerate(nonzero_indices):
            new_indices = copy.deepcopy(cur_indices)
            new_indices[idx] = 0
            new_text = self.hb.text_by_indices_v2(new_indices)
            if self.check_query_const(): break
            self.eval_and_add_datum(new_text)
            if new_text.text in self.hb.eval_texts_str:
                new_ind = self.hb.eval_texts_str.index(new_text.text)
                new_score = self.hb.eval_Y[new_ind].item()
                new_result = self.hb.eval_results[new_ind]   
            else:
                new_score = self.hb.eval_Y[-1].item()  
                new_result = self.hb.eval_results[-1]   
            imps.append(cur_score - new_score)
            scores.append(new_score)
            results.append(new_result)
            indices.append(new_indices)
        order = np.argsort(imps)
        if len(order)==0:
            cur_result = self.hb.eval_results[cur_ind]
            setattr(cur_result, 'expl_info',[order, imps, scores, cur_ind])
            return cur_result, cur_ind
        first_idx = order[0]
        prev_indices = indices[first_idx]
        prev_score = scores[first_idx]
        prev_result = results[first_idx]
        prev_idx = cur_ind
        if prev_score < 0:
            cur_result = self.hb.eval_results[cur_ind]
            setattr(cur_result, 'expl_info',[order, imps, scores, cur_ind])
            return self.hb.eval_results[ind], ind
        for idx in order[1:]:
            if self.check_query_const(): break
            new_indices = copy.deepcopy(prev_indices)
            new_indices[nonzero_indices[idx]] = 0
            #print("final", idx, new_indices)
            new_text = self.hb.text_by_indices_v2(new_indices)
            if self.eval_and_add_datum(new_text):
                prev_indices = new_indices
                prev_score = self.hb.eval_Y[-1].item()  
                prev_result = self.hb.eval_results[-1]
                prev_idx = len(self.hb.eval_Y)-1
                continue
        setattr(prev_result, 'expl_info',[order, imps, scores, cur_ind])
        return prev_result, prev_idx
    

    def final_exploitation_v3(self, text, ind, forced_inds=[]):
        print("final exploitation")
        cur_text = text
        best_ind = ind
        eff_len = len(self.hb.target_indices)
        max_patience = self.max_patience
        prev_radius = self.hb._hamming(self.orig_text, cur_text)
        patience = max_patience

        i=0
        nbd_size = 2
        opt_indices = [idx for idx in range(eff_len)] 
        whole_indices = [idx for idx in range(eff_len)] 
        _, sum_history = self.fit_surrogate_model_by_block_history(forced_inds)
        init_idx = len(self.hb.eval_Y)

        expl_info = []
        while True:
            self.clean_memory_cache()
            max_radius = self.hb._hamming(self.orig_text, cur_text) - 1
            if prev_radius == max_radius:
                if patience <= 0:
                    break
            else:
                patience = max_patience
                prev_radius = max_radius
                nbd_size = 2
            expl_info.append([i, max_radius, patience, max_patience, self.goal_function.num_queries, self.goal_function.query_budget])
            print("final", i, max_radius)
            if max_radius == 0:
                return self.hb.eval_results[best_ind]
            
            self.surrogate_model.fit_partial(self.hb, whole_indices, init_idx, sum_history)
            best_candidate = cur_text

            # best in ball text
            bib_text, bib_score, _ = self.hb.best_of_hamming_orig(distance=max_radius)

            best_indiced = self.hb.numbering_text(best_candidate)
            bib_indiced = self.hb.numbering_text(bib_text)  
            orig_indiced = self.hb.numbering_text(self.orig_text)
            rand_indices = self.hb.subset_sampler(best_indiced, 300, nbd_size)

            cand_indices = torch.cat([orig_indiced.view(1,-1), best_indiced.view(1,-1), bib_indiced.view(1,-1), rand_indices], dim=0)
            cand_indices = torch.unique(cand_indices.long(),dim=0).float()
            center_candidates = self.find_greedy_init_with_indices_v2(cand_indices, max_radius, num_candids=self.batch_size, reference=0.0)
            t5 = time.time()
  
            reference = self.hb.eval_Y[best_ind].item() - (max_radius + 1) * self.reg_coef
            best_candidates = acquisition_maximization_with_indices_v2(center_candidates, opt_indices=opt_indices, batch_size=self.batch_size, stage=max_radius-1, hb=self.hb, surrogate_model=self.surrogate_model, kernel_name=self.kernel_name, reference=reference, dpp_type=self.dpp_type, acq_with_opt_indices=False, reg_coef=self.reg_coef)
            if best_candidates == None:
                if max_radius + 1 == nbd_size:
                    break
                else:
                    nbd_size += 1
                    continue

            tmp = 0.0
            tmp_modif = eff_len
            if self.check_query_const(): break

            if self.one_at_once:
                for best_candidate in best_candidates:
                    if self.eval_and_add_datum(best_candidate):
                        modif = self.hb._hamming(self.orig_text, best_candidate)
                        if tmp < self.hb.eval_Y[-1].item() or (tmp == self.hb.eval_Y[-1].item() and tmp_modif > modif):
                            tmp = self.hb.eval_Y[-1].item()
                            tmp_modif = modif
                            cur_text = best_candidate
                            best_ind = len(self.hb.eval_results) - 1 
                    patience -= 1
                    if self.check_query_const() or patience <= 0: break
            else:
                cur_text, best_ind, patience = self.eval_and_add_data_best_ind(best_candidates, cur_text, best_ind, tmp, tmp_modif, patience)
                if self.check_query_const() or patience <= 0: break

            if self.check_query_const() or patience <= 0: break
            i += 1
        result = self.hb.eval_results[best_ind]
        setattr(result, 'expl_info',expl_info)
        return result

    def block_history_dict(self, forced_inds=[]):
        bhl = defaultdict(list)
        print("func block_history_dict")
        print("self.history_dict")
        print(self.HISTORY_DICT)
        for KEY, INDEX in self.INDEX_DICT.items():
            HISTORY = self.HISTORY_DICT[KEY]
            bhl[KEY[1]].extend(HISTORY)
        for key in bhl:
            bhl[key] = list(dict.fromkeys(bhl[key]))
            opt_indices = list(range(key*self.block_size,min((key+1)*self.block_size,len(self.hb.reduced_n_vertices))))
            num_samples = sum([self.hb.reduced_n_vertices[ind]-1 for ind in opt_indices]) 
            bhl[key] = self.subset_of_dataset(bhl[key],num_samples)
            print(f"num samples in {key} : {len(bhl[key])}")
        sum_history = copy.deepcopy(forced_inds)
        for key, l in bhl.items():
            sum_history.extend(l)
        return bhl, sum_history
    
    def fit_surrogate_model_by_block_history(self, forced_inds=[]):
        eff_len = len(self.hb.target_indices)
        bhl, sum_history = self.block_history_dict(forced_inds)
        whole_indices = [idx for idx in range(eff_len)] # nonzero indices
        self.surrogate_model.fit_partial(self.hb, whole_indices, len(self.hb.eval_Y), sum_history)
        return bhl, sum_history

    def clean_memory_cache(self,debug=False):
        # Clear garbage cache for memory.
        if self.memory_count == 10:
            gc.collect()
            torch.cuda.empty_cache()
            self.memory_count = 0
        else:
            self.memory_count += 1   
        if debug:
            print(torch.cuda.memory_allocated(0))
    @property
    def is_black_box(self):
        return True

    _perform_search = perform_search
