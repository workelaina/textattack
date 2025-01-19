"""
Search via Bayesian Optimization
===============

"""
from collections import defaultdict
import numpy as np
import torch
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
import random
import time
from copy import deepcopy
import gc

import os
import sys
# sys.path.append('../')
from algorithms import BlockBayesAttack
import time
from textattack.shared.utils import read_pkl


def get_query_budget(x, word_substitution_cache, baseline='textfooler'):
    '''
        input
            x : attacked_text
            syndict : synonym dictionary
            baseline : one of ['textfooler','pso','pwws','bae']
    '''
    assert baseline in ['textfooler', 'pso', 'pwws', 'bae'], "baseline should be one of ['textfooler','pso','pwws','bae']"
    if baseline == 'pso':
        query_budget = float('inf')
    else:
        query_budget_count = []
        for ind in range(len(x.words)):
            candids = word_substitution_cache[ind]
            query_budget_count.append(len(candids))
        if baseline in ['textfooler', 'bae']:
            # queries for importance calculation
            query_budget = len(query_budget_count)
            # max queries for search
            query_budget += sum([qc - 1 for qc in query_budget_count])
        elif baseline == 'pwws':
            # queries for importance calculation
            query_budget = sum([qc - 1 for qc in query_budget_count])
            # max queries for search
            query_budget += sum([qc - 1 for qc in query_budget_count])
    print("max query budget : ", query_budget)
    return query_budget


class BlackBoxModel():
    def __init__(self, model, goal_function, transformer):
        self.model = model
        self.goal_function = goal_function
        self.transformer = transformer
        self.initialize_num_queries()
        self.eval_cache = dict()
        self.query_budget = float('inf')

    def initialize_num_queries(self):
        self.num_queries = self.goal_function.num_queries

    def set_query_budget(self, query_budget):
        self.query_budget = query_budget

    def set_x(self, x0):
        self.x0 = x0
        self.len_seq = len(self.x0.words)
        self.word_substitution_cache = [[] for _ in range(self.len_seq)]
        for ind in range(self.len_seq):
            transformed_texts = self.transformer(
                self.x0, original_text=self.x0, indices_to_modify=[ind]
            )
            self.word_substitution_cache[ind].append(self.x0.words[ind])
            for txt in transformed_texts:
                if txt.words[ind] in self.word_substitution_cache[ind]:
                    continue
                else:
                    self.word_substitution_cache[ind].append(txt.words[ind])
        self.n_vertices = [
            len(w_candids) for w_candids in self.word_substitution_cache
        ]
        self.target_indices = [
            ind for ind in range(self.len_seq) if self.n_vertices[ind] > 1
        ]

    def seq2input(self, seq):
        assert type(seq) == torch.Tensor, f"type(seq) is {type(seq)}"
        if len(seq.shape) == 1:
            assert seq.shape[0] == self.len_seq, "indices length should be one of target indices length or seq length"
            seq_ = seq
        elif len(seq.shape) == 2:
            assert seq.shape[0] == 1 and seq.shape[1] == self.len_seq, "indices length should be one of target indices length or seq length"
            seq_ = seq.view(-1)

        cur_words = self.x0.words[:]
        if len(seq_) == self.len_seq:
            for ct, ind in enumerate(seq_):
                if ind > 0 and ct in self.target_indices:
                    cur_words[ct] = self.word_substitution_cache[ct][int(ind)]
        return self.x0.generate_new_attacked_text(cur_words)

    def get_initial_block_order(self, inds_list):
        leave_block_texts = []
        for inds in inds_list:
            start, end = self.target_indices[inds[0]], self.target_indices[inds[-1]]
            del_text = deepcopy(self.x0)
            for i in range(start, end + 1):
                if i == start:
                    assert del_text.words[i] == self.x0.words[start], "?"
                elif i == end:
                    assert del_text.words[start] == self.x0.words[end], "??"
                del_text = del_text.delete_word_at_index(start)
            leave_block_texts.append(
                del_text
            )
        leave_block_results, _ = self.model(leave_block_texts)
        init_result, _ = self.model([self.x0])
        init_score = init_result[0].score
        index_scores = np.array([abs(result.score - init_score) for result in leave_block_results])
        index_order = (-index_scores).argsort()
        return index_order

    def seq2str(self, seq):
        assert seq.type() == 'torch.LongTensor', f"{seq.type()} should be 'torch.LongTensor'"
        seq_ = seq.view(-1).cpu().detach()
        str_ = ''
        for i in seq_:
            str_ += f'{int(i)},'
        return str_

    def get_score(self, x, require_transform=True):
        if require_transform:
            x_ = self.seq2input(x.view(-1))
        else:
            x_ = x
        if self.num_queries >= self.query_budget:
            return None
        result = self.model([x_])[0][0]
        score = result.score
        self.num_queries = self.goal_function.num_queries  # update num queries
        return score

    def get_scores(self, xs, require_transform=True):
        if require_transform:
            xs_ = []
            for x in xs:
                x_ = self.seq2input(x.view(-1))
                xs_.append(x_)
        else:
            xs_ = xs
        if self.num_queries >= self.query_budget:
            return [None for _ in range(len(xs_))]
        results = self.model(xs_)[0]
        self.num_queries = self.goal_function.num_queries  # update num queries
        scores = self.get_scores_from_results(results, xs_)
        return scores

    def get_scores_from_results(self, results, inputs):
        if len(results) == len(inputs):
            return [result.score for result in results]
        else:
            ct = 0
            num = 0
            scores = [None for _ in range(len(inputs))]
            for result in results:
                while ct < len(inputs):
                    if result.attacked_text.text == inputs[ct].text:
                        scores[ct] = result.score
                        num += 1
                        break
                    else:
                        ct += 1
            assert num == len(results), f"num : {num}, len(results) : {len(results)}"
            assert self.num_queries == self.query_budget, "{self.num_queries} < {self.query_budget}, something wrong!"
            return scores


class DiscreteBlockBayesAttack(SearchMethod):
    """An attack based on Bayesian Optimization

    Args:
        dpp_type : dpp type. one of ['no','dpp_posterior']
    """

    def __init__(
        self,
        block_size=1024,
        batch_size=1,
        update_step=1,
        max_patience=50,
        post_opt='',
        use_sod=False,
        dpp_type='no',
        max_loop=5,
        fit_iter=1,
        max_budget_key_type='',
        ely_query_budget=None
    ):

        self.block_size = block_size
        self.batch_size = batch_size
        self.update_step = update_step
        self.max_patience = max_patience
        self.post_opt = post_opt
        self.use_sod = use_sod
        self.dpp_type = dpp_type
        self.max_loop = max_loop
        self.fit_iter = fit_iter
        self.max_budget_key_type = max_budget_key_type
        self.ely_query_budget = ely_query_budget

        self.memory_count = 0

    def perform_search(self, initial_result):
        init_time = time.time()
        BBM = BlackBoxModel(
            model=self.get_goal_results,
            goal_function=self.goal_function,
            transformer=self.get_transformations
        )
        BBM.initialize_num_queries()
        print(
            "initial query in perform_search func:",
            self.goal_function.num_queries,
            BBM.num_queries
        )
        x0 = initial_result.attacked_text
        BBM.set_x(x0)
        n_vertices = BBM.n_vertices

        if self.ely_query_budget is None:
            if self.max_budget_key_type == 'lsh':
                query_budget = read_pkl(
                    self.max_budget_path
                )[self.example_index]
            else:
                query_budget = get_query_budget(
                    x0,
                    BBM.word_substitution_cache,
                    baseline=self.max_budget_key_type
                )
        else:
            query_budget = self.ely_query_budget

        if query_budget <= 1:
            att_result = initial_result
            attack_logs = None
        else:
            BBM.set_query_budget(query_budget)
            self.goal_function.query_budget = query_budget
            attacker = BlockBayesAttack(
                block_size=self.block_size,
                batch_size=self.batch_size,
                update_step=self.update_step,
                max_patience=self.max_patience,
                post_opt=self.post_opt,
                use_sod=self.use_sod,
                dpp_type=self.dpp_type,
                max_loop=self.max_loop,
                fit_iter=self.fit_iter
            )

            attacker_input = torch.zeros(1, len(x0.words))
            x_att, attack_logs = attacker.perform_search(
                attacker_input,
                n_vertices,
                BBM
            )
            x_att_transformed = BBM.seq2input(x_att)

            att_result = self.get_goal_results([x_att_transformed])[0][0]
            assert BBM.num_queries == self.goal_function.num_queries, f"something wrong! {BBM.num_queries} != {self.goal_function.num_queries}"

        elapsed_time = time.time() - init_time
        setattr(att_result, 'elapsed_time', elapsed_time)
        setattr(att_result, 'attack_logs', attack_logs)
        setattr(att_result, 'query_budget', query_budget)
        return att_result

    @property
    def is_black_box(self):
        return True

    _perform_search = perform_search
