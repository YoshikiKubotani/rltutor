import copy
import numpy as np
from collections import defaultdict

import utils


class DAS3HStudent:
    def __init__(self, time_weight, n_items, n_skills, seed):
        np.random.seed(seed)
        self.alpha = np.random.normal(loc=-1.5, scale=0.3, size=1)
        self.delta = np.random.normal(loc=-1.0, scale=0.5, size=n_items)
        self.beta = np.random.normal(loc=-1.0, scale=0.5, size=n_skills)
        self.time_weight = time_weight
        self.weight = np.hstack((self.delta, self.beta, self.time_weight))
        self.h = 0.3
        self.d = 0.8

    def predict_proba(self, input_sparse_vec, lag):
        for_sigmoid = self.alpha + np.dot(self.weight, input_sparse_vec)

        ret = 1 / (1 + np.exp(-for_sigmoid)[0])
        ret = (1 - ret) * (1 + self.h * lag) ** (-self.d) + ret

        return ret


class StudentModel(object):
    def __init__(
        self, n_items, n_skills, n_wins, seed, item_skill_mat, model
    ):
        self.name = "DAS3H"
        np.random.seed(seed)
        self.n_items = n_items
        self.n_skills = n_skills
        self.n_wins = n_wins
        self.predictor = model
        self.item_skill_mat = item_skill_mat

        self.n_item_feats = int(np.log(2 * self.n_items))

        self.item_feats = np.random.normal(
            np.zeros(2 * self.n_items * self.n_item_feats),
            np.ones(2 * self.n_items * self.n_item_feats),
        ).reshape((2 * self.n_items, self.n_item_feats))

        self.now = 0
        self.last_time = defaultdict(lambda: -10)

        self.curr_item = np.random.randint(self.n_items)
        self.q = defaultdict(lambda: utils.OurQueue())
        self.curr_outcome = 0
        self.curr_delay = 0
        self.skill_ids = None

    def _make_input_vec(self, selected_item_id, now_q):
        item_vec = np.zeros(self.n_items)
        skill_vec = np.zeros(self.n_skills)
        correct_vec = np.zeros(self.n_wins * self.n_skills)
        attempt_vec = np.zeros(self.n_wins * self.n_skills)

        item_vec[selected_item_id] = 1

        index_of_selected_skills = np.argwhere(
            self.item_skill_mat[selected_item_id] == 1
        )
        self.skill_ids = index_of_selected_skills.transpose()[0].tolist()
        self.skill_ids = list(set(self.skill_ids))
        skill_vec[self.skill_ids] = 1

        for skill_id in self.skill_ids:
            correct_vec[skill_id * self.n_wins : (skill_id + 1) * self.n_wins] = np.log(
                1 + np.array(now_q[skill_id, "correct"].get_counters(self.now))
            )
            attempt_vec[skill_id * self.n_wins : (skill_id + 1) * self.n_wins] = np.log(
                1 + np.array(now_q[skill_id].get_counters(self.now))
            )

        return_np_vec = np.hstack((item_vec, skill_vec, correct_vec, attempt_vec))
        return return_np_vec

    def _encode_delay(self):
        v = np.zeros(2)
        v[self.curr_outcome] = np.log(1 + self.curr_delay)
        return v

    def _encode_delay2(self):
        v = np.zeros(2)
        delay = self.curr_delay
        if len(self.q.queue) != 0:
            delay = self.now - self.q.queue[-1]
        v[self.curr_outcome] = np.log(1 + delay)
        return v

    def _vectorized_obs(self):
        encoded_item = self.item_feats[
            self.n_items * self.curr_outcome + self.curr_item, :
        ]
        return np.hstack(
            (encoded_item, self._encode_delay(), np.array([self.curr_outcome]))
        )

    def step(self, action, now):
        self.curr_item = action
        self.curr_delay = now - self.now
        self.now += self.curr_delay
        input_vec = self._make_input_vec(self.curr_item, copy.deepcopy(self.q))
        lag = self.now - self.last_time[self.curr_item]
        recall_prob = self.predictor.predict_proba(input_vec, lag)
        self.curr_outcome = 1 if np.random.random() < recall_prob else 0
        self._update_model()

        obs = self._vectorized_obs()
        return self.curr_outcome, obs

    def _update_model(self):
        self.last_time[self.curr_item] = self.now
        for skill_id in self.skill_ids:
            _ = self.q[skill_id, "correct"].get_counters(self.now)
            _ = self.q[skill_id].get_counters(self.now)
            if self.curr_outcome == 1:
                self.q[skill_id, "correct"].push(self.now)
            self.q[skill_id].push(self.now)

    def get_retention_rate(self):
        retention_rate_list = []
        curr_q = copy.deepcopy(self.q)
        for item in range(self.n_items):
            input_vec = self._make_input_vec(item, curr_q)
            lag = self.now - self.last_time[item]
            recall_prob = self.predictor.predict_proba(input_vec, lag)
            retention_rate_list.append(recall_prob)
        return retention_rate_list

    def reset(self, seed):
        np.random.seed(seed)
        self.now = 0
        self.last_time = defaultdict(lambda: -10)
        self.curr_item = np.random.randint(self.n_items)
        self.q = defaultdict(lambda: utils.OurQueue())
        self.curr_outcome = 0
        self.curr_delay = 0
        self.skill_ids = None