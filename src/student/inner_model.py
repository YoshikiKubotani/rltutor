import sys
import copy
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logging import getLogger
from sklearn.preprocessing import OneHotEncoder
from gym import spaces
from collections import defaultdict
from torch.distributions.normal import Normal

import utils

logger = getLogger(__name__)

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data_np, lag_np, target_np):
        self.target = torch.tensor(target_np.astype(np.float32))
        self.lag = torch.tensor(lag_np.astype(np.float32))
        self.input = torch.tensor(data_np.astype(np.float32))
        self.len_data = self.input.shape[0]

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        out_data = self.input[idx]
        out_lag = self.lag[idx].view(-1, 1)
        out_label = self.target[idx].view(-1, 1)
        return out_data, out_lag, out_label


class DAS3HInner(torch.nn.Module):
    def __init__(self, n_users, n_items, n_skills, n_wins, seed):
        super().__init__()
        np.random.seed(seed)
        init_attempt = [0.10642711, 0.05139345, -0.05963878, -0.06978262, 0.01394569]
        init_correct = [0.39765859, 0.05107533, 0.1765534, 0.23617729, 0.19425494]
        self.alpha = torch.nn.Parameter(-1 * torch.ones(1).view(-1, 1))
        self.delta = torch.nn.Parameter(torch.zeros(n_items).view(-1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(n_skills).view(-1, 1))
        self.attempt = torch.nn.Parameter(
            torch.tensor(np.tile(init_attempt, n_skills).astype(np.float32)).view(-1, 1)
        )
        self.correct = torch.nn.Parameter(
            torch.tensor(np.tile(init_correct, n_skills).astype(np.float32)).view(-1, 1)
        )
        self.h = torch.ones(1).view(-1, 1) * 0.3
        self.d = torch.ones(1).view(-1, 1) * 0.8

    def forward(self, input_ts, lag_ts):
        weight = torch.cat(
            [self.alpha, self.delta, self.beta, self.correct, self.attempt], dim=0
        )
        memory = torch.sigmoid(torch.mm(input_ts, weight))
        y_pred = (1 - memory) * (1 + self.h * lag_ts) ** (-self.d) + memory

        return y_pred


class MyLoss(torch.nn.Module):
    def __init__(self, coef):
        super().__init__()
        self.coef = coef

    def _gaussian_loss(self, float_m, float_std, tensor_input):
        ret = (
            1
            - Normal(torch.tensor([float_m]), torch.tensor([float_std]))
            .log_prob(tensor_input)
            .exp()
            / Normal(torch.tensor([float_m]), torch.tensor([float_std]))
            .log_prob(torch.tensor([float_m]))
            .exp()
        )
        return ret

    def forward(self, pred, target, named_parameters, prev_model_weight_dict):
        model_weight_dict = {}
        for name, param in named_parameters:
            model_weight_dict[name] = param

        fitting_loss = torch.nn.functional.mse_loss(
            input=pred, target=target, reduction="mean"
        )
        alpha_gaussian_loss = self._gaussian_loss(0.0, 0.6, model_weight_dict["alpha"])
        delta_gaussian_loss = self._gaussian_loss(0.0, 0.95, model_weight_dict["delta"])
        beta_gaussian_loss = self._gaussian_loss(0.05, 0.4, model_weight_dict["beta"])
        attempt_gaussian_loss = self._gaussian_loss(
            0.008468969901277483, 0.1710594116668055, model_weight_dict["attempt"]
        )
        correct_gaussian_loss = self._gaussian_loss(
            0.21114390920786338, 0.3781300105771393, model_weight_dict["correct"]
        )
        alpha_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["alpha"],
            target=prev_model_weight_dict["alpha"],
            reduction="mean",
        )
        delta_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["delta"],
            target=prev_model_weight_dict["delta"],
            reduction="mean",
        )
        beta_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["beta"],
            target=prev_model_weight_dict["beta"],
            reduction="mean",
        )
        attempt_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["attempt"],
            target=prev_model_weight_dict["attempt"],
            reduction="mean",
        )
        correct_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["correct"],
            target=prev_model_weight_dict["correct"],
            reduction="mean",
        )
        alpha_loss = (1 - self.coef[0]) * alpha_gaussian_loss.mean() + self.coef[
            0
        ] * alpha_fix_loss
        delta_loss = (1 - self.coef[0]) * delta_gaussian_loss.mean() + self.coef[
            0
        ] * delta_fix_loss
        beta_loss = (1 - self.coef[0]) * beta_gaussian_loss.mean() + self.coef[
            0
        ] * beta_fix_loss
        attempt_loss = (1 - self.coef[0]) * attempt_gaussian_loss.mean() + self.coef[
            0
        ] * attempt_fix_loss
        correct_loss = (1 - self.coef[0]) * correct_gaussian_loss.mean() + self.coef[
            0
        ] * correct_fix_loss

        loss = (
            self.coef[1] * fitting_loss
            + self.coef[2] * alpha_loss
            + self.coef[3] * delta_loss
            + self.coef[4] * beta_loss
            + self.coef[5] * attempt_loss
            + self.coef[6] * correct_loss
        )
        return loss


class InnerModel:
    def __init__(
        self,
        n_items,
        n_wins,
        max_steps,
        seed,
        n_items_for_sessions,
        delay_in_each_session,
        isi,
        item_skill_mat,
        result_folder,
        lr,
        coef_for_loss_fn,
        log_prob,
    ):
        np.random.seed(seed)
        self.result_folder = result_folder
        os.makedirs(self.result_folder, exist_ok=True)
        self.fig_folder = self.result_folder + "/weight_loss_log"
        self.model_folder = self.result_folder + "/model_log"
        os.makedirs(self.fig_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)

        self.n_items = n_items
        self.n_wins = n_wins
        self.max_steps = max_steps
        self.n_items_for_sessions = n_items_for_sessions
        self.delay_in_each_session = delay_in_each_session
        self.isi = isi
        self.item_skill_mat = item_skill_mat
        self.n_skills = self.item_skill_mat.shape[1]
        self.log_prob = log_prob

        self.n_item_feats = int(np.log(2 * self.n_items))

        self.item_feats = np.random.normal(
            np.zeros(2 * self.n_items * self.n_item_feats),
            np.ones(2 * self.n_items * self.n_item_feats),
        ).reshape((2 * self.n_items, self.n_item_feats))

        self.action_space = spaces.Discrete(self.n_items)
        self.observation_space = spaces.Box(
            np.concatenate((np.ones(self.n_item_feats) * -sys.maxsize, np.zeros(3))),
            np.concatenate((np.ones(self.n_item_feats + 2) * sys.maxsize, np.ones(1))),
        )

        self.model = DAS3HInner(
            n_users=1,
            n_items=self.n_items,
            n_skills=self.n_skills,
            n_wins=self.n_wins,
            seed=seed,
        )

        self.loss_fn = MyLoss(coef=coef_for_loss_fn)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=2, gamma=0.2
        )

        self.now = 0
        self.init_session_time = 0
        self.regression_num = -1
        self.q = defaultdict(lambda: utils.OurQueue())
        self.q_for_RL = defaultdict(lambda: utils.OurQueue())
        self.last_time = defaultdict(lambda: -10)
        self.last_time_for_RL = defaultdict(lambda: -10)
        self.curr_delay = None
        self.curr_item = None
        self.curr_outcome = None
        self.curr_step = None
        self.skill_ids = None
        self.prev_weights_dict = None

        self._init_step()

    def _init_step(self):
        self.curr_step = -1
        self.q_for_RL = copy.deepcopy(self.q)
        self.last_time_for_RL = self.last_time.copy()
        self.now = self.init_session_time

    def _encode_log_data(self, session_data_path, data_path):
        log_data = pd.read_csv(session_data_path)

        onehot = OneHotEncoder(sparse=False)

        target = pd.read_csv(data_path, usecols=["outcome"]).values

        X = {}
        X["users"] = onehot.fit_transform(np.array(log_data["learner"]).reshape(-1, 1))
        X["items"] = np.empty((0, self.n_items))
        X["skills"] = np.empty((0, self.n_skills))
        X["corrects"] = np.empty((0, self.n_skills * self.n_wins))
        X["attempts"] = np.empty((0, self.n_skills * self.n_wins))
        now_lag = np.zeros(len(log_data))

        for index, row in log_data.iterrows():
            item_vec = np.zeros(self.n_items)
            skill_vec = np.zeros(self.n_skills)
            correct_vec = np.zeros(self.n_skills * self.n_wins)
            attempt_vec = np.zeros(self.n_skills * self.n_wins)

            item_id = row["action"]
            item_vec[item_id] = 1

            index_of_selected_skills = np.argwhere(self.item_skill_mat[item_id] == 1)

            skill_ids = index_of_selected_skills.transpose()[0].tolist()
            skill_ids = list(set(skill_ids))
            skill_vec[skill_ids] = 1

            last_time = self.last_time[item_id]
            now_lag[index] = row["time"] - last_time
            self.last_time[item_id] = row["time"]

            for skill_id in skill_ids:
                correct_vec[
                    skill_id * self.n_wins : (skill_id + 1) * self.n_wins
                ] = np.log(
                    1 + np.array(self.q[skill_id, "correct"].get_counters(row["time"]))
                )
                if row["outcome"] == 1:
                    self.q[skill_id, "correct"].push(row["time"])
                attempt_vec[
                    skill_id * self.n_wins : (skill_id + 1) * self.n_wins
                ] = np.log(1 + np.array(self.q[skill_id].get_counters(row["time"])))
                self.q[skill_id].push(row["time"])

            X["items"] = np.vstack((X["items"], item_vec))
            X["skills"] = np.vstack((X["skills"], skill_vec))
            X["corrects"] = np.vstack((X["corrects"], correct_vec))
            X["attempts"] = np.vstack((X["attempts"], attempt_vec))

        if os.path.isfile(self.result_folder + "/encoded_study_log.npy"):
            prev_encoded_data = np.load(self.result_folder + "/encoded_study_log.npy")
            now_encoded_data = np.hstack(
                (X["users"], X["items"], X["skills"], X["corrects"], X["attempts"])
            )
            encoded_log_data = np.vstack((prev_encoded_data, now_encoded_data))
        else:
            encoded_log_data = np.hstack(
                (X["users"], X["items"], X["skills"], X["corrects"], X["attempts"])
            )
        np.save(self.result_folder + "/encoded_study_log", encoded_log_data)
        if os.path.isfile(self.result_folder + "/lag_data.npy"):
            prev_lag = np.load(self.result_folder + "/lag_data.npy")
            lag_data = np.hstack((prev_lag, now_lag))
        else:
            lag_data = now_lag
        np.save(self.result_folder + "/lag_data", lag_data)
        logger.debug("The length of the encoded_log_data : {}.".format(len(encoded_log_data)))
        logger.debug("The length of the lag_data : {}.".format(len(lag_data)))
        return encoded_log_data, lag_data, target

    def _modified_default_collate(self, batch):
        batch = batch[0]
        return [each for each in batch]

    def _make_model_weight_dict(self):
        ret_dict = {}
        with torch.no_grad():
            ret_dict["alpha"] = self.model.state_dict()["alpha"].clone().detach()
            ret_dict["delta"] = self.model.state_dict()["delta"].clone().detach()
            ret_dict["beta"] = self.model.state_dict()["beta"].clone().detach()
            ret_dict["attempt"] = self.model.state_dict()["attempt"].clone().detach()
            ret_dict["correct"] = self.model.state_dict()["correct"].clone().detach()
            ret_dict["h"] = self.model.h
            ret_dict["d"] = self.model.d
        return ret_dict

    def _run_log_regression(self, data, lag, target, minibatch_size, epochs):
        self.regression_num += 1
        data_set = MyDataSet(data_np=data, lag_np=lag, target_np=target)
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(data_set),
            batch_size=minibatch_size,
            drop_last=True,
        )

        dataloader = torch.utils.data.DataLoader(
            data_set, sampler=sampler, collate_fn=self._modified_default_collate
        )

        loss_list = []
        weight_list_dict = {
            "alpha": [],
            "beta": [],
            "h": [],
            "d": [],
            "delta": [],
            "attempt": [],
            "correct": [],
        }
        if self.prev_weights_dict == None:
            self.prev_weights_dict = self._make_model_weight_dict()
        for epoch in range(epochs):
            for t, (data, lag, target) in enumerate(dataloader):
                y_pred = self.model(data, lag)

                with torch.no_grad():
                    weight_list_dict["alpha"].append(
                        self.prev_weights_dict["alpha"].view(-1).numpy()
                    )
                    weight_list_dict["delta"].append(
                        self.prev_weights_dict["delta"].view(-1).numpy()
                    )
                    weight_list_dict["beta"].append(
                        self.prev_weights_dict["beta"].view(-1).numpy()
                    )
                    weight_list_dict["attempt"].append(
                        self.prev_weights_dict["attempt"].view(-1, 5).numpy()
                    )
                    weight_list_dict["correct"].append(
                        self.prev_weights_dict["correct"].view(-1, 5).numpy()
                    )
                    weight_list_dict["h"].append(
                        self.prev_weights_dict["h"].view(-1).numpy()
                    )
                    weight_list_dict["d"].append(
                        self.prev_weights_dict["d"].view(-1).numpy()
                    )

                loss = self.loss_fn(
                    y_pred,
                    target,
                    self.model.named_parameters(),
                    self.prev_weights_dict,
                )
                self.prev_weights_dict = self._make_model_weight_dict()

                logger.info("Epoch{:0>2}_loss: {}".format(epoch, loss.item()))
                loss_list.append(loss.item())

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

        logger.info("Session{:0>2}_inner_lr: {}".format(self.regression_num, self.scheduler.get_last_lr()[0]))
        self.scheduler.step()

        color_list = list(plt.cm.get_cmap("tab20b").colors) + list(
            plt.cm.get_cmap("tab20c").colors
        )

        fig = plt.figure()

        ax = fig.add_subplot(
            111,
            title="loss_log",
            xlabel="epoch",
            ylabel="MSE_loss[reduction=sum]",
            xlim=(0, len(dataloader) + 1),
            ylim=(0, max(loss_list) + max(loss_list) / 20),
        )

        for epoch in range(epochs):
            x = range(1, len(dataloader) + 1)
            y = loss_list[epoch * len(dataloader) : (epoch + 1) * len(dataloader)]
            ax.plot(
                x,
                y,
                label="epoch%s" % str(epoch + 1),
                color=color_list[epoch],
                linewidth=1,
                zorder=epoch,
                marker="o",
                markersize=5,
            )
        ax.legend()

        self.model = self.model.to("cpu")
        torch.save(
            self.model.state_dict(),
            self.model_folder + "/" + str(self.regression_num) + "_inner_model_cpu.pth",
        )
        fig.savefig(
            self.fig_folder + "/" + str(self.regression_num) + "_loss_log.png",
            bbox_inches="tight",
        )
        fig.clf
        plt.close(fig)

    def _make_input_vec(self, selected_item_id, mode):
        user_num = 1
        item_num = self.n_items
        skill_num = self.n_skills

        user_vec = np.zeros(user_num)
        item_vec = np.zeros(item_num)
        skill_vec = np.zeros(skill_num)
        correct_vec = np.zeros(self.n_skills * self.n_wins)
        attempt_vec = np.zeros(self.n_skills * self.n_wins)

        last_time = self.last_time_for_RL[selected_item_id]
        lag = self.now - last_time
        if mode == "solve":
            self.last_time_for_RL[selected_item_id] = self.now

        user_vec[0] = 1
        item_vec[selected_item_id] = 1

        index_of_selected_skills = np.argwhere(
            self.item_skill_mat[selected_item_id] == 1
        )
        self.skill_ids = index_of_selected_skills.transpose()[0].tolist()
        self.skill_ids = list(set(self.skill_ids))
        skill_vec[self.skill_ids] = 1

        for skill_id in self.skill_ids:
            correct_vec[skill_id * self.n_wins : (skill_id + 1) * self.n_wins] = np.log(
                1 + np.array(self.q_for_RL[skill_id, "correct"].get_counters(self.now))
            )
            attempt_vec[skill_id * self.n_wins : (skill_id + 1) * self.n_wins] = np.log(
                1 + np.array(self.q_for_RL[skill_id].get_counters(self.now))
            )

        return_data_ts = torch.tensor(
            np.hstack((user_vec, item_vec, skill_vec, correct_vec, attempt_vec)).astype(
                np.float32
            )
        ).view(1, -1)
        return_lag_ts = torch.tensor(np.ones(1).astype(np.float32) * lag).view(-1, 1)
        return return_data_ts, return_lag_ts

    def inner_modeling(self, session_log_csv_path, log_csv_path):
        encoded_data, lag, target = self._encode_log_data(
            session_log_csv_path, log_csv_path
        )
        self._run_log_regression(
            data=encoded_data,
            lag=lag,
            target=target,
            minibatch_size=len(encoded_data),
            epochs=20,
        )

    def _encode_delay(self):
        v = np.zeros(2)
        v[self.curr_outcome] = np.log(1 + self.curr_delay)
        return v

    def _vectorized_obs(self):
        encoded_item = self.item_feats[
            self.n_items * self.curr_outcome + self.curr_item, :
        ]
        return np.hstack(
            (encoded_item, self._encode_delay(), np.array([self.curr_outcome]))
        )

    def step(self, action):
        self.curr_step += 1
        self.curr_item = action
        if self.curr_step % self.n_items_for_sessions == 1:
            self.curr_delay = self.isi
        elif self.curr_step == 0:
            self.curr_delay = 0
        else:
            self.curr_delay = self.delay_in_each_session
        self.now += self.curr_delay
        input_vec, lag = self._make_input_vec(self.curr_item, "solve")
        with torch.no_grad():
            recall_prob = self.model(input_vec, lag).item()
        self.curr_outcome = 1 if np.random.random() < recall_prob else 0

        for skill_id in self.skill_ids:
            if self.curr_outcome == 1:
                self.q_for_RL[skill_id, "correct"].push(self.now)
            self.q_for_RL[skill_id].push(self.now)

        obs = self._vectorized_obs()
        r = self._rew()
        done = self.curr_step == self.max_steps
        info = {}
        return obs, r, done, info

    def _rew(self):
        reward_list = []
        for item in range(self.n_items):
            input_vec, lag = self._make_input_vec(item, "get")
            with torch.no_grad():
                recall_prob = self.model(input_vec, lag).item()
            if self.log_prob:
                recall_prob = np.log(1e-9 + recall_prob)
            reward_list.append(recall_prob)
        rew = np.array(reward_list)
        return rew

    def reset(self):
        self._init_step()
        return self.step(np.random.randint(self.n_items))[0]
