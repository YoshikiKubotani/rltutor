import sys
import copy
import math
import pathlib
import json
import os
import torch

# import joblib
import logging
import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

import student
import tutor
import utils

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    save_root = os.getcwd()
    current_dir = pathlib.Path(hydra.utils.get_original_cwd())

    num_total_ses = cfg.days * cfg.num_sessions_per_day
    session_interval = 1 * 24 * 60 * 60 // cfg.num_sessions_per_day

    best_score = None
    AGENT_FOLDER = os.path.abspath(
        os.path.join(save_root, "logs/tutor")
    )
    STUDENT_LOG_FOLDER = os.path.abspath(
        os.path.join(save_root, "logs/study_log")
    )
    os.makedirs(STUDENT_LOG_FOLDER, exist_ok=True)

    RLTUTOR_STUDENT_LOG_FOLDER = STUDENT_LOG_FOLDER + "/RLTutor"
    RANDOM_STUDENT_LOG_FOLDER = STUDENT_LOG_FOLDER + "/Random"
    LEITNERTUTOR_STUDENT_LOG_FOLDER = STUDENT_LOG_FOLDER + "/LeitnerTutor"
    THRESHOLDTUTOR_STUDENT_LOG_FOLDER = STUDENT_LOG_FOLDER + "/ThresholdTutor"
    os.makedirs(RLTUTOR_STUDENT_LOG_FOLDER, exist_ok=True)
    os.makedirs(RANDOM_STUDENT_LOG_FOLDER, exist_ok=True)
    os.makedirs(LEITNERTUTOR_STUDENT_LOG_FOLDER, exist_ok=True)
    os.makedirs(LEITNERTUTOR_STUDENT_LOG_FOLDER + "/best_aprob", exist_ok=True)
    os.makedirs(THRESHOLDTUTOR_STUDENT_LOG_FOLDER, exist_ok=True)
    os.makedirs(THRESHOLDTUTOR_STUDENT_LOG_FOLDER + "/best_thresh", exist_ok=True)

    item_skill_mat = utils.make_item_skill_mat(cfg.num_items, cfg.skills, cfg.seed)
    pd.DataFrame(item_skill_mat).to_csv(
        os.path.join(save_root, "logs/item_%sskill_mat.csv" % str(cfg.skills)),
        index=False,
        header=False,
    )

    ### 1. RLTutor Tutoring ######
    logger.info("Start RLTuor Tutoring.")
    now = 0
    reward_list_for_plot = []

    learned_weight_folder = os.path.join(current_dir, "data/pretrained_weight.pkl")

    time_weight = utils.make_time_weight(
        num_skills=cfg.skills,
        folder_path=learned_weight_folder,
        seed=cfg.seed,
    )
    model = student.DAS3HStudent(time_weight, cfg.num_items, cfg.skills, cfg.seed)
    student_model = student.StudentModel(
        n_items=cfg.num_items,
        n_skills=cfg.skills,
        n_wins=cfg.time_windows,
        seed=cfg.seed,
        item_skill_mat=item_skill_mat,
        model=model,
    )

    def tutoring(num_questions, init_instruction, agent, now):
        actions = []
        outcomes = []
        times = []
        each_recall_probs = []
        obs = student_model._vectorized_obs()
        if init_instruction and agent.name != "RLTutor":
            first_actions = pd.read_csv(
                RLTUTOR_STUDENT_LOG_FOLDER + "/first_instruction.csv"
            )["action"].values.tolist()
            assert len(first_actions) == num_questions
        for i in range(num_questions):
            if i != 0:
                now += cfg.interval
            if init_instruction and agent.name == "RLTutor":
                action = np.random.randint(cfg.num_items)
            elif init_instruction and agent.name != "RLTutor":
                action = first_actions[i]
            else:
                if agent.name == "RLTutor":
                    action = agent.act(obs)
                elif agent.name == "LeitnerTutor":
                    action = agent.act(student_model.curr_item, student_model.curr_outcome)
                elif agent.name == "ThresholdTutor":
                    inner_now = student_model.now
                    student_model.now = now
                    action = agent.act()
                    student_model.now = inner_now
                elif agent.name == "RandomTutor":
                    action = agent.act()
            outcome, obs = student_model.step(action, now)
            each_recall_prob = student_model.get_retention_rate()
            actions.append(action)
            outcomes.append(outcome)
            times.append(now)
            each_recall_probs.append(each_recall_prob)

        df = pd.DataFrame(
            {
                "learner": 1,
                "action": actions,
                "time": times,
                "outcome": outcomes,
            }
        )
        return df, each_recall_probs, now


    inner_model = student.InnerModel(
        n_items=cfg.num_items,
        n_wins=cfg.time_windows,
        max_steps=cfg.steps_per_updates,
        seed=cfg.seed,
        n_items_for_sessions=cfg.num_items_for_each_session,
        delay_in_each_session=cfg.interval,
        isi=session_interval,
        item_skill_mat=item_skill_mat,
        result_folder=os.path.join(AGENT_FOLDER, "inner_model"),
        lr=cfg.inner_lr,
        coef_for_loss_fn=cfg.coefs,
        log_prob=cfg.log_reward,
    )
    rl_tutor = tutor.RLTutor(
        env=inner_model,
        num_iteration=cfg.updates,
        num_envs=cfg.parallel_envs,
        num_timesteps=cfg.steps_per_updates,
        seed=cfg.seed,
        gamma=cfg.gamma,
        lambd=cfg.lambd,
        value_func_coef=cfg.value_func_coef,
        entropy_coef=cfg.entropy_coef,
        clip_eps=cfg.clip_eps,
    )

    with open(STUDENT_LOG_FOLDER + "/student_model_parameters.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "alpha": student_model.predictor.alpha.tolist(),
                    "delta": student_model.predictor.delta.tolist(),
                    "beta": student_model.predictor.beta.tolist(),
                    "correct": np.vsplit(
                        student_model.predictor.time_weight.reshape(-1, 5), 2
                    )[0].tolist(),
                    "attempt": np.vsplit(
                        student_model.predictor.time_weight.reshape(-1, 5), 2
                    )[1].tolist(),
                    "h": student_model.predictor.h,
                    "d": student_model.predictor.d,
                },
                indent=4,
            )
        )
    df, retention_rate_list, now = tutoring(
        num_questions=cfg.num_items_for_pre_test, init_instruction=True, agent=rl_tutor, now=now
    )
    reward_list_for_plot += retention_rate_list
    df.to_csv(RLTUTOR_STUDENT_LOG_FOLDER + "/first_instruction.csv", index=False)
    df.to_csv(RLTUTOR_STUDENT_LOG_FOLDER + "/study_log.csv", index=False)

    inner_model.inner_modeling(
        session_log_csv_path=RLTUTOR_STUDENT_LOG_FOLDER + "/first_instruction.csv",
        log_csv_path=RLTUTOR_STUDENT_LOG_FOLDER + "/study_log.csv",
    )
    inner_model.init_session_time = now

    train_reward_log = np.zeros((num_total_ses, cfg.updates))

    for n in range(num_total_ses):
        print()
        train_reward_log[n, :] = rl_tutor.train(
            session_num=n,
            output_dir=os.path.join(AGENT_FOLDER, "rltutor"),
            logger=logger,
            lr=cfg.lr,
        )
        print()

        now += session_interval

        df, retention_rate_list, now = tutoring(
            num_questions=cfg.num_items_for_each_session, init_instruction=False, agent=rl_tutor, now=now
        )
        reward_list_for_plot += retention_rate_list
        df.to_csv(RLTUTOR_STUDENT_LOG_FOLDER + "/instruction_%s.csv" % n, index=False)
        df.to_csv(
            RLTUTOR_STUDENT_LOG_FOLDER + "/study_log.csv",
            index=False,
            mode="a",
            header=False,
        )

        inner_model.inner_modeling(
            session_log_csv_path=RLTUTOR_STUDENT_LOG_FOLDER + "/instruction_%s.csv" % n,
            log_csv_path=RLTUTOR_STUDENT_LOG_FOLDER + "/study_log.csv",
        )
        inner_model.init_session_time = now

    state_dict = inner_model.model.state_dict()
    with open(STUDENT_LOG_FOLDER + "/RL_inner_model_final_parameters.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "alpha": state_dict["alpha"].numpy().squeeze().tolist(),
                    "delta": state_dict["delta"].numpy().squeeze().tolist(),
                    "beta": state_dict["beta"].numpy().squeeze().tolist(),
                    "correct": state_dict["correct"].numpy().squeeze().tolist(),
                    "attempt": state_dict["attempt"].numpy().squeeze().tolist(),
                },
                indent=4,
            )
        )
    column = ["item%s" % str(n + 1) for n in range(cfg.num_items)]
    result_df = pd.DataFrame(reward_list_for_plot, columns=column)
    result_df.to_csv(RLTUTOR_STUDENT_LOG_FOLDER + "/result.csv", index=False)


    ### 2. RandomTutor Tutoring ######
    logger.info("Start RandomTutor Tutoring.")
    student_model.reset(seed=cfg.seed)
    random_tutor = tutor.RandomTutor(
        n_items=cfg.num_items,
        num_offering_questions=num_total_ses * cfg.num_items_for_each_session,
        seed=cfg.seed,
    )
    now = 0
    reward_list_for_plot = []

    df, retention_rate_list, now = tutoring(
        num_questions=cfg.num_items_for_pre_test, init_instruction=True, agent=random_tutor, now=now
    )
    reward_list_for_plot += retention_rate_list
    df.to_csv(RANDOM_STUDENT_LOG_FOLDER + "/R_first_instruction.csv", index=False)
    df.to_csv(RANDOM_STUDENT_LOG_FOLDER + "/R_study_log.csv", index=False)

    for n in range(num_total_ses):
        now += session_interval

        df, retention_rate_list, now = tutoring(
            num_questions=cfg.num_items_for_each_session, init_instruction=False, agent=random_tutor, now=now
        )
        reward_list_for_plot += retention_rate_list
        df.to_csv(RANDOM_STUDENT_LOG_FOLDER + "/R_instruction_%s.csv" % n, index=False)
        df.to_csv(
            RANDOM_STUDENT_LOG_FOLDER + "/R_study_log.csv",
            index=False,
            mode="a",
            header=False,
        )

    column = ["item%s" % str(n + 1) for n in range(cfg.num_items)]
    result_df = pd.DataFrame(reward_list_for_plot, columns=column)
    result_df.to_csv(RANDOM_STUDENT_LOG_FOLDER + "/R_result.csv", index=False)


    ### 3. LeitnerTutoring Tutoring ######
    logger.info("Start LeinerTutor Tutoring.")
    cut = lambda x: math.floor(x * 10 ** 2) / (10 ** 2)
    for aprob in np.arange(0, 1, 0.01):
        student_model.reset(seed=cfg.seed)
        leitner_tutor = tutor.LeitnerTutor(n_items=cfg.num_items, arrival_prob=aprob)

        now = 0
        reward_list_for_plot = []

        df, retention_rate_list, now = tutoring(
            num_questions=cfg.num_items_for_pre_test,
            init_instruction=True,
            agent=leitner_tutor,
            now=now
        )
        reward_list_for_plot += retention_rate_list
        df.to_csv(
            LEITNERTUTOR_STUDENT_LOG_FOLDER + "/LEIT_first_instruction.csv", index=False
        )
        df.to_csv(
            LEITNERTUTOR_STUDENT_LOG_FOLDER + "/%s_LEIT_study_log.csv" % str(cut(aprob)),
            index=False,
        )

        for n in range(num_total_ses):
            now += session_interval

            df, retention_rate_list, now = tutoring(
                num_questions=cfg.num_items_for_each_session, init_instruction=False, agent=leitner_tutor, now=now
            )
            reward_list_for_plot += retention_rate_list
            df.to_csv(
                LEITNERTUTOR_STUDENT_LOG_FOLDER
                + "/%s_LEIT_study_log.csv" % str(cut(aprob)),
                index=False,
                mode="a",
                header=False,
            )

        now_final_mean_retention = sum(reward_list_for_plot[-1]) / len(
            reward_list_for_plot[-1]
        )
        if best_score is None or now_final_mean_retention > best_score:
            best_score = now_final_mean_retention
            best_reward_list_for_plot = reward_list_for_plot
            best_aprob = aprob

    column = ["item%s" % str(n + 1) for n in range(cfg.num_items)]
    result_df = pd.DataFrame(best_reward_list_for_plot, columns=column)
    with open(
        LEITNERTUTOR_STUDENT_LOG_FOLDER + "/best_aprob/best_arrival_probability.json", "w"
    ) as f:
        f.write(
            json.dumps(
                {
                    "arrival_probability": cut(best_aprob),
                },
                indent=4,
            )
        )
    result_df.to_csv(
        LEITNERTUTOR_STUDENT_LOG_FOLDER + "/best_aprob/LEIT_best_result.csv", index=False
    )


    ### 4. ThresholdTutor Tutoring ######
    logger.info("Start ThresholdTutor Tutoring.")
    best_score = 0
    cut = lambda x: math.floor(x * 10 ** 2) / (10 ** 2)
    for thresh in np.arange(0, 1, 0.01):
        student_model.reset(seed=cfg.seed)
        threshold_tutor = tutor.ThresholdTutor(env=student_model, threshold=thresh)

        now = 0
        reward_list_for_plot = []

        df, retention_rate_list, now = tutoring(
            num_questions=cfg.num_items_for_pre_test,
            init_instruction=True,
            agent=threshold_tutor,
            now=now
        )
        reward_list_for_plot += retention_rate_list
        df.to_csv(
            THRESHOLDTUTOR_STUDENT_LOG_FOLDER + "/TH_first_instruction.csv", index=False
        )
        df.to_csv(
            THRESHOLDTUTOR_STUDENT_LOG_FOLDER + "/%s_TH_study_log.csv" % str(cut(thresh)),
            index=False,
        )

        for n in range(num_total_ses):
            now += session_interval

            df, retention_rate_list, now = tutoring(
                num_questions=cfg.num_items_for_each_session,
                init_instruction=False,
                agent=threshold_tutor,
                now=now
            )
            reward_list_for_plot += retention_rate_list
            df.to_csv(
                THRESHOLDTUTOR_STUDENT_LOG_FOLDER
                + "/%s_TH_study_log.csv" % str(cut(thresh)),
                index=False,
                mode="a",
                header=False,
            )

        now_final_mean_retention = sum(reward_list_for_plot[-1]) / len(
            reward_list_for_plot[-1]
        )
        if best_score is None or now_final_mean_retention > best_score:
            best_score = now_final_mean_retention
            best_reward_list_for_plot = reward_list_for_plot
            best_thresh = thresh

    column = ["item%s" % str(n + 1) for n in range(cfg.num_items)]
    result_df = pd.DataFrame(best_reward_list_for_plot, columns=column)
    with open(
        THRESHOLDTUTOR_STUDENT_LOG_FOLDER + "/best_thresh/best_arrival_probability.json",
        "w",
    ) as f:
        f.write(
            json.dumps(
                {
                    "arrival_probability": cut(best_thresh),
                },
                indent=4,
            )
        )
    result_df.to_csv(
        THRESHOLDTUTOR_STUDENT_LOG_FOLDER + "/best_thresh/TH_best_result.csv", index=False
    )

if __name__ == "__main__":
    main()