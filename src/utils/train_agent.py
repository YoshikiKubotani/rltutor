from collections import deque
import logging
import os

import numpy as np
import pandas as pd

from pfrl.experiments.evaluator import save_agent


def train_agent_batch(
    agent,
    env,
    steps,
    outdir,
    session_num,
    num_steps,
    num_items,
    log_interval=None,
    max_episode_len=None,
    step_offset=0,
    step_hooks=(),
    return_window_size=100,
    logger=None,
):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        steps (int): Number of total time steps for training.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)
    recent_returns = deque(maxlen=return_window_size)

    num_envs = env.num_envs
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype="i")
    episode_len = np.zeros(num_envs, dtype="i")

    # o_0, r_0
    obss = env.reset()

    reward = []
    reward_train = [[] for i in range(num_envs)]
    reward_train_mean_over_training = []
    reward_df = pd.DataFrame()

    recall_prob_transition = np.zeros((num_steps, num_items))
    counter = 0
    agent_folder = os.path.abspath(outdir + "/agent")
    reward_folder = os.path.abspath(outdir + "/reward_log")
    retention_folder = os.path.abspath(outdir + "/retention_log")
    os.makedirs(agent_folder, exist_ok=True)
    os.makedirs(reward_folder, exist_ok=True)
    os.makedirs(retention_folder, exist_ok=True)

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    try:
        while True:
            # a_t
            actions = agent.batch_act(obss)
            # o_{t+1}, r_{t+1}
            obss, rs, dones, infos = env.step(actions)

            recall_prob_transition[counter % 200, :] = rs[0]
            rs = tuple(np.array(rs).mean(axis=1))

            for i in range(num_envs):
                reward_train[i].append(rs[i])

            if dones[0]:
                reward_train_mean_over_agent = np.mean(reward_train, axis=0)
                real_mean = np.mean(reward_train_mean_over_agent)
                real_mean_df = pd.DataFrame(
                    {"update%s" % str(episode_idx[0] + 1): [real_mean]}, index=["mean"]
                )
                reward_train_mean_over_agent_df = pd.DataFrame(
                    {"update%s" % str(episode_idx[0] + 1): reward_train_mean_over_agent}
                )
                each_reward_df = pd.concat(
                    [real_mean_df, reward_train_mean_over_agent_df], axis=0
                )
                reward_df = pd.concat([reward_df, each_reward_df], axis=1)
                reward_train_mean_over_training.append(real_mean)
                reward_train = [[] for i in range(num_envs)]

            episode_r += rs
            episode_len += 1

            # Compute mask for done and reset
            if max_episode_len is None:
                resets = np.zeros(num_envs, dtype=bool)
            else:
                resets = episode_len == max_episode_len
            resets = np.logical_or(
                resets, [info.get("needs_reset", False) for info in infos]
            )
            # Agent observes the consequences
            # agent update once per 12000 times
            agent.batch_observe(obss, rs, dones, resets)

            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)

            # For episodes that ends, do the following:
            #   1. increment the episode count
            #   2. record the return
            #   3. clear the record of rewards
            #   4. clear the record of the number of steps
            #   5. reset the env to start a new episode
            # 3-5 are skipped when training is already finished.
            episode_idx += end
            recent_returns.extend(episode_r[end])

            for _ in range(num_envs):
                t += 1

                for hook in step_hooks:
                    hook(env, agent, t)

            if (
                log_interval is not None
                and t >= log_interval
                and t % log_interval < num_envs
            ):
                """
                sum_step:   sum of all the loop in whole training (upper bound = steps)
                timestep:   the number of steps for each agentbatch (upper bound = steps // num_envs)
                episode:    the number of episodes (upper bound = optimize_times)
                last_reward:    the last reward the agent got through the interaction
                last_update_reward_average: the average reward of the last episode
                """
                logger.info(
                    "sum_step:{} timestep:{} episode:{} last_reward: {} last_update_reward_average:{}".format(  # NOQA
                        t,
                        t // num_envs,
                        episode_idx[0],
                        np.mean(rs),
                        reward_train_mean_over_training[-1]
                        if reward_train_mean_over_training != []
                        else np.nan,
                    )
                )
                logger.debug("statistics: {}".format(agent.get_statistics()))

            if t >= steps:
                reward_df.to_csv(reward_folder + "/ses_%s_reward.csv" % session_num)
                break

            # Start new episodes if needed
            episode_r[end] = 0
            episode_len[end] = 0
            obss = env.reset(not_end)
            counter += 1

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        env.close()
    else:
        # Save the final model
        save_name = "session_%s" % session_num
        save_agent(agent, save_name, agent_folder, logger, suffix="_finish")

    # return reward
    return reward_train_mean_over_training