import numpy as np
import joblib


def make_time_weight(num_skills, folder_path, seed):
    NUM_TIME_WINDOWS = 5
    NUM_SKILLS = num_skills
    predictor = joblib.load(folder_path)
    learned_weight = np.array(predictor.steps[1][1].coef_).flatten()
    correct = np.empty(0)
    attempt = np.empty(0)
    use_skill = [
        1,
        4,
        7,
        10,
        13,
        16,
        18,
        20,
        26,
        28,
        29,
        32,
        33,
        38,
        40,
    ]
    for skill_id in use_skill:
        start = NUM_TIME_WINDOWS * skill_id
        each_correct = learned_weight[
            137906 + start : 137906 + start + NUM_TIME_WINDOWS
        ]
        each_attempt = learned_weight[
            138846 + start : 138846 + start + NUM_TIME_WINDOWS
        ]
        correct = np.hstack((correct, each_correct))
        attempt = np.hstack((attempt, each_attempt))
    time_weight = np.hstack((correct, attempt))

    return time_weight
