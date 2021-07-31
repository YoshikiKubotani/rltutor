import numpy as np


def make_item_skill_mat(n_items, n_skills, seed):
    i = 0
    while True:
        np.random.seed(seed + i)
        item_skill_mat = np.zeros((n_items, n_skills)).astype("int8")
        weight = [0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        for each_item in range(n_items):
            n_multi_skills = np.random.choice(range(1, 9), p=weight)
            selected_skills = np.random.choice(
                range(0, n_skills), n_multi_skills, replace=False
            )
            item_skill_mat[each_item, selected_skills] = 1
        if all(
            [
                skill_id in set(np.argwhere(item_skill_mat == 1).transpose()[1])
                for skill_id in range(n_skills)
            ]
        ):
            print("seed:", seed + i)
            break
        i += 1
    return item_skill_mat
