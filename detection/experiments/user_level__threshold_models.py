import pandas as pd
import numpy as np
import os, sys
f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, "../.."))
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, auc, roc_curve, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from config.detection_config import user_level_execution_config, user_level_conf, post_level_execution_config
from detection.detection_utils.factory import create_dir_if_missing
sns.set(rc={'figure.figsize': (10, 10)}, font_scale=1.4)
from scipy.optimize import minimize
from utils.my_timeit import timeit
from utils.general import init_log

logger = init_log("user_level_simple_models")


def get_hs_count(current_preds):
    return len(current_preds[current_preds > 0.5])


def fixed_threshold_num_of_posts(user2pred: pd.DataFrame, labeled_users: pd.DataFrame, output_path: str, test_ratio: float):
    """
    Hard threshold of number of HS predictions per user. Threshold is an integer and above 1.
    :param user2pred:
    :param labeled_users:
    :return:
    """
    logger.info("Executing fixed threshold...")
    output_path = os.path.join(output_path, "hard_threshold")
    create_dir_if_missing(output_path)
    user2pred["user_id"] = user2pred["user_id"].astype(str)
    labeled_users["user_id"] = labeled_users["user_id"].astype(str)
    train_idx = labeled_users.sample(frac=(1-test_ratio)).index
    train_labeled_users = labeled_users.loc[train_idx]
    test_labeled_users = labeled_users.drop(train_labeled_users.index, axis=0)

    train_user2pred = user2pred[user2pred["user_id"].isin(list(train_labeled_users["user_id"]))].reset_index(drop=True)
    test_user2pred = user2pred[user2pred["user_id"].isin(list(test_labeled_users["user_id"]))].reset_index(drop=True)


    train_g_df = train_user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(columns={"predictions": "hs_count"})
    test_g_df = test_user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(columns={"predictions": "hs_count"})

    to_plot = {"thresholds": [], "f-scores": [], "precisions": [], "recalls": [], "accuracies": []}
    max_f1 = 0.0
    best_th = 0
    for threshold in tqdm(range(1, 300)):
        to_plot["thresholds"].append(threshold)
        train_g_df["y_pred"] = train_g_df["hs_count"].apply(lambda h_count: 1 if h_count >= threshold else 0)

        true_pred = pd.merge(train_labeled_users, train_g_df, on='user_id')
        y_true = true_pred["label"]
        y_pred = true_pred["y_pred"]
        current_f1_score = f1_score(y_true, y_pred)
        if max_f1 < current_f1_score:
            max_f1 = current_f1_score
            best_th = threshold
        to_plot["f-scores"].append(current_f1_score)
        to_plot["precisions"].append(precision_score(y_true, y_pred))
        to_plot["recalls"].append(recall_score(y_true, y_pred))
        to_plot["accuracies"].append(accuracy_score(y_true, y_pred))
    plt.figure()
    sns.set(rc={'figure.figsize': (6, 6)}, font_scale=1.7)

    for score_ in ["f-score", "precision", "recall", "accuracy"]:
        current_score_name = "accuracies" if score_.endswith("y") else f"{score_}s"
        if score_ != "recall":
            sns.lineplot(to_plot["thresholds"], to_plot[current_score_name],
                         label=f"{score_}" if score_ != 'f-score' else f"{score_} (max: {max(to_plot['f-scores']):.3f})")
        else:
            sns.lineplot(to_plot["thresholds"], to_plot[current_score_name], label=f"{score_}")
    plt.title("Fixed threshold")
    plt.xlabel('Threshold')
    plt.ylabel('Measurement score')
    plt.savefig(os.path.join(output_path, "hard_threshold_plot.png"))
    pd.DataFrame(to_plot).to_csv(os.path.join(output_path, "hard_threshold.csv"), index=False)
    logger.info(f"Max f1-score: {max_f1}")
    logger.info(f"Best threshold: {best_th}")
    # evaluate on test
    test_g_df["y_pred"] = test_g_df["hs_count"].apply(lambda h_count: 1 if h_count >= best_th else 0)
    true_pred = pd.merge(test_labeled_users, test_g_df, on='user_id')
    y_true = true_pred["label"]
    y_pred = true_pred["y_pred"]
    with open(os.path.join(output_path, "evaluation.txt"), "w") as fout:
        fout.write(f"F1-score: {f1_score(y_true, y_pred):.3f}\n")
        fout.write(f"Precision: {precision_score(y_true, y_pred):.3f}\n")
        fout.write(f"Recall: {recall_score(y_true, y_pred):.3f}\n")
        fout.write(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
        fout.write(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")


def relational_threshold(user2pred: pd.DataFrame, labeled_users: pd.DataFrame, output_path: str, dataset_name: str, test_ratio: float):
    """
    Here we consider the assumption that relation to followers/followees effect the users' behaviour.
    For each user - get his average HS score, and the average HS scores of his followers and followees.
    then search for the optimal relational threshold to yield the best f1-score.
    This threshold will be combined from a self-TH + followers-TH + followees-TH.

    :param user2pred:
    :param labeled_users:
    :return:
    """
    logger.info("Executing relational threshold...")
    output_path = os.path.join(output_path, "relational_threshold")
    create_dir_if_missing(output_path)
    user2pred["user_id"] = user2pred["user_id"].astype(str)
    labeled_users["user_id"] = labeled_users["user_id"].astype(str)

    min_mention_threshold = 3
    avg_hs_score_per_user = user2pred.groupby('user_id').agg({"predictions": "mean"}).reset_index() \
        .rename(columns={"predictions": "avg_hs_score"})
    hs_count_per_user = user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
        columns={"predictions": "hs_count"})

    # get followers/followees
    network_dir = f"hate_networks/outputs/{dataset_name.split('_')[0]}_networks/network_data/"
    edges_dir = os.path.join(network_dir, "edges")
    mentions_df = pd.read_csv(os.path.join(edges_dir, "data_users_mention_edges_df.tsv"), sep='\t')
    for col in ['source', 'dest']:
        mentions_df[col] = mentions_df[col].astype(str)
    # keep only mentions above the minimal threshold
    mentions_df = mentions_df[mentions_df["weight"] >= min_mention_threshold].reset_index(drop=True)
    mentions_dict = {}  # users mentioned by the observed user
    mentioned_by_dict = {}  # users mentioning the observed user
    for idx, row in mentions_df.iterrows():
        src = row['source']
        dest = row['dest']
        if src not in mentions_dict.keys():
            mentions_dict[src] = []
        if dest not in mentioned_by_dict.keys():
            mentioned_by_dict[dest] = []
        mentions_dict[src].append(dest)
        mentioned_by_dict[dest].append(src)
    res = pd.DataFrame()
    # SELF_WEIGHT = 0.5
    # FOLLOWERS_WEIGHT = 0.25
    # FOLLOWEES_WEIGHT = 0.25
    for SELF_WEIGHT in np.linspace(0, 1, num=5):
        for FOLLOWERS_WEIGHT in np.linspace(0, 1, num=5):
            if SELF_WEIGHT + FOLLOWERS_WEIGHT >= 1:
                continue
            else:
                FOLLOWEES_WEIGHT = 1.0 - SELF_WEIGHT - FOLLOWERS_WEIGHT
                # logger.info(f"self-weight: {SELF_WEIGHT:.2f}, followers-weight: {FOLLOWERS_WEIGHT:.2f}, followees-weight: {FOLLOWEES_WEIGHT:.2f}")
                user_ids = []
                relational_scores = []
                type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                for user_id in labeled_users["user_id"].tolist():
                    user_ids.append(user_id)
                    has_followees = True
                    has_followers = True
                    if user_id in mentions_dict.keys():
                        current_followees = mentions_dict[user_id]
                        followees_df = hs_count_per_user.loc[
                            hs_count_per_user["user_id"].isin(current_followees), "hs_count"]
                        if len(followees_df) == 0:
                            has_followees = False
                        else:
                            followees_hs_counts = followees_df.mean()
                    else:
                        has_followees = False
                    if user_id in mentioned_by_dict.keys():
                        current_followers = mentioned_by_dict[user_id]
                        followers_df = hs_count_per_user.loc[
                            hs_count_per_user["user_id"].isin(current_followers), "hs_count"]
                        if len(followers_df) == 0:
                            has_followers = False
                        else:
                            followers_hs_counts = followers_df.mean()
                    else:
                        has_followers = False

                    user_hs_count = int(hs_count_per_user.loc[hs_count_per_user["user_id"] == user_id, "hs_count"].iloc[0])
                    if has_followers and has_followees:
                        type_counts[1] += 1
                        current_score = SELF_WEIGHT * user_hs_count + FOLLOWEES_WEIGHT * followees_hs_counts + FOLLOWERS_WEIGHT * followers_hs_counts
                    elif has_followees and not has_followers:
                        type_counts[2] += 1
                        current_score = SELF_WEIGHT * user_hs_count + FOLLOWEES_WEIGHT * followees_hs_counts
                    elif not has_followees and has_followers:
                        type_counts[3] += 1
                        current_score = SELF_WEIGHT * user_hs_count + FOLLOWERS_WEIGHT * followers_hs_counts
                    else:
                        type_counts[4] += 1
                        current_score = SELF_WEIGHT * user_hs_count

                    relational_scores.append(current_score)
                logger.info(type_counts)
                user2relational_score = pd.DataFrame({"user_id": user_ids, "relational_score": relational_scores})

                train_idx = user2relational_score.sample(frac=(1-test_ratio)).index
                train_user2relational_score = user2relational_score.loc[train_idx]

                max_f1 = 0.0
                best_th = 0
                for threshold in tqdm(range(1, 300)):
                    # to_plot["thresholds"].append(threshold)
                    train_user2relational_score["y_pred"] = train_user2relational_score["relational_score"].apply(lambda rs: 1 if rs >= threshold else 0)
                    true_pred = pd.merge(labeled_users, train_user2relational_score, on='user_id')
                    y_true = true_pred["label"]
                    y_pred = true_pred["y_pred"]
                    current_f1_score = f1_score(y_true, y_pred)
                    if max_f1 < current_f1_score:
                        max_f1 = current_f1_score
                        best_th = threshold
                logger.info(f"Max f1-score: {max_f1}")
                logger.info(f"Best threshold: {best_th}")
                res = res.append({"self_weight": SELF_WEIGHT, "followers_weight": FOLLOWERS_WEIGHT,
                                  "followees_weight": FOLLOWEES_WEIGHT, "best_f1_score": max_f1, "best_th": best_th}, ignore_index=True)
    res.to_csv(os.path.join(output_path, "relational_threshold_grid_search.csv"), index=False)


def calc_soft_threhold(hs_score, **kwargs):
    if hs_score < kwargs["LOWER_BOUND"]:
        th = kwargs["high_th"]
    elif kwargs["LOWER_BOUND"] <= hs_score < kwargs["HIGHER_BOUND"]:
        th = kwargs["medium_th"]
    else:
        th = kwargs["low_th"]
    return th


def dynamic_threshold_hs_score(user2pred: pd.DataFrame, labeled_users: pd.DataFrame, output_path: str, test_ratio: float):
    """

    :param user2pred:
    :param labeled_users:
    :param output_path:
    :return:
    """
    logger.info("Executing dynamic/adjusted threshold...")
    output_path = os.path.join(output_path, "soft_threshold")
    create_dir_if_missing(output_path)
    user2pred["user_id"] = user2pred["user_id"].astype(str)
    labeled_users["user_id"] = labeled_users["user_id"].astype(str)
    user2pred = user2pred[user2pred["user_id"].isin(list(labeled_users["user_id"]))].reset_index(drop=True)
    avg_hs_score_per_user = user2pred.groupby('user_id').agg({"predictions": "mean"}).reset_index() \
        .rename(columns={"predictions": "avg_hs_score"})
    avg_hs_score_per_user_with_true = pd.merge(labeled_users, avg_hs_score_per_user, on='user_id')

    hs_count_per_user = user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
        columns={"predictions": "hs_count"})

    res = pd.DataFrame(
        columns=["lower_bound", "higher_bound", "low_th", "medium_th", "high_th", "f1_score", "precision_score",
                 "recall_score", "accuracy_score"])

    for LOWER_BOUND in tqdm(np.linspace(0.1, 0.4, 1)):
        for HIGHER_BOUND in np.linspace(0.2, 0.6, 1):
            if LOWER_BOUND >= HIGHER_BOUND:
                continue
            for low_th in range(1, 10, 2):
                for medium_th in range(2, 50, 3):
                    for high_th in range(3, 300,2):
                        if low_th >= medium_th or low_th >= high_th or medium_th >= high_th:
                            continue
                        kwargs = {"LOWER_BOUND": LOWER_BOUND, "HIGHER_BOUND": HIGHER_BOUND,
                                  "low_th": low_th, "medium_th": medium_th, "high_th": high_th}
                        #                         avg_hs_score_per_user_with_true_copy = avg_hs_score_per_user_with_true.copy()
                        avg_hs_score_per_user_with_true[
                            f"soft_threshold_{LOWER_BOUND}_{HIGHER_BOUND}_{low_th}_{medium_th}_{high_th}"] = \
                        avg_hs_score_per_user_with_true["avg_hs_score"]. \
                            apply(lambda avg_hs_score: calc_soft_threhold(avg_hs_score, **kwargs))

    bound_cols = [c for c in avg_hs_score_per_user_with_true.columns if 'soft' in c]
    y_preds_cols = [f"y_pred_{b_col}" for b_col in bound_cols]
    avg_hs_score_per_user_with_true = pd.merge(avg_hs_score_per_user_with_true, hs_count_per_user, on='user_id')
    y_true = avg_hs_score_per_user_with_true["label"]
    def apply_soft_th_pred(col, hs_count):
        return hs_count >= col
    avg_hs_score_per_user_with_true[y_preds_cols] = avg_hs_score_per_user_with_true[bound_cols].\
        apply(lambda col: apply_soft_th_pred(col, avg_hs_score_per_user_with_true['hs_count']), axis=0)

    for col in tqdm(bound_cols):

        current_bound = col.split("soft_threshold_")[1]
        avg_hs_score_per_user_with_true[f"y_pred_{current_bound}"] = avg_hs_score_per_user_with_true.apply(
            lambda row: 1 if row["hs_count"] >= row[col] else 0, axis=1)

        y_pred = avg_hs_score_per_user_with_true[f"y_pred_{current_bound}"]

        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        scb = current_bound.split('_')
        res = res.append(
            {"lower_bound": scb[0], "higher_bound": scb[1], "low_th": scb[2], "medium_th": scb[3], "high_th": scb[4],
             "f1_score": f1, "precision_score": precision, "recall_score": recall,
             "accuracy_score": accuracy}, ignore_index=True)

    res.to_csv(os.path.join(output_path, "soft_threshold.csv"), index=False)
    return res


@timeit
def run_simple_ulm_experiments():
    # take the dataset to predict from config
    dataset = user_level_execution_config["inference_data"]

    logger.info(f"executing dataset {dataset}...")
    model_name = post_level_execution_config["kwargs"]["model_name"]  # new_bert_fine_tuning
    user2pred = pd.read_parquet(f"detection/outputs/{dataset}/{model_name}/user_level/split_by_posts/no_text/")
    user2label_path = user_level_conf[dataset]["data_path"]
    sep = ","
    if user2label_path.endswith("tsv"):
        sep = "\t"
    labeled_users = pd.read_csv(user2label_path, sep=sep)
    output_path = f"detection/outputs/{dataset}/{model_name}/user_level/"

    fixed_threshold_num_of_posts(user2pred, labeled_users, output_path, test_ratio=0.2)
    relational_threshold(user2pred, labeled_users, output_path, dataset, test_ratio=0.2)
    dynamic_threshold_hs_score(user2pred, labeled_users, output_path, test_ratio=0.2)

if __name__ == '__main__':
    run_simple_ulm_experiments()

