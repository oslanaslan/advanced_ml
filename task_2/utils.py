'''
Help functions
'''
import os
import gc
import sys
from collections import defaultdict

import pickle
from scipy.stats import kendalltau, spearmanr
import pandas as pd
import numpy as np
from tqdm import tqdm


DATA_PATH = 'data'
PLAYERS_DATA = os.path.join(DATA_PATH, 'players.pkl')
RESULTS_DATA = os.path.join(DATA_PATH, 'results.pkl')
TOURNAMENT_DATA = os.path.join(DATA_PATH, 'tournaments.pkl')
TRAIN_DATA_CSV = os.path.join(DATA_PATH, 'train_data.csv')
TEST_DATA_CSV = os.path.join(DATA_PATH, 'test_data.csv')
TRAIN_TOURNAMENTS = os.path.join(DATA_PATH, 'train_tournaments.pkl')
TEST_TOURNAMENTS = os.path.join(DATA_PATH, 'test_tournaments.pkl')


def read_data() -> tuple:
    '''
    Read data from json and split on train and test
    '''
    print('Reading data ...', file=sys.stderr)

    with open(TOURNAMENT_DATA, 'rb') as fin:
        tournaments = pickle.load(fin)
    with open(RESULTS_DATA, 'rb') as fin:
        results = pickle.load(fin)

    train_tournaments = {}
    test_tournaments = {}

    for idx, tournament_info in tqdm(tournaments.items()):
        try:
            if results[idx][0]["mask"] and len(results[idx][0]["teamMembers"]) > 0:
                if tournament_info["dateStart"][:4] == "2019":
                    train_tournaments[idx] = tournament_info
                elif tournament_info["dateStart"][:4] == "2020":
                    test_tournaments[idx] = tournament_info
        except IndexError as e:
            continue
        except KeyError as e:
            continue

    return train_tournaments, test_tournaments, results

def tables_gen(df: dict, results: dict) -> pd.DataFrame:
    '''
    Generate train and test tables with question levels and player correct ratio
    '''
    correct_ans_count = defaultdict(int)
    players_amount = defaultdict(int)
    question_level = defaultdict(int)

    print('Generating ratio and question level ...', file=sys.stderr)

    for tour_idx in tqdm(df):
        tour_ans_lst = []

        for info in results[tour_idx]:
            if not info["mask"]:
                continue

            cur_mask = info["mask"].replace('X', '0').replace('?', '0')
            cur_mask = list(map(int, cur_mask))
            tour_ans_lst.append(cur_mask)

            for players in info["teamMembers"]:
                correct_ans_count[players["player"]["id"]] += sum(cur_mask)
                players_amount[players["player"]["id"]] += len(cur_mask)

        is_all_same_length = len(set([len(itm) for itm in tour_ans_lst])) == 1

        if not is_all_same_length:
            continue

        tour_ans_lst = np.array(tour_ans_lst).sum(axis=0)
        tour_ans_lst = 1 - tour_ans_lst / len(results[tour_idx])
        question_level[tour_idx] = tour_ans_lst

    correct_ratio = {player_id: correct_ans_count[player_id] / players_amount[player_id] for player_id in correct_ans_count}

    print('Creating final table ...', file=sys.stderr)

    final_data_df = []

    for tour_idx, q_level in tqdm(question_level.items()):
        cmd_cnt = len(results[tour_idx])

        for info in results[tour_idx]:
            if not info["mask"]:
                continue

            cur_mask = info["mask"].replace('X', '0').replace('?', '0')
            cur_mask = list(map(int, cur_mask))
            position = info["position"] / cmd_cnt
            players_count = len(info['teamMembers'])

            for players in info["teamMembers"]:
                for idx, (cur_ans, cur_lvl) in enumerate(zip(cur_mask, q_level)):
                    final_data_df.append([
                        players["player"]["id"],
                        str(tour_idx) + "_" + str(idx),
                        str(tour_idx) + "_" + str(info['team']['id']),
                        correct_ratio[players["player"]["id"]],
                        cur_lvl,
                        position,
                        cmd_cnt,
                        1 / players_count,
                        sum(cur_mask) / len(cur_mask),
                        cur_ans
                    ])

    print('Creating final dataframe ...', file=sys.stderr)

    final_data_df = pd.DataFrame(
        final_data_df,
        columns=[
            'player_id',
            'tourn_question_id',
            'team_id',
            'correct_ratio',
            'question_comp',
            'team_position',
            'team_count',
            'pi',
            'theta',
            'answer'
        ]
    )

    return final_data_df

def get_corr_score(df: dict, preds_df: pd.DataFrame) -> tuple:
    '''
    Spearman and Kendall score
    '''
    print('Spearman and Kendall corr ...')

    spearman_lst = []
    kendall_lst = []
    preds_for_team_df = preds_df.groupby('player_id')['pred_proba'].mean()

    with open(RESULTS_DATA, 'rb') as fin:
        results = pickle.load(fin)

    for tour_idx in tqdm(df):
        cmds_power = []

        for info in results[tour_idx]:
            try:
                cmds_power.append(np.mean(list(map(
                    lambda x: preds_for_team_df[x["player"]["id"]], info["teamMembers"]
                ))))
            except KeyError:
                continue

        temp = np.array(cmds_power).argsort()[::-1]
        ranks = np.zeros_like(temp)
        ranks[temp] = np.arange(1, len(cmds_power) + 1)
        sp_coef = spearmanr(range(1, len(cmds_power) + 1), ranks)[0]
        kd_coef = kendalltau(range(1, len(cmds_power) + 1), ranks)[0]

        if not np.isnan(sp_coef):
            spearman_lst.append(sp_coef)
        if not np.isnan(kd_coef):
            kendall_lst.append(kd_coef)

    return spearman_lst, kendall_lst

def em_step(df):
    '''
    Em-algo step
    '''
    def get_z_hidden(pi_vec, theta_vec, x_vec, players_cnt, questions_cnt):
        prob_vec = np.multiply(theta_vec, x_vec) + \
            np.multiply(
                (np.ones_like(theta_vec, dtype=float) - theta_vec),
                (np.ones_like(x_vec, dtype=float) - x_vec)
            )
        numerators = np.multiply(prob_vec, pi_vec).reshape(questions_cnt, players_cnt)
        denominators = numerators.sum(axis=1).reshape(-1, 1)

        return numerators / denominators

    def expectation(df, players_cnt, questions_cnt):
        z_hidden = get_z_hidden(
            df["pi"].values, 
            df["theta"].values,
            df["answer"].values,
            players_cnt,
            questions_cnt
        )

        return z_hidden

    def maximization(z_matrix, df, questions_cnt, players_cnt):
        z_sum = z_matrix.sum(axis=0)
        new_pi_vec = np.tile((z_sum / z_matrix.shape[0]).reshape(-1), questions_cnt)
        new_theta_vec = np.tile((df["answer"].values[:questions_cnt].reshape(1, -1) @ z_matrix) / \
            (z_sum.reshape(-1, 1)).reshape(-1), questions_cnt).reshape(-1)

        return new_pi_vec, new_theta_vec

    players_cnt = len(pd.unique(df["player_id"]))
    questions_cnt = len(pd.unique(df["tourn_question_id"]))
    z_matrix = expectation(df, players_cnt, questions_cnt)
    new_pi_vec, new_theta_vec = maximization(z_matrix, df, questions_cnt, players_cnt)
    df["pi"] = new_pi_vec
    df["theta"] = new_theta_vec

    return df

def create_tables():
    '''
    Help function for creating CSV train and test tables
    '''
    train_tournaments, test_tournaments, results = read_data()
    with open(TRAIN_TOURNAMENTS, 'wb') as fout:
        pickle.dump(train_tournaments, fout)
    with open(TEST_TOURNAMENTS, 'wb') as fout:
        pickle.dump(train_tournaments, fout)
    gc.collect()
    print('Start creating train data ...', file=sys.stderr)
    train_data_df = tables_gen(train_tournaments, results)
    gc.collect()
    print('Saving train dataframe to CSV ...', file=sys.stderr)
    train_data_df.to_csv(TRAIN_DATA_CSV, index=False)
    del train_data_df
    del train_tournaments
    gc.collect()
    print('Start creating test data ...', file=sys.stderr)
    test_data_df = tables_gen(test_tournaments, results)
    gc.collect()
    print('Saving test dataframe to CSV ...', file=sys.stderr)
    test_data_df.to_csv(TEST_DATA_CSV, index=False)
    del test_data_df
    del test_tournaments
    gc.collect()


if __name__ == '__main__':
    create_tables()
    