import json
import numpy as np
import sys
from .utils import load_summEval, load_newsroom, load_TopicalChat
from .sorting import merge_sort_indices
from .utils import calculate_correlation
from tqdm import tqdm
import os
import pandas as pd


def load_matrix(matrix_path):
    with open(matrix_path, "r") as f:
        matrix = json.load(f)
    return matrix


def load_rsults(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
    return results


def get_prob_gap(results):
    prob_gap_list = []
    norm_prob_gap_list = []
    cal_list = [[], []]
    for data_id, id_result in results.items():
        for key, values in id_result.items():
            if key == "prob_list":
                for v in values:
                    prob_gap_list.append(np.abs(v[0] - v[1]))
            if key == "norm_prob_list":
                for v in values:
                    norm_prob_gap_list.append(np.abs(v[0] - v[1]))
            if key == "logit_list":
                for v in values:
                    cal_list[0].append(v[0])
                    cal_list[1].append(v[1])
    base_prior = np.mean([np.mean(cal) for cal in cal_list])
    bc_prior = np.mean([np.abs(base_prior - np.mean(cal)) for cal in cal_list])
    return np.mean(prob_gap_list), np.mean(norm_prob_gap_list), bc_prior


def convert_to_small_matrix(matrix):
    rows, cols = matrix.shape
    mask = np.ones(matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    return matrix[mask].reshape(rows, cols - 1)


def get_a_rate(args, preference_matrixs):
    if args.do_permutate:
        A_rate_list = []
        for matrix in preference_matrixs:
            matrix = np.triu(matrix)
            small_matrix = convert_to_small_matrix(np.array(matrix))
            A_count = np.sum(np.logical_and(small_matrix > 0.5, small_matrix != 0))
            B_count = np.sum(np.logical_and(small_matrix < 0.5, small_matrix != 0))
            A_rate = A_count / (A_count + B_count)
            A_rate_list.append(A_rate)
    else:
        A_rate_list = []
        for matrix in preference_matrixs:
            small_matrix = convert_to_small_matrix(np.array(matrix))
            A_count = np.sum(small_matrix > 0.5)
            B_count = np.sum(small_matrix < 0.5)
            A_rate = A_count / (A_count + B_count)
            A_rate_list.append(A_rate)

    print("Average A rate:", np.mean(A_rate_list))
    return np.mean(A_rate_list)


def compute_spearman(args, preference_matrixs, scores_doc):
    params = {
        # 'dataset': args.dataset,
        # 'engine': "meta-llama/Llama-2-7b-chat-hf",
        # 'engine': "meta-llama/Llama-2-13b-chat-hf",
        "engine": "mistralai/Mistral-7B-Instruct-v0.1",
        # 'engine': 'gpt-3.5-turbo',
        # 'aspect': args.aspect,
        # 'eval_method': args.eval_method,
        "confidence_beam": False,
        "beam_size": 1,
        "api_call": 0,
        "prob_gap": 0.1,
        # 'with_input': args.with_input,
        # 'calibration': args.calibration,
        # 'compare_log': {},
    }

    if args.dataset == "SummEval":
        n_candidate = 16
    elif args.dataset == "newsroom":
        n_candidate = 7
    else:
        n_candidate = 5
    repeat_times = 100
    spearman_log = []
    tau_log = []
    total_comparison_log = []
    for _ in range(repeat_times):
        spearman_list = []
        tau_list = []
        params["api_call"] = 0
        for i, _ in enumerate(preference_matrixs):
            ranking_indices = merge_sort_indices(
                preference_matrixs[i], params, permutate=args.do_permutate
            )
            rho, tau = calculate_correlation(
                list(range(n_candidate)), scores_doc[i][ranking_indices]
            )
            spearman_list.append(rho)
            tau_list.append(tau)

        spearman_log.append(spearman_list)
        tau_log.append(tau_list)
        total_comparison_log.append(params["api_call"] / len(preference_matrixs))

    spearman_avg = np.mean(spearman_log)
    spearman_std = np.std(np.mean(spearman_log, axis=-1))
    comparison_avg = np.mean(total_comparison_log)
    a_rate = get_a_rate(args, preference_matrixs)
    print(
        f"spearman_avg: {spearman_avg}, spearman_std: {spearman_std}, comparison_avg: {comparison_avg}, A rate: {a_rate}"
    )
    return spearman_avg, spearman_std, comparison_avg, a_rate


def get_corr_df(args, saving_dir, test_list_id=range(0, 10)):
    if args.dataset == "SummEval":
        SummEval_path = "./data/model_annotations.aligned.paired.jsonl"
        input_doc, output_doc, scores_doc = load_summEval(
            SummEval_path, flat_output=False, truncate_num_for_eval=args.eval_data_num
        )
    elif args.dataset == "newsroom":
        newsroom_path = "./data/newsroom/newsroom.json"
        input_doc, output_doc, scores_doc = load_newsroom(
            newsroom_path, flat_output=False, truncate_num_for_eval=args.eval_data_num
        )
    elif args.dataset == "TopicalChat":
        TC_path = "data/topicalchat_usr.json"
        input_doc, output_doc, scores_doc = load_TopicalChat(
            TC_path, truncate_num_for_eval=args.eval_data_num
        )
    scores_doc = np.array(scores_doc[args.aspect_name])
    scores_doc = np.round(scores_doc, 1)
    collect_corr = []
    collect_a_rate = []
    collect_norm_prob_gap = []
    collect_bc = []
    for test_id in test_list_id:
        print(f"Test id: {test_id}")
        test_matrix = load_matrix(
            f"{saving_dir}_preference_matrix_log_cot_False_{args.aspect_name}_{test_id}.json"
        )
        if args.do_permutate:
            test_matrix = [
                (np.array(matrix) + 1 - np.array(matrix).T) / 2
                for matrix in test_matrix
            ]
        results = compute_spearman(args, test_matrix, scores_doc)
        collect_corr.append(results[0])
        collect_a_rate.append(results[3])
        prob_results = load_rsults(
            f"{saving_dir}_compare_result_log_cot_False_{args.aspect_name}_{test_id}.json"
        )
        prob_gap, norm_prob_gap, bc = get_prob_gap(prob_results)
        collect_norm_prob_gap.append(norm_prob_gap)
        collect_bc.append(bc)

    mod_a = [abs(a - 0.5) for a in collect_a_rate]
    df = pd.DataFrame(
        {
            "Test id": test_list_id,
            "Spearman": collect_corr,
            "A rate": collect_a_rate,
            "Norm Gap": collect_norm_prob_gap,
            "BC": collect_bc,
            "Fairness": [-1 * a for a in mod_a],
        }
    )
    df = df.assign(task="PairS")
    return df
