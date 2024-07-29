import numpy as np
import json
from utils import load_TopicalChat, load_summEval, load_gsm8k, load_newsroom
from tqdm import tqdm
from prompts import (
    get_pairwise_prompt_template,
    get_cot_compare_prompt_template,
    get_cot_eval_prompt_template,
)
from jinja2 import Environment
from models.llama2 import Llama2ModelLocal
from models.llama3 import Llama3ModelLocal
from models.mistral import MistralModelLocal
from models.openai_api import OpenAIChatModel
from models.local_model import LocalModel
import os


def pairwise_non_diagonal_to_list(size):
    """
    Convert a Square pairwise matrix to a list of non-diagonal elements
    """
    rows, cols = size, size
    result = []
    for i in range(rows):
        for j in range(cols):
            if i != j:  # Check if the element is not on the diagonal
                result.append((i, j))
    return result


def list_to_pairwise_non_diagonal(non_diagonal_list, size):
    """
    Convert a list of non-diagonal elements back to a pairwise matrix
    Square matrix
    """
    rows, cols = size, size
    matrix = [[0] * cols for _ in range(rows)]
    index = 0
    for i in range(rows):
        for j in range(cols):
            if i != j:  # Check if the element is not on the diagonal
                matrix[i][j] = non_diagonal_list[index]
                index += 1
    return np.array(matrix)


def compute_pairwise_preference_matrix(
    model,
    input,
    output,
    prompt_templates,
    do_cot=False,
    worker_num=1,
    batch_size=1,
    instruction=None,
):
    """
    worker_num is for closed source models async parallel processing
    batch_size is for open source models parallel processing
    """
    response_size = len(output)
    full_pairwise_list = pairwise_non_diagonal_to_list(size=response_size)

    prompts = []
    count = 0
    for pair in full_pairwise_list:
        if instruction:
            prompt = prompt_templates[0].render(
                input=input,
                output_1=output[pair[0]],
                output_2=output[pair[1]],
                instruction=instruction,
            )
        else:
            prompt = prompt_templates[0].render(
                input=input,
                output_1=output[pair[0]],
                output_2=output[pair[1]],
            )
        if count == 0:
            # print('Prompt:', prompt)
            count += 1
        prompts.append(prompt)

    # print(prompts[0])
    # print(prompts[1])
    # assert False
    # If CoT, generate first
    if do_cot:
        model.max_tokens = 256
        if worker_num > 1:  # Parallel processing for closed source models
            prompts = model.generate(prompts)
        else:
            analysis = []
            for i in tqdm(range(0, len(prompts), batch_size)):
                batch_prompts = prompts[i : i + batch_size]
                analysis.extend(model.generate(batch_prompts))
        eval_promtps = [
            prompt_templates[1].render(cot_response=decoded_sequence)
            for decoded_sequence in analysis
        ]
        prompts = eval_promtps

    model.max_tokens = 32
    if worker_num > 1:  # Parallel processing for closed source models
        compare_result_list = model.compare(prompts, max_workers=worker_num)
    else:
        compare_result_list = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i : i + batch_size]
            compare_results = model.compare(batch_prompts)
            compare_result_list.extend(compare_results)

    pairwise_preference_list = (
        []
    )  # [compare_result.prob_A for compare_result in compare_result_list]
    norm_prob_list = []
    logit_list = []
    prob_list = []
    for compare_result in compare_result_list:
        norm_prob_list.append(
            [compare_result.prob_A, compare_result.prob_B, compare_result.prob_C]
        )
        logit_list.append(
            [compare_result.logit_A, compare_result.logit_B, compare_result.logit_C]
        )
        prob_list.append(
            [
                compare_result.raw_prob_A,
                compare_result.raw_prob_B,
                compare_result.raw_prob_C,
            ]
        )
        if (
            compare_result.prob_C > compare_result.prob_A
            and compare_result.prob_C > compare_result.prob_B
        ):
            pairwise_preference_list.append(0.5)
        else:
            pairwise_preference_list.append(compare_result.prob_A)

    # Count if prob is balanced
    A_cnt, B_cnt, C_cnt = 0, 0, 0
    for prob in pairwise_preference_list:
        if prob > 0.5:
            A_cnt += 1
        elif prob < 0.5:
            B_cnt += 1
        else:
            C_cnt += 1
    print("A_cnt:", A_cnt, "B_cnt:", B_cnt, "C_cnt:", C_cnt)

    pairwise_preference_matrix = list_to_pairwise_non_diagonal(
        pairwise_preference_list, response_size
    )
    log_results = {
        "norm_prob_list": norm_prob_list,
        "logit_list": logit_list,
        "prob_list": prob_list,
    }
    return pairwise_preference_matrix, log_results


def pairwise_compare(args, instruction_list, round_id):
    # Load datasets
    aspect_name = args.aspect_name
    if args.dataset == "TopicalChat":
        TC_path = "data/topicalchat_usr.json"
        input_doc, output_doc, scores = load_TopicalChat(
            TC_path, truncate_num_for_eval=args.eval_data_num
        )
        scores = scores[args.aspect_name]

    elif args.dataset == "SummEval":
        SummEval_path = "data/model_annotations.aligned.paired.jsonl"
        input_doc, output_doc, scores = load_summEval(
            SummEval_path, flat_output=False, truncate_num_for_eval=args.eval_data_num
        )
        scores = scores[args.aspect_name]

    elif args.dataset == "newsroom":
        newsroom_path = "data/newsroom/newsroom.json"
        input_doc, output_doc, scores = load_newsroom(
            newsroom_path, flat_output=False, truncate_num_for_eval=args.eval_data_num
        )
        scores = scores[args.aspect_name]

    elif args.dataset == "GSM8k":
        GSM8k_path = "gsm8k_augment/{}_test_responses.jsonl".format(
            args.engine.split("/")[-1]
        )
        input, output_doc = load_gsm8k(GSM8k_path, cot=False)
        input_doc = [[i] for i in input]
        response_size = len(output_doc[0])

    # Load model
    if "mistral" in args.engine:
        model = MistralModelLocal({"model": args.engine})
    elif "Llama-3" in args.engine:
        model = Llama3ModelLocal({"model": args.engine, "cot": args.do_cot})
    elif "Llama-2" in args.engine:
        model = Llama2ModelLocal({"model": args.engine})
    elif "gpt" in args.engine:
        model = OpenAIChatModel({"model": args.engine})
    elif "gemma" in args.engine:
        model = LocalModel({'model': args.engine, 
                    'temperature': 0, 
                    'do_sample': False,
                    'qa_style': False,
                    'max_tokens': 32})
    saving_dir = args.saving_dir
    # Load prompt template
    for i_id, instruction in enumerate(instruction_list):
        if i_id > -1 and i_id < 21:
            if args.do_cot:
                prompt_template = get_cot_compare_prompt_template(dataset=args.dataset)
                cot_eval_template = get_cot_eval_prompt_template()
                print("Prompt template:", cot_eval_template)
                environment = Environment()
                prompt_template = environment.from_string(prompt_template)
                environment = Environment()
                cot_eval_template = environment.from_string(cot_eval_template)
                prompt_templates = [prompt_template, cot_eval_template]
            else:
                prompt_template = get_pairwise_prompt_template(
                    dataset=args.dataset, use_instruction=True
                )
                print("Prompt template:", prompt_template)
                environment = Environment()
                prompt_template = environment.from_string(prompt_template)
                prompt_templates = [prompt_template]
            pairwise_preference_matrix_log = []
            compare_result_log = {}
            for i in range(len(input_doc)):
                print("Data point:", i + 1, "out of", len(input_doc), "data points.")
                input = input_doc[i][0]
                output = output_doc[i]
                pairwise_preference_matrix, compare_results = (
                    compute_pairwise_preference_matrix(
                        model,
                        input,
                        output,
                        prompt_templates,
                        do_cot=args.do_cot,
                        worker_num=args.worker_num,
                        batch_size=args.batch_size,
                        instruction=instruction,
                    )
                )
                pairwise_preference_matrix_log.append(
                    pairwise_preference_matrix.tolist()
                )
                compare_result_log[i] = compare_results
            print("Model: ", args.engine)

            if not os.path.exists(saving_dir):
                os.makedirs(saving_dir)
            saving_path = f"{saving_dir}{args.engine.split('/')[-1]}_preference_matrix_log_cot_{args.do_cot}_{aspect_name}_{i_id}.json"
            with open(saving_path, "w") as f:
                json.dump(pairwise_preference_matrix_log, f, indent=4)
            f.close()
            saving_path = f"{saving_dir}{args.engine.split('/')[-1]}_compare_result_log_cot_{args.do_cot}_{aspect_name}_{i_id}.json"
            with open(saving_path, "w") as f:
                json.dump(compare_result_log, f, indent=4)
            f.close()
    saving_path = (
        f"{saving_dir}{args.engine.split('/')[-1]}_instruction_set_{aspect_name}.json"
    )
    with open(saving_path, "w") as f:
        json.dump(instruction_list, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SummEval")
    parser.add_argument(
        "--engine", type=str, default="mistralai/Mistral-7B-Instruct-v0.1"
    )
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--do_cot", action="store_true", default=False)
    parser.add_argument("--aspect_name", type=str, default="coherence")
    parser.add_argument("--eval_data_num", type=int, default=10)

    args = parser.parse_args()

    if args.engine == "full":
        engine_list = [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "gpt-3.5-turbo",
        ]
    else:
        engine_list = [args.engine]

    results_to_report = []
    for engine in engine_list:
        args.engine = engine
        print("Engine: ", engine)
        if "gpt" in engine:
            args.worker_num = 8
        else:
            args.worker_num = 1
        pairwise_compare(args)
