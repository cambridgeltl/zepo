import sys
sys.path.append('../')

from tqdm import tqdm
import numpy as np
from utils import load_summEval
from prompts import get_prompt_template, get_aspect_instruction
import json
import torch
from utils import load_summEval
from llama2 import Llama2ModelLocal
from mistral import MistralModelLocal
from openai_api import OpenAIChatModel
from jinja2 import Environment



def pairwise_non_diagonal_to_list(size):
    '''
    Convert a Square pairwise matrix to a list of non-diagonal elements
    '''
    rows, cols = size, size
    result = []
    for i in range(rows):
        for j in range(cols):
            if i != j:  # Check if the element is not on the diagonal
                result.append((i, j))
    return result


def list_to_pairwise(non_diagonal_list, size, include_diagonal=False):
    '''
    Convert a list of non-diagonal elements back to a pairwise matrix
    Square matrix
    '''
    rows, cols = size, size
    matrix = [[0] * cols for _ in range(rows)]
    index = 0
    for i in range(rows):
        for j in range(cols):
            if i != j or include_diagonal:
            # if i != j:  # Check if the element is not on the diagonal
                matrix[i][j] = non_diagonal_list[index]
                index += 1
    return np.array(matrix)


def compute_pairwise_preference_matrix(model, input, output, prompt_template, worker_num=1, batch_size=1):
    '''
    worker_num is for closed source models async parallel processing
    batch_size is for open source models parallel processing
    '''
    task_instruction = get_aspect_instruction('coherence', eval_method='pairwise comparison', dataset='SummEval')

    response_size = len(output)
    full_pairwise_list = pairwise_non_diagonal_to_list(size=response_size)
    mask_matrix = np.ones((response_size, response_size))
    mask_matrix = np.triu(mask_matrix, k=1).astype(bool)
    mask_matrix_list = mask_matrix.flatten().tolist()

    prompts = []
    for pair, mask in zip(full_pairwise_list, mask_matrix_list):
        # if mask == False:
        #     continue
        prompt = prompt_template.render(
            instruction=task_instruction,
            input=input,
            output_1=output[pair[0]],
            output_2=output[pair[1]],
            aspect="coherence",
        )
        prompts.append(prompt)
    # print(len(prompts))
    # assert False

    if worker_num > 1:  # Parallel processing for closed source models
        compare_result_list = model.compare(prompts)
    else:
        compare_result_list = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            compare_results = model.compare(batch_prompts)
            compare_result_list.extend(compare_results)

    pairwise_preference_list = [compare_result.prob_A for compare_result in compare_result_list]


    # Count if prob is balanced
    A_cnt,B_cnt,C_cnt = 0,0,0
    for prob in pairwise_preference_list:
        if prob > 0.5:
            A_cnt +=1
        elif prob < 0.5:
            B_cnt +=1
        else:
            C_cnt +=1
    print('A_cnt:', A_cnt, 'B_cnt:', B_cnt, 'C_cnt:', C_cnt)

    # prompt_idx = 0
    # full_pairwise_preference_list = []
    # for mask in mask_matrix_list:
    #     if mask == True:
    #         full_pairwise_preference_list.append(pairwise_preference_list[prompt_idx])
    #         prompt_idx += 1
    #     else:
    #         full_pairwise_preference_list.append(0)
    # pairwise_preference_list = full_pairwise_preference_list

    pairwise_preference_matrix = list_to_pairwise(pairwise_preference_list, response_size, include_diagonal=False)
    return pairwise_preference_matrix


def main(args):
    if args.dataset == 'SummEval':
        SummEval_path = '/home/yinhong/Documents/source/PairS/data/SummEval/model_annotations.aligned.paired.jsonl'
        input_doc, output_doc, scores = load_summEval(SummEval_path, flat_output=False)
        scores = scores['coherence']

    # Load model
    if 'mistral' in args.engine:
        model = MistralModelLocal({'model': args.engine})
    # elif 'Llama-3' in args.engine:
    #     model = Llama3ModelLocal({'model': args.engine, 'cot': args.do_cot})
    elif 'Llama-2' in args.engine:
        model = Llama2ModelLocal({'model': args.engine})
    elif 'gpt' in args.engine:
        model = OpenAIChatModel({'model': args.engine})

    prompt_template = get_prompt_template(
            prompt_name="pairwise comparison", 
            aspect='coherence', 
            dataset=args.dataset, 
            model_name=None, 
            with_input=True
        )
    environment = Environment()
    prompt_template = environment.from_string(prompt_template)


    # intrans_list, trans_list = [], []
    pairwise_preference_matrix_log = []
    # preference_matrix_list = []
    for i in range(len(input_doc)):
        print('Data point:', i+1, 'out of', len(input_doc), 'data points.')
        input = input_doc[i][0]
        output = output_doc[i]
        # score = scores[i]
        pairwise_preference_matrix = compute_pairwise_preference_matrix(
                model, 
                input, 
                output, 
                prompt_template, 
                worker_num=args.worker_num,
                batch_size=args.batch_size
            )
        pairwise_preference_matrix_log.append(pairwise_preference_matrix.tolist())


    # Release the model from GPU
    del model
    torch.cuda.empty_cache()

    # Save list of matrices to JSONL file
    with open(f'{args.engine.split('/')[-1]}_preference_matrix_log.json', 'w') as f:
        json.dump(pairwise_preference_matrix_log, f)
    f.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SummEval')
    parser.add_argument('--engine', type=str, default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--worker_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--do_cot', action='store_true')  
    args = parser.parse_args()

    # with open(f'{args.engine.split('/')[-1]}_preference_matrix_log.json', 'w') as f:
    #     pass
    # f.close()

    if args.engine == 'full':
        engine_list = [
            # 'mistralai/Mistral-7B-Instruct-v0.1', 
            # 'meta-llama/Llama-2-7b-chat-hf', 
            # 'meta-llama/Llama-2-13b-chat-hf', 
            # 'meta-llama/Meta-Llama-3-8B-Instruct', 
            'gpt-3.5-turbo'
            ]
    else:
        engine_list = [args.engine]

    results_to_report = []
    for engine in engine_list:
        args.engine = engine
        print('Engine: ', engine)
        if 'gpt' in engine:
            args.worker_num = 6
        else:
            args.worker_num = 1
        # trans, intrans = 
        main(args)
        # report = {
        #     'engine': engine,
        #     'dataset': args.dataset,
        #     'transitivity': trans,
        #     'intransitivity': intrans
        # }
        # results_to_report.append(report)
    print('==========================================================')
    
    # for report in results_to_report:
    #     print(report)