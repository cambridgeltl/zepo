from typing import List
from models.openai_api import OpenAIChatModel
from pairwise_comparison import pairwise_compare
import pairs
import argparse
import wandb
import numpy as np
import json
import os


openai_api_key = os.environ.get("OPENAI_API_KEY")


def get_instruction(args: argparse.Namespace, _instruction: str, iteration: int = 5) -> List[str]:
    """
    Generate paraphrased instructions using OpenAI API.

    Args:
        args: command line arguments including aspect name.
        _instruction: initial instruction to paraphrase.
        iteration: number of paraphrased instructions to generate.

    Returns:
        List of paraphrased instructions.
    """
    example_prompt = f"""\
    Paraphrase the following instruction for a pairwise comparison task. Do not change the keyword "{args.aspect_name}". Be diverse and creative in paraphrasing. Return the instruction only. \

    Input: {_instruction}\

    Output: 
    """
    model = OpenAIChatModel(
        {"engine": "gpt-3.5-turbo", "temperature": 0.9}, api_key=openai_api_key
    )
    prompts = [example_prompt] * iteration
    results = model.generate(prompts)
    return results


def zepo(args: argparse.Namespace):
    # Initialize variables
    optimize_metric = "Fairness"

    init_instruction_dict = json.load(open("init_prompts.json"))

    init_instruction = init_instruction_dict[args.aspect_name]
    instruction_set = get_instruction(
        args, init_instruction, iteration=args.sample_num - 1
    )
    instruction_set = [init_instruction] + instruction_set
    collect_instruction = []
    collect_instruction += instruction_set
    collect_results = {}
    log_new_instruction = []
    print("Initial instruction set: ", instruction_set)
    best_metric = -99
    best_corr = 0
    wandb.init(
        project="zepo",
        config={
            "dataset": args.dataset,
            "aspect_name": args.aspect_name,
            "engine": args.engine,
            "batch_size": args.batch_size,
            "sample_num": args.sample_num,
            "eval_data_num": args.eval_data_num,
            "epoch_num": args.epoch_num,
            "instruction": init_instruction,
            "instruction_set": instruction_set,
            "best_metric": best_metric,
            "best_corr": best_corr,
        },
    )

    # Optimize instructions over multiple epochs
    for epoch in range(args.epoch_num):
        args.saving_dir = f"results/{args.engine}/permutate_{args.do_permutate}/{args.dataset}/{args.aspect_name}/{epoch}/"

        # Evaluation instructions in pairwise comparisons
        pairwise_compare(args, instruction_set, round_id=epoch)
        saving_dir = f"results/{args.engine}/permutate_{args.do_permutate}/{args.dataset}/{args.aspect_name}/{epoch}/"
        saving_path = f"{saving_dir}{args.engine.split('/')[-1]}"

        # Retrieve fairness
        df = pairs.pairs_eval.get_corr_df(
            args, saving_path, test_list_id=range(0, args.sample_num)
        )
        df.to_csv(f"{saving_dir}{args.engine.split('/')[-1]}_results.csv")
        print(df)
        best_id = df[optimize_metric].idxmax()
        new_metric = df[optimize_metric].max()

        # Greedy selection of the best instruction
        if new_metric > best_metric:
            best_metric = new_metric
            new_instruction = instruction_set[best_id]
        log_new_instruction.append(new_instruction)
        wandb.log({})
        print(f"Best instruction: ", new_instruction)
        new_corr = df["Spearman"][best_id]
        if new_corr > best_corr:
            best_corr = new_corr
        print("Best Correlation: ", best_corr)

        # Generate new set of instructions for the next epoch
        if epoch != args.epoch_num - 1:
            instruction_set = get_instruction(
                args, new_instruction, iteration=args.sample_num
            )
            print(f"New instruction set at epoch {epoch+1}: ", instruction_set)
            collect_instruction += instruction_set

        wandb.log(
            {
                "best_corr": best_corr,
                "best_metric": best_metric,
                "instruction": new_instruction,
                "instruction_set": instruction_set,
                "epoch": epoch,
            }
        )

    # evaluate the final instruction
    print("Final instruction: ", new_instruction)
    args.eval_data_num = 100
    epoch = "final"
    args.saving_dir = f"results/{args.engine}/permutate_{args.do_permutate}/{args.dataset}/{args.aspect_name}/{epoch}/"
    pairwise_compare(args, [new_instruction], round_id=epoch)
    saving_path = f"{args.saving_dir}{args.engine.split('/')[-1]}"
    df = pairs.pairs_eval.get_corr_df(args, saving_path, test_list_id=[0])
    best_id = df[optimize_metric].idxmax()
    print(f"Best instruction id: {best_id}")
    best_corr = df["Spearman"][best_id]
    print("Best Correlation: ", best_corr)
    collect_results["test corr"] = best_corr
    collect_results["instruction set"] = collect_instruction
    collect_results["final instruction"] = new_instruction
    collect_results["log best instruction"] = log_new_instruction

    # compare with the initial instruction
    epoch = "init"
    args.saving_dir = f"results/{args.engine}/permutate_{args.do_permutate}/{args.dataset}/{args.aspect_name}/{epoch}/"
    pairwise_compare(args, [init_instruction], round_id=epoch)
    saving_path = f"{args.saving_dir}{args.engine.split('/')[-1]}"
    df = pairs.pairs_eval.get_corr_df(args, saving_path, test_list_id=[0])
    best_id = df[optimize_metric].idxmax()
    best_corr = df["Spearman"][best_id]
    print("Init Correlation: ", best_corr)
    collect_results["init corr"] = best_corr

    # Save the file
    saving_path = f"{args.saving_dir}{args.engine.split('/')[-1]}_{args.aspect_name}_{args.eval_data_num}_{args.sample_num}_{args.epoch_num}_results.json"
    with open(saving_path, "w") as f:
        json.dump(collect_results, f, indent=4)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SummEval")
    parser.add_argument("--aspect_name", type=str, default="coherence")
    parser.add_argument(
        "--engine", type=str, default="mistralai/Mistral-7B-Instruct-v0.1"
    )
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--sample_num", type=int, default=5)
    parser.add_argument("--eval_data_num", type=int, default=5)
    parser.add_argument("--epoch_num", type=int, default=5)
    parser.add_argument("--do_cot", action="store_true", default=False)
    parser.add_argument("--do_permutate", action="store_true", default=False)
    parser.add_argument("--saving_dir", type=str, default="results/")

    args = parser.parse_args()
    zepo(args)
