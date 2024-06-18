from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import torch.nn.functional as F
import sys

sys.path.append("../")
from utils import CompareResultObject, calculate_uncertainty


device = "cuda"


def is_integer_string(s):
    return s.isdigit()


class Llama2ModelLocal:
    def __init__(self, params):
        self.model_name = params["model"]
        self.temperature = params["temperature"] if "temperature" in params else 0
        self.max_tokens = params["max_tokens"] if "max_tokens" in params else 64
        self.do_sample = params["do_sample"] if "do_sample" in params else False
        self.device = device
        if "cache_dir" not in params:
            params["cache_dir"] = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        load_in_8bit = True if "13" in self.model_name else False
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            load_in_8bit=load_in_8bit,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        self.A_ids = self.tokenizer.convert_tokens_to_ids(["A", "▁A"])  # A: 330
        self.B_ids = self.tokenizer.convert_tokens_to_ids(["B", "▁B"])  # B: 365
        self.C_ids = self.tokenizer.convert_tokens_to_ids(["C", "▁C"])  # C:
        self.score_ids = self.tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])

    def rate_score(self, prompt):
        sequence, output = self.local_model_chat_completion(prompt)
        score, logprobs = self.extract_score(sequence, output.logits)
        print(score)
        return score, logprobs

    def compare(self, prompt) -> CompareResultObject:
        # model run locally
        # print(prompt)
        sequence, output = self.local_model_chat_completion(prompt)
        # print(self.tokenizer.batch_decode(sequence))
        compare_result = self.extract_probs(sequence, output.logits)
        return compare_result

    def generate(self, prompt, chat_system_instruction=None, num_samples=1):
        sequence, output = self.local_model_chat_completion(
            prompt, chat_system_instruction, num_samples
        )
        # Skip special tokens and role tokens
        generated_text = self.tokenizer.batch_decode(
            sequence[:, 1:], skip_special_tokens=True
        )
        return generated_text

    def extract_score(self, sequence, logits):
        """
        sequence: [batch_size, seq_len]
        logits: seq_len x [batch_size, vocab_size]
        output: int score
        """
        for idx, token_id in enumerate(sequence[0]):
            logit = logits[idx][0]
            logprobs = F.log_softmax(logit, dim=-1).cpu()
            score_logprobs = logprobs[self.score_ids].tolist()
            token = self.tokenizer.decode(token_id)
            if is_integer_string(token):
                return int(token), score_logprobs
        print("Failed to extract score")
        print(self.tokenizer.batch_decode(sequence))
        return 3, [np.log(0.2)] * 5

    def extract_probs(self, sequence, logits) -> CompareResultObject:
        """
        sequence: [batch_size, seq_len]
        logits: seq_len x [batch_size, vocab_size]
        output: compare_result_object
        """
        # First token logit
        for idx, token_id in enumerate(sequence[0]):
            if token_id in self.A_ids or token_id in self.B_ids:
                logit = logits[idx]
                probs = F.softmax(logit, dim=-1)[0]
                prob_A = sum([probs[a_id].item() for a_id in self.A_ids])
                prob_B = sum([probs[b_id].item() for b_id in self.B_ids])
                prob_C = sum([probs[c_id].item() for c_id in self.C_ids])
                uncertainty = calculate_uncertainty([prob_A, prob_B])
                compare_result = CompareResultObject(
                    raw_prob_A=prob_A,
                    raw_prob_B=prob_B,
                    raw_prob_C=prob_C,
                    uncertainty=uncertainty,
                )
                return compare_result
        print("Failed to extract probs")
        print(self.tokenizer.batch_decode(sequence))
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)

    def local_model_chat_completion(
        self, prompt, chat_system_instruction=None, num_samples=1
    ):

        msg = Llama2ModelLocal.get_chat_message(prompt, chat_system_instruction)
        input = self.tokenizer.apply_chat_template(
            msg, return_tensors="pt", return_dict=True
        )

        input = {k: v.expand(num_samples, -1) for k, v in input.items()}
        input_ids = input["input_ids"].to(device)

        output = self.model.generate(
            inputs=input_ids,
            return_dict_in_generate=True,
            output_logits=True,
            max_new_tokens=self.max_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )

        newly_generated_tokens = output.sequences[:, input_ids.shape[-1] :]
        return newly_generated_tokens, output

    @staticmethod
    def get_chat_message(prompt, chat_system_instruction=None):
        if chat_system_instruction:
            message = [
                {"role": "system", "content": chat_system_instruction},
                {"role": "user", "content": prompt},
            ]
        else:
            message = [{"role": "user", "content": prompt}]
        return message


if __name__ == "__main__":
    model = Llama2ModelLocal({"model": "meta-llama/Llama-2-7b-chat-hf"})
