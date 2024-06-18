from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F
import sys

sys.path.append("../")
from utils import CompareResultObject, calculate_uncertainty


device = "cuda"


def is_integer_string(s):
    return s.isdigit()


class MistralModelLocal:
    def __init__(self, params):
        self.model_name = params["model"]
        self.temperature = params["temperature"] if "temperature" in params else 0
        self.max_tokens = params["max_tokens"] if "max_tokens" in params else 128
        self.do_sample = params["do_sample"] if "do_sample" in params else False
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            attn_implementation="flash_attention_2",  # flash attention is not easy to install
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )  # , cache_dir="models")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.A_ids = self.tokenizer.convert_tokens_to_ids(["A", "▁A"])  # A: 330
        self.B_ids = self.tokenizer.convert_tokens_to_ids(["B", "▁B"])  # B: 365
        self.C_ids = self.tokenizer.convert_tokens_to_ids(["C", "▁C"])  # C:
        self.score_ids = self.tokenizer.convert_tokens_to_ids(["1", "2", "3", "4", "5"])

    def compare(self, prompts):
        sequence, output = self.local_model_chat_completion(prompts)
        compare_results = []
        for idx in range(sequence.shape[0]):
            seq_logits = [
                logits[idx] for logits in output.logits
            ]  # convert to [seq_len, vocab_size]
            compare_result = self.extract_probs(sequence[idx], seq_logits)
            compare_results.append(compare_result)
        return compare_results

    def generate(self, prompts):
        sequence, output = self.local_model_chat_completion(prompts)
        # Skip special tokens and role tokens
        generated_text = self.tokenizer.batch_decode(sequence, skip_special_tokens=True)
        return generated_text

    def rate_score(self, prompts):
        sequence, output = self.local_model_chat_completion(prompts)
        # print(output.logits)
        # return sequence, output.logits
        scores, logprobs = [], []
        for idx in range(sequence.shape[0]):
            seq_logits = [
                logits[idx] for logits in output.logits
            ]  # convert to [seq_len, vocab_size]
            score, logprob = self.extract_score(sequence[idx], seq_logits)
            scores.append(score)
            logprobs.append(logprob)
        return scores, logprobs

    def extract_score(self, sequence, logits):
        """
        sequence: [batch_size, seq_len]
        logits: seq_len x [batch_size, vocab_size]
        output: int score
        """
        for idx, token_id in enumerate(sequence):
            logit = logits[idx]
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
        for idx, token_id in enumerate(sequence):
            if token_id in self.A_ids or token_id in self.B_ids:
                logit = logits[idx]
                probs = F.softmax(logit, dim=-1)
                prob_A = sum([probs[a_id].item() for a_id in self.A_ids])
                prob_B = sum([probs[b_id].item() for b_id in self.B_ids])
                prob_C = sum([probs[c_id].item() for c_id in self.C_ids])
                logit_A = sum([logit[a_id].item() for a_id in self.A_ids])
                logit_B = sum([logit[b_id].item() for b_id in self.B_ids])
                logit_C = sum([logit[c_id].item() for c_id in self.C_ids])
                uncertainty = calculate_uncertainty([prob_A, prob_B])
                compare_result = CompareResultObject(
                    raw_prob_A=prob_A,
                    raw_prob_B=prob_B,
                    raw_prob_C=prob_C,
                    uncertainty=uncertainty,
                    logit_A=logit_A,
                    logit_B=logit_B,
                    logit_C=logit_C,
                )
                return compare_result
        print("Failed to extract probs")
        print(self.tokenizer.decode(sequence))
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)

    def local_model_chat_completion(self, prompts):
        messages = []
        for prompt in prompts:
            msg = MistralModelLocal.get_chat_message(prompt)
            msg = self.tokenizer.apply_chat_template(
                msg, tokenize=False
            )  # return_tensors="pt", return_dict=True)
            messages.append(msg)

        input = self.tokenizer(messages, return_tensors="pt", padding=True)
        input = input.to(device)
        output = self.model.generate(
            **input,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
            output_logits=True,
            max_new_tokens=self.max_tokens,
            do_sample=self.do_sample,
            temperature=None,
            top_p=None
        )

        newly_generated_tokens = output.sequences[:, input.input_ids.shape[-1] :]
        return newly_generated_tokens, output

    @staticmethod
    def get_chat_message(prompt, chat_system_instruction=None):
        if chat_system_instruction:
            message = [
                # {'role': 'assistant', 'content': chat_system_instruction},
                {"role": "user", "content": prompt},
            ]
        else:
            message = [{"role": "user", "content": prompt}]
        return message


if __name__ == "__main__":
    example_prompt = """\
Evaluate and compare the coherence of the two following summary candidates for the given input source text.

Input source text: Paul Merson has restarted his row with Andros Townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team's 0-0 draw with Burnley on Sunday. 'Just been watching the game, did you miss the coach? #RubberDub #7minutes,' Merson put on Twitter. Merson initially angered Townsend for writing in his Sky Sports column that 'if Andros Townsend can get in (the England team) then it opens it up to anybody.' Paul Merson had another dig at Andros Townsend after his appearance for Tottenham against Burnley Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley Andros Townsend scores England's equaliser in their 1-1 friendly draw with Italy in Turin on Tuesday night The former Arsenal man was proven wrong when Townsend hit a stunning equaliser for England against Italy and he duly admitted his mistake. 'It's not as though I was watching hoping he wouldn't score for England, I'm genuinely pleased for him and fair play to him – it was a great goal,' Merson said. 'It's just a matter of opinion, and my opinion was that he got pulled off after half an hour at Manchester United in front of Roy Hodgson, so he shouldn't have been in the squad. 'When I'm wrong, I hold my hands up. I don't have a problem with doing that - I'll always be the first to admit when I'm wrong.' Townsend hit back at Merson on Twitter after scoring for England against Italy Sky Sports pundit  Merson (centre) criticised Townsend's call-up to the England squad last week Townsend hit back at Merson after netting for England in Turin on Wednesday, saying 'Not bad for a player that should be 'nowhere near the squad' ay @PaulMerse?' Any bad feeling between the pair seemed to have passed but Merson was unable to resist having another dig at Townsend after Tottenham drew at Turf Moor.

Compare the following outputs:

Summary candidate A: paul merson was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . andros townsend scored the tottenham midfielder in the 89th minute . paul merson had another dig at andros townsend after his appearance . the midfielder had been brought on to the england squad last week . click here for all the latest arsenal news news .

Summary candidate B: paul merson has restarted his row with andros townsend . the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . andros townsend scores england 's equaliser in their 1-1 friendly draw with italy in turin .

Question: Which summary candidate has better coherence? If the candidate A is better, please return 'A'. If the candidate B is better, please return 'B'. You must return the choice only.
Answer: \
"""

    import os

    model = MistralModelLocal({"model": "mistralai/Mistral-7B-Instruct-v0.1"})
    print(example_prompt)
    result = model.compare(example_prompt)
