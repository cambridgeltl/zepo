
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F
import sys
import json
sys.path.append('../')
from utils import CompareResultObject, calculate_uncertainty
from prompts import get_cot_eval_prompt_template
from jinja2 import Environment, Template


device = 'cuda'

class LocalModel:
    def __init__(self, params):
        self.model_name = params['model']
        self.temperature = params['temperature'] if 'temperature' in params else 0
        self.max_tokens = params['max_tokens'] if 'max_tokens' in params else 64
        self.do_sample = params['do_sample'] if 'do_sample' in params else False
        self.top_p = params['top_p'] if 'top_p' in params else 1
        self.qa_style = params['qa_style'] if 'qa_style' in params else False
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                            device_map=self.device, 
                                                            # attn_implementation="flash_attention_2",   # flash attention is not easy to install
                                                            torch_dtype=torch.bfloat16)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if 'mistral' in self.model_name or 'Llama-2' in self.model_name or \
           'vicuna' in self.model_name or 'zephyr' in self.model_name or \
           'Phi' in self.model_name or 'gemma' in self.model_name:
            self.A_ids = self.tokenizer.convert_tokens_to_ids(['A','▁A'])   # A: 330
            self.B_ids = self.tokenizer.convert_tokens_to_ids(['B','▁B'])   # B: 365
            self.C_ids = self.tokenizer.convert_tokens_to_ids(['C','▁C'])   # C: 
            self.Yes_ids = self.tokenizer.convert_tokens_to_ids(['Yes','▁Yes'])
            self.No_ids = self.tokenizer.convert_tokens_to_ids(['No','▁No'])
        elif 'Llama-3' in self.model_name:
            self.A_ids = self.tokenizer.convert_tokens_to_ids(['A','ĠA'])   
            self.B_ids = self.tokenizer.convert_tokens_to_ids(['B','ĠB'])   
            self.C_ids = self.tokenizer.convert_tokens_to_ids(['C','ĠC'])  
            self.Yes_ids = self.tokenizer.convert_tokens_to_ids(['Yes','ĠYes'])
            self.No_ids = self.tokenizer.convert_tokens_to_ids(['No','ĠNo'])
        self.score_ids = self.tokenizer.convert_tokens_to_ids(['1','2','3','4','5'])


    def compare(self, prompts):
        sequence, output = self.local_model_chat_completion(prompts)
        compare_results = []
        for idx in range(sequence.shape[0]):
            seq_logits = [logits[idx] for logits in output.logits]      # convert to [seq_len, vocab_size]
            if self.do_sample:
                compare_result = self.extract_decision(sequence[idx])
            else:
                compare_result = self.extract_probs(sequence[idx], seq_logits)
            compare_results.append(compare_result)
        return compare_results
    
    
    def cot_compare(self, prompts):
        self.max_tokens = 512
        sequence, output = self.local_model_chat_completion(prompts)
        decoded_sequence = self.tokenizer.batch_decode(sequence, skip_special_tokens=True)
        
        compare_prompt_template = Template(get_cot_eval_prompt_template())
        compare_prompts = [compare_prompt_template.render(cot_response=decoded_sequence[idx]) for idx in range(len(decoded_sequence))]
        self.max_tokens = 16
        compare_results = self.compare(compare_prompts)
        return compare_results
        # compare_results = []
        # for idx in range(sequence.shape[0]):
        #     compare_result = self.extract_json_decision(sequence[idx])
        #     compare_results.append(compare_result)
        # return compare_results


    def generate(self, prompts):
        sequence, output = self.local_model_chat_completion(prompts)
        # Skip special tokens and role tokens
        generated_text = self.tokenizer.batch_decode(sequence, skip_special_tokens=True)
        # Llama-3 should skip from 4: to skip the role token
        return generated_text, sequence


    # def rate_score(self, prompts):
    #     sequence, output = self.local_model_chat_completion(prompts)
    #     # print(output.logits)
    #     # return sequence, output.logits
    #     scores, logprobs = [], []
    #     for idx in range(sequence.shape[0]):
    #         seq_logits = [logits[idx] for logits in output.logits]      # convert to [seq_len, vocab_size]
    #         score, logprob = self.extract_score(sequence[idx], seq_logits)
    #         scores.append(score)
    #         logprobs.append(logprob)
    #     return scores, logprobs


    # def extract_score(self, sequence, logits):
    #     '''
    #     sequence: [batch_size, seq_len]
    #     logits: seq_len x [batch_size, vocab_size]
    #     output: int score
    #     '''
    #     for idx, token_id in enumerate(sequence):
    #         logit = logits[idx]
    #         logprobs = F.log_softmax(logit, dim=-1).cpu()
    #         score_logprobs = logprobs[self.score_ids].tolist()
    #         token = self.tokenizer.decode(token_id)
    #         if is_integer_string(token):
    #             return int(token), score_logprobs
    #     print("Failed to extract score")
    #     print(self.tokenizer.batch_decode(sequence))
    #     return 3, [np.log(0.2)]*5
    def extract_json_decision(self, sequence) -> CompareResultObject:
        '''
        sequence: [batch_size, seq_len]
        output: compare_result_object
        '''
        decoded_sequence = self.tokenizer.decode(sequence, skip_special_tokens=True).split("```json")[-1].split("```")[0]
        print(decoded_sequence)
        decoded_json = json.loads(decoded_sequence)
        if decoded_json['Answer'] == 'A':
            return CompareResultObject(raw_prob_A=1, uncertainty=0)
        elif decoded_json['Answer'] == 'B':
            return CompareResultObject(raw_prob_B=1, uncertainty=0)
        print("Failed to extract json decision")
        print(decoded_json)
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)
        
    
    
    def extract_decision(self, sequence) -> CompareResultObject:
        '''
        sequence: [batch_size, seq_len]
        output: compare_result_object
        '''
        if self.qa_style == False:
            for idx, token_id in enumerate(sequence):
                if token_id in self.A_ids:
                    return CompareResultObject(raw_prob_A=1, uncertainty=0)            
                elif token_id in self.B_ids:
                    return CompareResultObject(raw_prob_B=1, uncertainty=0)
        else:
            for idx, token_id in enumerate(sequence):
                if token_id in self.Yes_ids:
                    return CompareResultObject(raw_prob_A=1, uncertainty=0)
                elif token_id in self.No_ids:
                    return CompareResultObject(raw_prob_B=1, uncertainty=0)
        print("Failed to extract probs")
        print(self.tokenizer.decode(sequence))
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)
    
        
    def extract_probs(self, sequence, logits) -> CompareResultObject:
        '''
        sequence: [batch_size, seq_len]
        logits: seq_len x [batch_size, vocab_size]
        output: compare_result_object
        '''
        if self.qa_style == False:
            for idx, token_id in enumerate(sequence):
                if token_id in self.A_ids or token_id in self.B_ids:
                    logit = logits[idx]
                    probs = F.softmax(logit, dim=-1)
                    prob_A = max([probs[a_id].item() for a_id in self.A_ids])
                    prob_B = max([probs[b_id].item() for b_id in self.B_ids])
                    prob_C = max([probs[c_id].item() for c_id in self.C_ids])
                    uncertainty = calculate_uncertainty([prob_A, prob_B])
                    compare_result = CompareResultObject(raw_prob_A=prob_A, raw_prob_B=prob_B, raw_prob_C=prob_C, uncertainty=uncertainty)
                    return compare_result
        else:
            for idx, token_id in enumerate(sequence):
                if token_id in self.Yes_ids or token_id in self.No_ids:
                    logit = logits[idx]
                    probs = F.softmax(logit, dim=-1)
                    prob_Yes = max(probs[self.Yes_ids]).item()
                    prob_No = max(probs[self.No_ids]).item()
                    uncertainty = calculate_uncertainty([prob_Yes, prob_No])
                    compare_result = CompareResultObject(raw_prob_A=prob_Yes, raw_prob_B=prob_No, uncertainty=uncertainty)
                    return compare_result
        print("Failed to extract probs")
        print(self.tokenizer.decode(sequence))
        return CompareResultObject(raw_prob_A=0.5, raw_prob_B=0.5, uncertainty=1)


    def local_model_chat_completion(self, prompts):
        if 'vicuna' in self.model_name:
            input = self.tokenizer(prompts, return_tensors="pt", padding=True)
        else:
            messages = []
            for prompt in prompts:
                msg = LocalModel.get_chat_message(prompt)
                msg = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)# return_tensors="pt", return_dict=True)
                messages.append(msg)

            input = self.tokenizer(messages, return_tensors="pt", padding=True)
        # print(self.tokenizer.batch_decode(input.input_ids))
        if self.model_name == 'google/gemma-2-9b-it':
            input.pop('attention_mask')
        input = input.to(device)
        output = self.model.generate(
                    # input_ids=input.input_ids,
                    # attention_mask=input.attention_mask,
                    **input,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id, 
                    output_logits=True,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

        newly_generated_tokens = output.sequences[:, input.input_ids.shape[-1]:]
        # print(self.tokenizer.decode(newly_generated_tokens[0]))
        return newly_generated_tokens, output
    

    @staticmethod
    def get_chat_message(prompt, chat_system_instruction=None):
        if chat_system_instruction:
            message = [
                # {'role': 'assistant', 'content': chat_system_instruction},
                {'role': 'user', 'content': prompt},
            ]
        else:
            message = [{'role': 'user', 'content': prompt}]
        return message

