from textwrap import dedent


def get_pairwise_prompt_template(dataset, use_instruction=None):
    if dataset == "SummEval":
        prompt = dedent(
            """\
        Source text: {{ input }}

        Summary candidate A: {{ output_1 }}
                        
        Summary candidate B: {{ output_2 }}
                                            
        Question: Evaluate and compare the coherence of the two summary candidates for the given source text. \
Which summary candidate has better coherence? \
If the candidate A is better, please return 'A'. \
If the candidate B is better, please return 'B'. \
You must return the choice only.
        Answer: """
        )
        if use_instruction:
            prompt = dedent(
                """\
        Source text: {{ input }}

        Summary candidate A: {{ output_1 }}
                        
        Summary candidate B: {{ output_2 }}
                                            
        Question: {{ instruction }}
        Answer: """
            )

    elif dataset == "newsroom":
        prompt = dedent(
            """\
        Source text: {{ input }}

        Summary candidate A: {{ output_1 }}

        Summary candidate B: {{ output_2 }}

        Question: Evaluate and compare the coherence of the two summary candidates for the given source text. \
Which summary candidate has better coherence? \
If the candidate A is better, please return 'A'. \
If the candidate B is better, please return 'B'. \
You must return the choice only.
        Answer: """
        )
        if use_instruction:
            prompt = dedent(
                """\
            Source text: {{ input }}

            Summary candidate A: {{ output_1 }}

            Summary candidate B: {{ output_2 }}

            Question: {{ instruction }}
            Answer: """
            )

    elif dataset == "TopicalChat":
        prompt = dedent(
            """\
        Dialog history: 
        {{ input }}

        Response candidate A: {{ output_1 }}
        Response candidate B: {{ output_2 }}
                                            
        Question: Which response is overall better for the given dialog history? \
Please consider aspects including naturalness, understandability, context consistency and knowledge richness. \
If the candidate A is better, please return 'A'. \
If the candidate B is better, please return 'B'. \
You must return the choice only.
        Answer: """
        )
        if use_instruction:
            prompt = dedent(
                """\
                    Dialog history: {{ input }}

                    Response candidate A: {{ output_1 }}
                    Response candidate B: {{ output_2 }}

                    Question: {{ instruction }}
                    Answer: """
            )

    elif dataset == "GSM8k":
        prompt = dedent(
            """\
        Math question: {{ input }}

        Solution candidate A: {{ output_1 }}
                        
        Solution candidate B: {{ output_2 }}
                                            
        Instruction: Compare the quality of the two solution candidates for the given math question. \
Which solution candidate is better explained and more logical? \
If the candidate A is better, please return 'A'. \
If the candidate B is better, please return 'B'. \
You must only return your choice and make no explanation.
        Answer: """
        )

    else:
        assert False, f"Invalid dataset: {dataset}"

    return prompt


def get_pointwise_prompt_template(dataset, with_input):
    if with_input:
        prompt = dedent(
            """\
        Evaluate the overall quality of the following output candidate for the given input.

        Input: {{ input }}

        Output candidate: {{ output }}
                                            
        Question: How would you rate the overall quality of the output candidate? \
Please provide a score between 1 and 10. \
You must return the score only.
        Answer: """
        )
    else:
        prompt = dedent(
            """\
        Evaluate the overall quality of the following output candidate.

        Output candidate: {{ output }}
                                            
        Question: How would you rate the overall quality of the output candidate? \
Please provide a score between 1 and 10. \
You must return the score only.
        Answer: """
        )
    return prompt


def get_cot_compare_prompt_template(dataset):
    if dataset == "SummEval":
        prompt = dedent(
            """\
        Source text: {{ input }}

        Summary candidate A: {{ output_1 }}
                        
        Summary candidate B: {{ output_2 }}
                                            
        Instruction: Please briefly analyse and compare the coherence of the two summary candidates for the given source text, \
and then conclude which candidate is more coherent."""
        )

    elif dataset == "TopicalChat":
        prompt = dedent(
            """\
        Dialog history: 
        {{ input }}

        Response candidate A: {{ output_1 }}
        Response candidate B: {{ output_2 }}
                                            
        Question: Which response is overall better for the given dialog history? \
Please consider aspects including naturalness, understandability, context consistency and knowledge richness. \
If the candidate A is better, please return 'A'. \
If the candidate B is better, please return 'B'. \
You must return the choice only.
        Answer: """
        )

    elif dataset == "GSM8k":
        prompt = dedent(
            """\
        Math question: {{ input }}

        Solution candidate A: {{ output_1 }}
                        
        Solution candidate B: {{ output_2 }}
                                            
        Instruction: Analyse and compare the quality of the two solution candidates for the given math question. \
Please briefly discuss the strengths and weaknesses of both solution candidates and conclude which is more logical and correct?"""
        )

    else:
        assert False, f"Invalid dataset: {dataset}"
    return prompt


def get_cot_eval_prompt_template():
    prompt = dedent(
        """\
    {{ cot_response}}
                    
    Based on the above evaluation, which candidate is preferred according to the analysis? 
    If the candidate A is preferred, please return 'A'. \
If the candidate B is preferred, please return 'B'. \
If both candidates are equally preferred, please return 'C'. \
You must return the choice only.
    Answer: """
    )

    return prompt
