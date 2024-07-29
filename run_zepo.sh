

python3 zepo.py \
        --dataset='SummEval' \
        --engine='google/gemma-2-9b-it' \
        --aspect_name='coherence' \
        --eval_data_num=10 \
        --sample_num=5 \
        --epoch_num=5 \
        --batch_size=10 

        # --engine='meta-llama/Meta-Llama-3-8B-Instruct' \
