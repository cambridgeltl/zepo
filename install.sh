conda create -n zepo python=3.10 -y
conda activate zepo
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install wandb
pip install openai
pip install jupyter
pip install accelerate
conda install -c conda-forge cudatoolkit-dev
pip install flash-attn --no-build-isolation
pip install bitsandbytes

pip install torchtext
pip install pandas
pip install datasets
pip install sentencepiece
pip install mistralai
pip install scikit-learn
pip install scipy

pip install sacrebleu
pip install rouge
pip install termcolor