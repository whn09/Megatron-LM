cd $WORKSPACE/checkpoints

huggingface-cli download --repo-type model --resume-download meta-llama/Meta-Llama-3.1-8B --local-dir Meta-Llama-3.1-8B # You may need to login

cd $WORKSPACE/dataset
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Enter docker

cd megatron

export WORKSPACE=/workspace

pip install megatron-energon[av_decode]
pip install mamba-ssm

python tools/checkpoint/convert.py \
   --bf16 \
   --model-type GPT \
   --loader llama_mistral \
   --saver core \
   --target-tensor-parallel-size 1 \
   --checkpoint-type hf \
   --load-dir $WORKSPACE/checkpoints/Meta-Llama-3.1-8B \
   --save-dir $WORKSPACE/checkpoints/Meta-Llama-3.1-8B-mcore \
   --tokenizer-model $WORKSPACE/checkpoints/Meta-Llama-3.1-8B \
   --model-size llama3

cd $WORKSPACE/megatron
examples/llama/pretrain_llama.sh