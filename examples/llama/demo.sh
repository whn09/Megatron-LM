cd $HOME/checkpoints

huggingface-cli download --repo-type model --resume-download meta-llama/Meta-Llama-3.1-8B --local-dir Meta-Llama-3.1-8B # You may need to login

cd $WORKSPACE/dataset
# wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
aws s3 cp --recursive s3://datalab/dataset/c4 c4

# Enter docker
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $HOME/Megatron-LM:/workspace/megatron -v $HOME/dataset:/workspace/dataset -v $HOME/checkpoints:/workspace/checkpoints megatron-multimodal:latest

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
   --target-pipeline-parallel-size 1 \
   --checkpoint-type hf \
   --load-dir $WORKSPACE/checkpoints/Meta-Llama-3.1-8B \
   --save-dir $WORKSPACE/checkpoints/Meta-Llama-3.1-8B-mcore \
   --tokenizer-model $WORKSPACE/checkpoints/Meta-Llama-3.1-8B \
   --model-size llama3

cd $WORKSPACE/megatron

python tools/preprocess_data.py \
   --input $WORKSPACE/dataset/c4/c4_demo.json \
   --json-keys 'text' \
   --output-prefix $WORKSPACE/dataset/c4/c4_demo \
   --workers 1 \
   --tokenizer-type HuggingFaceTokenizer \
   --tokenizer-model $WORKSPACE/checkpoints/Meta-Llama-3.1-8B

examples/llama/pretrain_llama.sh