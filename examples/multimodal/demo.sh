# p5en.48xlarge

cd $HOME

mkdir -p $HOME/dataset
mkdir -p $HOME/checkpoints
sudo chmod -R 777 $HOME/dataset
sudo chmod -R 777 $HOME/checkpoints

# git clone https://github.com/NVIDIA/Megatron-LM/
git clone https://github.com/whn09/Megatron-LM/

# For multimodal, refer to https://github.com/NVIDIA/Megatron-LM/tree/main/examples/multimodal

# sudo apt install -y git-lfs
# or
sudo yum install -y git-lfs

git lfs install

source activate pytorch
pip install transformers
huggingface-cli login  # Enter your huggingface token

cd $HOME/dataset
# git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
huggingface-cli download --repo-type dataset --resume-download liuhaotian/LLaVA-Pretrain --local-dir LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip

cd $HOME/checkpoints
# git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 # You may need to login
# git clone https://huggingface.co/openai/clip-vit-large-patch14-336
huggingface-cli download --repo-type model --resume-download mistralai/Mistral-7B-Instruct-v0.3 --local-dir Mistral-7B-Instruct-v0.3 # You may need to login
huggingface-cli download --repo-type model --resume-download openai/clip-vit-large-patch14-336 --local-dir clip-vit-large-patch14-336

cd $HOME/Megatron-LM/examples/multimodal
docker build -t megatron-multimodal .

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 579019700964.dkr.ecr.us-east-1.amazonaws.com
aws ecr create-repository --repository-name megatron-multimodal
docker tag megatron-multimodal:latest 579019700964.dkr.ecr.us-east-1.amazonaws.com/megatron-multimodal:latest
docker push 579019700964.dkr.ecr.us-east-1.amazonaws.com/megatron-multimodal:latest

docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $HOME/Megatron-LM:/workspace/megatron -v $HOME/dataset:/workspace/dataset -v $HOME/checkpoints:/workspace/checkpoints megatron-multimodal:latest

export WORKSPACE=/workspace
# export WORKSPACE=/home/ubuntu

cd megatron

CUDA_DEVICE_MAX_CONNECTIONS=1 python tools/checkpoint/convert.py \
   --bf16 \
   --model-type GPT \
   --loader llama_mistral \
   --saver core \
   --target-tensor-parallel-size 4 \
   --checkpoint-type hf \
   --load-dir $WORKSPACE/checkpoints/Mistral-7B-Instruct-v0.3 \
   --save-dir $WORKSPACE/checkpoints/Mistral-7B-Instruct-v0.3-mcore \
   --tokenizer-model $WORKSPACE/checkpoints/Mistral-7B-Instruct-v0.3 \
   --model-size mistral

python examples/multimodal/model_converter/clip_converter.py \
   --download-root $WORKSPACE/checkpoints/clip-vit-large-patch14-336 \
   --output $WORKSPACE/checkpoints/clip-vit-large-patch14-336-mcore \
   --tensor-parallel-size 4 \
   --use-te

examples/multimodal/combine_lm_vision_checkpoints.sh $WORKSPACE/checkpoints/Mistral-7B-Instruct-v0.3-mcore $WORKSPACE/checkpoints/clip-vit-large-patch14-336-mcore $WORKSPACE/checkpoints/llava-mcore

# Edit examples/multimodal/convert_llava_pretrain_to_wds.py

python examples/multimodal/convert_llava_pretrain_to_wds.py

cd $WORKSPACE/dataset/LLaVA-Pretrain/wds
energon prepare ./

# > Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 9,1,0
# > Do you want to create a dataset.yaml interactively? [Y/n]: Y
# > Please enter a number to choose a class: 9 (VQASample)
# > Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]: Y
# > Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): jpg
# > Please enter a webdataset field name for 'context' (<class 'str'>): json[0][value]
# > Please enter a webdataset field name for 'answers' (typing.Optional[typing.List[str]], default: None): json[1][value]
# > Please enter a webdataset field name for 'answer_weights' (typing.Optional[torch.Tensor], default: None):

# Edit pretrain_dataset.yaml

pip install megatron-energon[av_decode]
pip install mamba-ssm

cd $WORKSPACE/megatron
examples/multimodal/pretrain_mistral_clip.sh

# examples/multimodal/sft_mistral_clip.sh

# # Clean memory
# sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches"

# pip install s5cmd

# s5cmd cp s3://datalab/dataset/LLaVA-Pretrain/* $HOME/dataset/LLaVA-Pretrain


# LLaVA-NeXT

cd $HOME/dataset
huggingface-cli download --repo-type dataset --resume-download lmms-lab/LLaVA-NeXT-Data --include llava_next_raw_format/ --local-dir LLaVA-NeXT-Data

cd LLaVA-NeXT-Data/llava_next_raw_format
for file in *.tar.gz; do
    tar -xzf "$file"
done

cd $HOME/checkpoints
huggingface-cli download --repo-type model --resume-download llava-hf/llama3-llava-next-8b --local-dir llama3-llava-next-8b
# git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct # You may need to login
huggingface-cli download --repo-type model --resume-download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir Meta-Llama-3.1-8B-Instruct # You may need to login
huggingface-cli download --repo-type model --resume-download openai/clip-vit-large-patch14-336 --local-dir clip-vit-large-patch14-336

# Enter docker

cd megatron

CUDA_DEVICE_MAX_CONNECTIONS=1 python tools/checkpoint/convert.py \
   --bf16 \
   --model-type GPT \
   --loader llama_mistral \
   --saver core \
   --target-tensor-parallel-size 4 \
   --checkpoint-type hf \
   --load-dir $WORKSPACE/checkpoints/Meta-Llama-3.1-8B-Instruct \
   --save-dir $WORKSPACE/checkpoints/Meta-Llama-3.1-8B-Instruct-mcore \
   --tokenizer-model $WORKSPACE/checkpoints/Meta-Llama-3.1-8B-Instruct \
   --model-size llama3

python examples/multimodal/model_converter/clip_converter.py \
   --download-root $WORKSPACE/checkpoints/clip-vit-large-patch14-336 \
   --output $WORKSPACE/checkpoints/clip-vit-large-patch14-336-mcore \
   --tensor-parallel-size 4 \
   --use-te

examples/multimodal/combine_lm_vision_checkpoints.sh $WORKSPACE/checkpoints/Meta-Llama-3.1-8B-Instruct-mcore $WORKSPACE/checkpoints/clip-vit-large-patch14-336-mcore $WORKSPACE/checkpoints/llava-next-mcore

python examples/multimodal/convert_llava_next_pretrain_to_wds.py

cd $WORKSPACE/dataset/LLaVA-NeXT-Data/llava_next_raw_format/wds
energon prepare ./

# > Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 9,1,0
# > Do you want to create a dataset.yaml interactively? [Y/n]: Y
# > Please enter a number to choose a class: 9 (VQASample)
# > Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]: Y
# > Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): jpg
# > Please enter a webdataset field name for 'context' (<class 'str'>): json[0][value]
# > Please enter a webdataset field name for 'answers' (typing.Optional[typing.List[str]], default: None): json[1][value]
# > Please enter a webdataset field name for 'answer_weights' (typing.Optional[torch.Tensor], default: None):


examples/multimodal/pretrain_llama_clip.sh
