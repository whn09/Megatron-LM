#!/bin/bash
# Pretrain a multimodal model.

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WORKSPACE=/workspace/checkpoints
export LOAD_NAME=Meta-Llama-3.1-8B-mcore
MODEL_NAME="mcore-llama3.1-8b-pretraining"

# Check that the user has set an output path for model checkpoints.
if [[ -z $WORKSPACE ]]; then
    echo "Please set WORKSPACE for storing your model checkpoints."
    exit 1
fi

SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

if [[ -z $LOAD_NAME ]]; then
    echo "Please set LOAD_NAME for input model name."
    exit 1
fi

# CHECKPOINT_DIR="${WORKSPACE}/${LOAD_NAME}/checkpoints"
CHECKPOINT_DIR="${WORKSPACE}/${LOAD_NAME}"

DATA_TRAIN="/workspace/dataset/c4/"

DEBUG=0
if [[ $DEBUG -eq 1 ]]; then
    BZ=32
    NW=2
    HD=0.0
    LI=1
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
else
    BZ=512
    NW=2
    HD=0.1
    LI=10
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
fi

OPTIONS=" \
    --apply-layernorm-1p \
    --attention-softmax-in-fp32 \
    --use-checkpoint-args \
    --use-distributed-optimizer \
    --transformer-impl transformer_engine \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout ${HD} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --decoder-seq-length 1024 \
    --max-position-embeddings 131072 \
    --ffn-hidden-size 14336 \
    --train-iters 20000 \
    --micro-batch-size 64 \
    --global-batch-size ${BZ} \
    --lr-decay-iters 20000 \
    --lr-warmup-fraction .01 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --log-throughput \
    --timing-log-level 2 \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /workspace/checkpoints/Meta-Llama-3.1-8B \
    --data-path ${DATA_TRAIN} \
    --save-interval 1000 \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --eod-mask-loss \
    --patch-dim 14 \
    --img-h 336 \
    --img-w 336 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --ckpt-format torch \
    --no-load-optim \
    --no-load-rng \
    --rotary-base 500000 \
    --use-rope-scaling
"

    # --exit-on-missing-checkpoint \

    # --rotary-base 1000000 \

    # --seq-length 576 \

    # --freeze-LM \
    # --freeze-ViT \

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

torchrun --nproc_per_node 8 pretrain_gpt.py ${OPTIONS}