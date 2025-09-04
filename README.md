# TASH
This repository provides TASH (Token Aggregation and Selection Hashing), the first deep hashing framework specifically designed for underwater image retrieval.
Built upon a teacher-student self-distillation Vision Transformer (ViT), TASH introduces three key modules:

Underwater Image Augmentation (UIA): simulates realistic underwater degradations (e.g., color shift, low contrast) to improve robustness.

Multi-layer Token Aggregation (MTA): fuses hierarchical features across layers for better context representation.

Attention-based Token Selection (ATS): highlights discriminative tokens while suppressing background noise.

The model learns discriminative features and compresses them into compact binary codes, enabling efficient and accurate large-scale image retrieval. Extensive experiments on two underwater datasets show that TASH outperforms state-of-the-art methods and sets new benchmarks in this field.

# Dataset Preparation
1.Download your dataset (e.g., WildFish) and place it under:/path/to/dataset/WildFish/

2.Prepare split files:wildfish_Train.txt、wildfish_DB.txt、wildfish_Query.txt

Owing to the file size limitation, the project includes only the Sea Animals dataset as an example. The text files for WildFish follow the same format and structure.

# Required environment
- Python 3.9
- torch==2.1.0
- torchvision==0.16.0
- kornia==0.8.0
- timm==1.0.15

# setting
Our TASH is trained with Adam optimizer (weight decay: 1e-5, learning rate: 3e-4) for 150 epochs. A linear warm-up followed by a cosine annealing schedule is applied. The batch size is 64.The loss weights are set as  λ₁ = 0.1 and λ₂ = 0.1, following DHD. The fused layer number Nf is 6, and the learnable α in score is constrained within [0, 1] by the Sigmoid function.

## Training
CUDA_VISIBLE_DEVICES=0 python main_DHD.py --mode train \
    --dataset wildfish \
    --data_dir /path/to/dataset/WildFish/ \
    --train_txt /path/to/wildfish_Train.txt \
    --db_txt /path/to/wildfish_DB.txt \
    --query_txt /path/to/wildfish_Query.txt \
    --encoder ViT \
    --N_bits 32 \
    --batch_size 64 \
    --max_epoch 150

## Testing
sameQueryDB parameter (in Evaluate_mAP): When computing mAP, if the database and the query set are identical, set this parameter to True to exclude self-matching results (i.e., to prevent a query item from retrieving itself).

CUDA_VISIBLE_DEVICES=0 python main_DHD.py --mode test \
    --dataset wildfish \
    --data_dir /path/to/dataset/WildFish/ \
    --db_txt /path/to/wildfish_DB.txt \
    --query_txt /path/to/wildfish_Query.txt \
    --encoder ViT \
    --N_bits 32 \
    --resume_path ./model/epoch-150-wildfish-checkpoint-32-model.pt
This will compute mAP and save results to:./logs/final(32bit)_test_map.txt
