# TASH
This repository provides TASH (Token Aggregation and Selection Hashing), the first deep hashing framework specifically designed for underwater image retrieval.
Built upon a teacher-student self-distillation Vision Transformer (ViT), TASH introduces three key modules:

Underwater Image Augmentation (UIA): simulates realistic underwater degradations (e.g., color shift, low contrast) to improve robustness.

Multi-layer Token Aggregation (MTA): fuses hierarchical features across layers for better context representation.

Attention-based Token Selection (ATS): highlights discriminative tokens while suppressing background noise.

The model learns discriminative features and compresses them into compact binary codes, enabling efficient and accurate large-scale image retrieval. Extensive experiments on two underwater datasets show that TASH outperforms state-of-the-art methods and sets new benchmarks in this field.
