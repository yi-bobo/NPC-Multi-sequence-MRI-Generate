# CPC_BM model
## 💫model change && question
- [x] Conditional images use the same backbone network for feature extraction.
❗ out of GPU memory

- [x] Only use one residual block combined with a downsampling block to extract conditional image features(version: 1.0.0), and **convert all text information into tensors in advance during the preprocessing stage using a pre-trained CLIP model.**
❗ Processing a single data point takes about 2.8 seconds, and for data that requires 6,400 iterations, the training time is too long.

- [x] Only use one residual block combined with a downsampling block to form a new conditional image feature extraction module, version 1.0.1.
❗ Processing a single data point takes about 2.8 seconds, and for data that requires 6,400 iterations, the training time is too long.

- [x] Directly fuse text and conditional image features through gating without performing feature alignment first.
❗ Processing a single data point takes about 2.8 seconds, and for data that requires 6,400 iterations, the training time is too long.

- [x] Do not add conditional training.
❗ Processing a single data point takes about 2 seconds

- [ ] Rewrite a UNet network for diffusion models, with top and bottom sampling implemented using convolution and adding residual blocks for further feature extraction, and an extra Multi-Head self-attention module in the middle layer.
❗ 18.9%% of GPU memory per layer using 1 residual blocks. 20.1% of GPU memory per layer using 2 residual blocks.  **Residual block takes up less GPU memory**. 
❗ The up and down sampling process of the backbone network of the generated model are added two residual blocks in each layer, and the structure of attention-residual block-attention is chosen in the middle layer, and the conditional textual and image information are used in the form of convolution to extract multi-scale features, which occupies 22.4% of the GPU graphics memory.

- [ ] 在浅层使用动态卷积对条件特征与网络特征进行融合，在深层使用注意力机制进行融合