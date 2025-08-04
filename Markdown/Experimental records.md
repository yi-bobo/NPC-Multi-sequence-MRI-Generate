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

- [ ] Fusion of conditional and network features using dynamic convolution at shallow level and attention mechanism at deep level




h: 
Down-{1,8,256,256 ->(conv_in)-> 32,8,256,256 ->[(res)-> 32,8,256,256 ->(res)-> 32,8,256,256 ->(down)-> 32,4,128,128] ->[(res)-> 64,4,128,128 ->(res)-> 64,4,128,128 ->(down)-> 64,2,64,64] -> [(res)-> 128,2,64,64 ->(res)-> 128,2,64,64 ->(down)-> 128,1,32,32] ->[(res)-> 256,1,32,32 ->(attn)-> 256,1,32,32 ->(res)-> 256,1,32,32 ->(attn)-> 256,1,32,32 ->(down)-> 256,1,16,16]}
Mid-{[256,1,16,16 ->(res)-> 256,1,16,16 ->(attn)-> 256,1,16,16 ->(res) -> 256,1,16,16] -> [256,1,16,16 ->(res)-> 256,1,16,16 ->(attn)-> 256,1,16,16 ->(res) -> 256,1,16,16]}
Up-{256,1,16,16 ->[(up)-> 
256,1,32,32 + 256,1,32,32 ->(res)-> 256,1,32,32 ->(attn) -> 
256,1,32,32 + 256,1,32,32 ->(res)-> 256,1,32,32 ->(attn) ->
256,1,32,32 ->(res)-> 128,1,32,32] ->[(up)->
128,2,64,64 + 128,2,64,64 ->(res)-> 
128,2,64,64 + 128,2,64,64 ->(res)-> 
128,2,64,64 ->(res)-> 64,2,64,64] ->[(up)->
64,4,128,128 + 64,4,128,128 ->(res)-> 
64,4,128,128 + 64,4,128,128 ->(res)->
64,4,128,128 ->(res)-> 32,4,128,128] ->[(up)->
32,8,256,256 + 32,8,256,256 ->(res)->
32,8,256,256 + 32,8,256,256 ->(res)-> 
32,8,256,256]}