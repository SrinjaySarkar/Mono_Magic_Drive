# Mono_Magic_Drive
Code for monocular view conditioned scene generation using Diffusion Models 

## Training and Validation
How to train :
```
accelerate launch --mixed_precision fp16 --gpu_ids all --num_processes 8 train.py \
  +exp=224x400 runner=8gpus
```
How to test:
```
python test.py
```
Make changes in the config file accordingly [here:](./configs)

## Current Results
- With Spatial Attention
![](https://github.com/SrinjaySarkar/Mono_Magic_Drive/blob/main/assets/0_gen0.png?raw=true)
![](https://github.com/SrinjaySarkar/Mono_Magic_Drive/blob/main/assets/1_gen0.png?raw=true)
![](https://github.com/SrinjaySarkar/Mono_Magic_Drive/blob/main/assets/2_gen0.png?raw=true)
![](https://github.com/SrinjaySarkar/Mono_Magic_Drive/blob/main/assets/3_gen0.png?raw=true)
![](https://github.com/SrinjaySarkar/Mono_Magic_Drive/blob/main/assets/4_gen0.png?raw=true)

- With Spatia-Temporal Attention
