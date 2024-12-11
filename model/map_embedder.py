from typing import Any, Dict, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.controlnet import zero_module



class BEVControlNetConditioningEmbedding(torch.nn.Module):
    #conditioning module of controlnet
    def __init__(self,conditioning_embedding_channels=320,conditioning_size=(25,200,200),block_out_channels=(32,64,128,256)):
        super().__init__()
        # input size   25, 200, 200 (bev map change to front cam image)
        # output size 320,  28,  50
        
        # print("#######################",conditioning_size)
        
        self.conv_in=torch.nn.Conv2d(conditioning_size[0],block_out_channels[0],kernel_size=3,padding=1)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",conditioning_size[0])
        # print("##########################################",block_out_channels[0])
        self.blocks=torch.nn.ModuleList([])
        
        for i in range(len(block_out_channels)-2):
            channel_in=block_out_channels[i]
            channel_out=block_out_channels[i+1]
            self.blocks.append(torch.nn.Conv2d(channel_in,channel_in,kernel_size=3,padding=1))
            self.blocks.append(torch.nn.Conv2d(channel_in,channel_out,kernel_size=3,padding=(2,1),stride=2))
            
        channel_in=block_out_channels[-2]
        channel_out=block_out_channels[-1]
        self.blocks.append(torch.nn.Conv2d(channel_in,channel_in,kernel_size=3,padding=(2,1)))
        self.blocks.append(torch.nn.Conv2d(channel_in,channel_out,kernel_size=3,padding=(2,1),stride=(2,1)))
        
        self.conv_out=zero_module(torch.nn.Conv2d(block_out_channels[-1],conditioning_embedding_channels,kernel_size=3,padding=1))


    def forward(self,x):
        embedding=self.conv_in(x)
        embedding=F.silu(embedding)
        
        for block in self.blocks:
            embedding=block(embedding)
            embedding=F.silu(embedding)
        embedding=self.conv_out(embedding)
        return (embedding)