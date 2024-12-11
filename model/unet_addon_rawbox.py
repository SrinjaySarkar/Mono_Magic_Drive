from typing import Any, Dict, Optional, Tuple, Union, List
import logging
from dataclasses import dataclass
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange

from diffusers.utils import BaseOutput
from diffusers import UNet2DConditionModel
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (CrossAttnDownBlock2D,DownBlock2D,UNetMidBlock2DCrossAttn,get_down_block,)
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.controlnet import zero_module

#local imports
from model.embedder import get_embedder
from model.output_cls import BEVControlNetOutput
from model.map_embedder import BEVControlNetConditioningEmbedding
from misc.common import load_module

# BEVControlNet 

class BEVControlNetModel(ModelMixin,ConfigMixin):
    _supports_gradient_checkpointing=True
    
    @register_to_config
    def __init__(self,in_channels:int=4,flip_sin_to_cos:bool=True,freq_shift:int=0,down_block_types:Tuple[str]=("CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"),only_cross_attention:Union[bool, Tuple[bool]]=False,block_out_channels=(320, 640, 1280, 1280),layers_per_block: int = 2,downsample_padding: int = 1,mid_block_scale_factor: float = 1,act_fn: str = "silu",norm_num_groups: Optional[int] = 32,norm_eps: float = 1e-5,
    cross_attention_dim: int = 1280,attention_head_dim: Union[int, Tuple[int]] = 8,use_linear_projection: bool = False,class_embed_type: Optional[str] = None,num_class_embeds: Optional[int] = None,upcast_attention: bool = False,resnet_time_scale_shift: str = "default",projection_class_embeddings_input_dim: Optional[int] = None,controlnet_conditioning_channel_order: str = "rgb",
    conditioning_embedding_out_channels: Optional[Tuple[int]] = None,
    # these two kwargs will be used in `self.config`
    global_pool_conditions: bool = False,
    # BEV params
    uncond_cam_in_dim: Tuple[int, int] = (3, 7),camera_in_dim: int = 189, camera_out_dim: int = 768,  # same as word embeddings
    map_embedder_cls: str = None, map_embedder_param: dict = None, map_size: Tuple[int, int, int] = None, use_uncond_map: str = None,
    drop_cond_ratio: float = 0.0,drop_cam_num: int = 1,drop_cam_with_box: bool = False,cam_embedder_param: Optional[Dict] = None,bbox_embedder_cls: str = None,bbox_embedder_param: dict = None):
        super().__init__()
        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}.")

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}.")

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`:{down_block_types}.")
        
        self.cam2token=torch.nn.Linear(camera_in_dim,camera_out_dim)
        if uncond_cam_in_dim:
            self.uncond_cam=torch.nn.Embedding(1,uncond_cam_in_dim[0]*uncond_cam_in_dim[1])
            # the num of len-3 vectors. We use Fourier emb on len-3 vector.
            self.uncond_cam_num=uncond_cam_in_dim[1]
        self.drop_cond_ratio=drop_cond_ratio
        self.drop_cam_num=drop_cam_num
        self.drop_cam_with_box=drop_cam_with_box
        
        # in 3, freq 4->embedder.out_dim = 27
        self.cam_embedder=get_embedder(**cam_embedder_param) #camera embedding is just frequnecy embedding for camera pose
        
        # input
        conv_in_kernel=3
        conv_in_padding=(conv_in_kernel-1)//2
        self.conv_in=torch.nn.Conv2d(in_channels,block_out_channels[0],kernel_size=conv_in_kernel,padding=conv_in_padding)
        
        # time
        time_embed_dim=block_out_channels[0]*4 # timsestep embedding
        # time_embed_dim=block_out_channels[0] * 4
        self.time_proj=Timesteps(block_out_channels[0],flip_sin_to_cos,freq_shift)
        timestep_input_dim=block_out_channels[0]
        self.time_embedding=TimestepEmbedding(timestep_input_dim,time_embed_dim,act_fn=act_fn,)
        
        #class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embeding=torch.nn.Embedding(num_class_embeds,time_embed_dim)
        elif class_embed_type=="timestep":
            self.class_embedding=TimestepEmbedding(timestep_input_dim,time_embed_dim)
        elif class_embed_type=="identity":
            self.class_embedding=torch.nn.Identity(time_embed_dim,time_embed_dim)
        elif class_embed_type=="projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError("`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set")
            self.class_embedding=TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding=None
        
        #controlnet conditioning embedding (BEV MAP IMAGE)
        if map_embedder_cls is None:
            print("map embedder here")
            cond_embedder_cls=BEVControlNetConditioningEmbedding
            embedder_param={"conditioning_size":map_size,"block_out_channels": conditioning_embedding_out_channels}
            print(embedder_param)
        else:
            cond_embedder_cls=load_module(map_embedder_cls)
            embedder_param=map_embedder_param
        
        #controlnet BEV conditioning
        self.controlnet_cond_embedding=cond_embedder_cls(conditioning_embedding_channels=block_out_channels[0],**embedder_param)
        logging.debug(f"[BEVControlNetModel] map_embedder: {self.controlnet_cond_embedding}")
        
        #uncond map
        if use_uncond_map is not None and drop_cond_ratio>0:
            if use_uncond_map=="negative1":
                tmp=torch.ones(map_size)
                self.register_buffer("uncond_map",tmp)
            elif use_uncond_map=="learnable":
                tmp=torch.nn.Parameter(torch.randn(map_size))
                self.register_buffer("uncond_map",tmp)
            elif use_uncond_map=="random":
                tmp=torch.randn(map_size)
                self.register_buffer("uncond_map",tmp)
            else:
                raise TypeError(f"Unknown map type: {use_uncond_map}.")
        else:
            self.uncond_map=None
        
        #bev box embedder
        model_cls=load_module(bbox_embedder_cls)
        self.bbox_embedder=model_cls(**bbox_embedder_param)
        
        self.down_blocks=torch.nn.ModuleList([])
        self.controlnet_down_blocks=torch.nn.ModuleList([])
        
        if isinstance(only_cross_attention,bool):
            only_cross_attention=[only_cross_attention]*len(down_block_types)
        if isinstance(attention_head_dim, int):
            attention_head_dim=(attention_head_dim,)*len(down_block_types)

        
        #down
        output_channel=block_out_channels[0]
        controlnet_block=torch.nn.Conv2d(output_channel,output_channel,kernel_size=1)
        controlnet_block=zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)
        
        for i,down_block_type in enumerate(down_block_types):
            input_channel=output_channel
            output_channel=block_out_channels[i]
            is_final_block=i==len(block_out_channels)-1
            
            # print(attention_head_dim[i])
            # print(only_cross_attention[i])
            
            down_block=get_down_block(down_block_type,num_layers=layers_per_block,in_channels=input_channel,out_channels=output_channel,temb_channels=time_embed_dim,add_downsample=not is_final_block,resnet_eps=norm_eps,resnet_act_fn=act_fn,resnet_groups=norm_num_groups,cross_attention_dim=cross_attention_dim,attn_num_head_channels=attention_head_dim[i],downsample_padding=downsample_padding,use_linear_projection=use_linear_projection,only_cross_attention=only_cross_attention[i],upcast_attention=upcast_attention,resnet_time_scale_shift=resnet_time_scale_shift)
            
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",down_block)
            
            self.down_blocks.append(down_block)
            
            # controlnet_block=torch.nn.Conv2d(320,output_channel,kernel_size=1)
            # controlnet_block=zero_module(controlnet_block)
            
            for _ in range(layers_per_block):
                controlnet_block=torch.nn.Conv2d(output_channel,output_channel,kernel_size=1)
                controlnet_block=zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
            if not is_final_block:
                controlnet_block=torch.nn.Conv2d(output_channel,output_channel,kernel_size=1)
                controlnet_block=zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
        
        #mid
        mid_block_channel=block_out_channels[-1]
        controlnet_block=torch.nn.Conv2d(mid_block_channel,mid_block_channel,kernel_size=1)
        controlnet_block=zero_module(controlnet_block)
        self.controlnet_mid_block=controlnet_block
        
        self.mid_block=UNetMidBlock2DCrossAttn(in_channels=mid_block_channel,temb_channels=time_embed_dim,resnet_eps=norm_eps,resnet_act_fn=act_fn,output_scale_factor=mid_block_scale_factor,resnet_time_scale_shift=resnet_time_scale_shift,cross_attention_dim=cross_attention_dim,attn_num_head_channels=attention_head_dim[-1],resnet_groups=norm_num_groups,use_linear_projection=use_linear_projection,upcast_attention=upcast_attention)
    
    def _embed_camera(self,camera_param):
        #camera_param (torch.Tensor):[N,6,3,7], 7 for 3+4(3 columns for intrinsics and 4 for extrinsics)
        (bs,N_cam,C_param,emb_num)=camera_param.shape
        assert C_param==3
        assert emb_num==self.uncond_cam_num or self.uncond_cam_num is None,(
        f"You assign `uncond_cam_in_dim[1]={self.uncond_cam_num}`,"f"but your data actually have {emb_num} to embed. Please change your config.")
        
        camera_emb=self.cam_embedder(rearrange(camera_param, "b n d c -> (b n c) d"))
        camera_emb=rearrange(camera_emb,"(b n c) d -> b n (c d)",n=N_cam,b=bs)           
        return (camera_emb)

    def uncond_cam_param(self,repeat_size: Union[List[int], int] = 1):
        if isinstance(repeat_size,int):
            repeat_size=[1,repeat_size]
        repeat_size_num=int(np.prod(repeat_size))
        # we only have one uncond cam, embedding input is always 0
        param=self.uncond_cam(torch.LongTensor([0]*repeat_size_num).to(device=self.device))
        param=param.reshape(*repeat_size,-1,self.uncond_cam_num)
        # print("uncond param shape:",param.shape)
        return (param)
    
    # revise functions
        
    def add_cam_states(self,encoder_hidden_states,camera_emb=None):
        # encoder_hidden_states (torch.Tensor): b, len, 768 (text embedding)
        # camera_emb (torch.Tensor): b, n_cam, dim. if None, use uncond cam
        bs=encoder_hidden_states.shape[0] 
        # for conditional text this is embedding of positional encoding of camera params (bs,n_Cam,768)
        # for unconditional text this is None
        # print("add cam states",camera_emb.shape)
        if camera_emb is None:
            #bs.1.768
            cam_hidden_states=self.cam2token(self._embed_camera(self.uncond_cam_param(bs)))
            # print("bs:",bs)#1 beacuse the padding token ebedding is of 1,n_tokens_padded,758
            # print("camera emb is none for padding token:",self.uncond_cam_param(bs).shape)#make it into shape (1,1,3,7)
            # print("embed it:",self._embed_camera(self.uncond_cam_param(bs)).shape)#mebed it (1,1,189) 
            # print("cam hidden states with padding tokens uncond cam:",cam_hidden_states.shape)#convert to tokens(1,1,768)
            # exit()
        else:
            cam_hidden_states=self.cam2token(camera_emb)#b,n_cam,dim(linear layer)
        # print("camera token of camera param embedding:",cam_hidden_states.shape)
        N_cam=cam_hidden_states.shape[1]
        # print("encoder_hidden_states toke nemebdding of caption:",encoder_hidden_states.shape)
        encoder_hidden_states_with_cam=torch.cat([cam_hidden_states.unsqueeze(2),repeat(encoder_hidden_states,'b c ... -> b repeat c ...',repeat=N_cam)],dim=2)
        # (bs,n_cam,768) => (bs,n_cam,1,768) then cconcat with (bs,n_token,768)=>repeat for 6 cams(bs,n_cam,n_token,768) then concat((bs,n_cam,1,768),(bs,n_cam,n_token,,768)) ==> (bs,n_cam,n_token+1,768) this is for text tokens of captions with camera
        # for padding tokens is (1,1,n+token+1,768)
        # print(encoder_hidden_states_with_cam.shape)
        return (encoder_hidden_states_with_cam)
    
    def substitute_with_uncond_cam(self,encoder_hidden_states_with_cam,encoder_hidden_states_uncond,mask: Optional[torch.LongTensor] = None):
        # print("encoder with cam states:",encoder_hidden_states_with_cam.shape)
        # print("encoder uncond:",encoder_hidden_states_uncond.shape)
        # print("mask",mask.shape)
        # exit()
        
        encoder_hidden_states_uncond_with_cam=self.add_cam_states(encoder_hidden_states_uncond)
        if mask is None:#all to uncond
            mask=torch.ones(encoder_hidden_states_with_cam.shape[:2],dtype=torch.long)
        mask=mask>0
        # print("after concatenating random cam and then uncond text params:",encoder_hidden_states_uncond_with_cam.shape)(1,1,n_token+1,768)
        encoder_hidden_states_with_cam[mask]=encoder_hidden_states_uncond_with_cam[None]
        # exit()
        return (encoder_hidden_states_with_cam)
    
    def _random_use_uncond_cam(self,encoder_hidden_states_with_cam,encoder_hidden_states_uncond):
        # encoder_hidden_states_with_cam (_type_): B, n_cam, n_padded_token + 1, 768
        # encoder_hidden_states_uncond (_type_): 1, n_padded_token, 768
        # uncond prompt with camera
        assert self.drop_cond_ratio > 0.0 and self.training
        mask=torch.zeros(encoder_hidden_states_with_cam.shape[:2],dtype=torch.long)
        # print("mask",mask.shape)
        for bs in range(len(encoder_hidden_states_with_cam)):
            if random.random() > self.drop_cond_ratio:
                cam_id=random.sample(range(encoder_hidden_states_with_cam.shape[1]),self.drop_cam_num)
                mask[bs,cam_id]=1
        # print("mask:",mask.shape)
        # print("mask:",mask)
        encoder_hidden_states_with_cam=self.substitute_with_uncond_cam(encoder_hidden_states_with_cam,encoder_hidden_states_uncond,mask)
        # print("replace the mask==1 locations in encoder_hidden_states_with_cam with encoder_hidden_states_uncond_with_cam")
        return (encoder_hidden_states_with_cam,mask)
    
    def substitute_with_uncond_map(self,controlnet_cond,mask=None):
        # controlnet_cond (Tensor): map with B, C, H, W
        # mask (LongTensor): binary mask on B dim
        # returns controlnet_cond
        if mask is None:
            mask=torch.ones(controlnet_cond.shape[0],dtype=torch.long)
        if any(mask>0) and self.uncond_map is None:
            raise RuntimeError(f"You cannot use uncond_map before setting it.")
        if all(mask==0):
            return controlnet_cond
        controlnet_cond[mask>0]=self.uncond_map[None]
        return (controlnet_cond)

    def _random_use_uncond_map(self,controlnet_cond):
        # randomly replace map to unconditional map (if not None)
        #contro;net_cond(bs,c,200,200)
        if self.uncond_map is None:
            # print(self.uncond_map)
            return controlnet_cond
        mask=torch.zeros(len(controlnet_cond),dtype=torch.long)
        for i in range(len(mask)):
            if random.random() < self.drop_cond_ratio:
                mask[i]=1
        # print("controlnet cond shape ,mask shape:",controlnet_cond.shape,mask.shape)
        exit()
        return (self.substitute_with_uncond_map(controlnet_cond,mask))


    @classmethod
    def from_unet(cls,unet:UNet2DConditionModel,controlnet_conditioning_channel_order: str="rgb",load_weights_from_unet=True,**kwargs):
        r"""
        Instantiate BEVControlnet class from UNet2DConditionModel.

        Parameters: cls
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """
        bev_controlnet=cls(in_channels=unet.config.in_channels,flip_sin_to_cos=unet.config.flip_sin_to_cos,freq_shift=unet.config.freq_shift,down_block_types=unet.config.down_block_types,only_cross_attention=unet.config.only_cross_attention,block_out_channels=unet.config.block_out_channels,layers_per_block=unet.config.layers_per_block,downsample_padding=unet.config.downsample_padding,mid_block_scale_factor=unet.config.mid_block_scale_factor,act_fn=unet.config.act_fn,norm_num_groups=unet.config.norm_num_groups,norm_eps=unet.config.norm_eps,cross_attention_dim=unet.config.cross_attention_dim,attention_head_dim=unet.config.attention_head_dim,use_linear_projection=unet.config.use_linear_projection,class_embed_type=unet.config.class_embed_type,num_class_embeds=unet.config.num_class_embeds,upcast_attention=unet.config.upcast_attention,resnet_time_scale_shift=unet.config.resnet_time_scale_shift,projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
        ## BEV params
        **kwargs) #initilaized with the ldm pretrained model
        
        #cls is bevcontrolnet model
        #unet is the instant of the ldm model (unet2dconditionModel) pretraine ldm unet
        
        # print("CLS##############",cls)
        # print(unet)
        
        if load_weights_from_unet:
            bev_controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            bev_controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            bev_controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
            
            if bev_controlnet.class_embedding:
                bev_controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())
            
            bev_controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            bev_controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())
        return (bev_controlnet)
    
    

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}
        def fn_recursive_add_processors(name: str,module: torch.nn.Module,processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"]=module.processor
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}",child,processors)
            return (processors)
        for name,module in self.named_children():
            fn_recursive_add_processors(name,module,processors)
        return (processors)       
    
    

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self,processor: Union [AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:
        """
        count=len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes.")

        def fn_recursive_attn_processor(name:str,module:torch.nn.Module,processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor,dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}",child,processor)
        for name,module in self.named_children():
            fn_recursive_attn_processor(name,module,processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())
    
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []
        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)
            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)
        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)
        num_sliceable_layers=len(sliceable_head_dims)
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]
        slice_size = ( num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size)

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different" f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}.")

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")
        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())
            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)
        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)
        
    
    def _set_gradient_checkpointing(self,module,value=False):
        if isinstance(module,(CrossAttnDownBlock2D,DownBlock2D)):
            module.gradient_checkpointing = value
    
    def add_uncond_to_kwargs(self,camera_param,bboxes_3d_data,image,max_len=None,**kwargs):
        
        # uncond in the front, cond in the tail
        batch_size,N_cam=camera_param.shape[:2]
        ret=dict()
        
        # print("here")
        # print("camera param:",camera_param.shape)
        # print("bboxes_3d_Data",bboxes_3d_data["bboxes"].shape)
        # print("max len:",max_len)
        # print("###################################")
        # exit()
        
        ret['camera_param']=torch.cat([self.uncond_cam_param([batch_size,N_cam]),camera_param])
        if bboxes_3d_data is None:
            logging.warn("Your 'bboxes_3d_data' should not be None. If this warning keeps ""popping, please check your code.")
            if max_len is not None:
                device=camera_param.device
                #fmt:off
                ret["bboxed_3d_data"]={"bboxes":torch.zeros([batch_size*2,N_cam,max_len,8,3],device=device),"classes":torch.zeros([batch_size*2,N_cam,max_len],device=device,dtype=torch.long),"masks":torch.zeros([batch_size*2,N_cam,max_len],device=device,dtype=torch.bool)}
                
                for k,v in ret["bboxed_3d_data"].items():
                    logging.debug(f"padding {k} to {v.shape}.")
            else:
                ret["bboxes_3d_data"]=None
        else:
            ret["bboxes_3d_data"]=dict()# do not change the original dict
            for key in ["bboxes","classes","masks"]:
                ret["bboxes_3d_data"][key]=torch.cat([torch.zeros_like(bboxes_3d_data[key]),bboxes_3d_data[key]])
                if max_len is not None:
                    token_num=max_len-ret["bboxes_3d_data"][key].shape[2]
                    assert token_num>=0
                    to_pad=torch.zeros_like(ret["bboxes_3d_data"][key])
                    to_pad=repeat(to_pad[:, :, 1], 'b n ... -> b n l ...', l=token_num)
                    ret["bboxes_3d_data"][key]=torch.cat([ret["bboxes_3d_data"][key], to_pad,],dim=2)
                    logging.debug(f"padding {key} with {token_num}, final size: "f"{ret['bboxes_3d_data'][key].shape}")
        
        # print(ret)
        # exit()
        
        if self.uncond_map is None:
            # print(0)
            ret['image']=image
            # print(image.shape)
        else:
            ret['image']=self.substitute_with_uncond_map(image,None)
        # others, keep original
        for k,v in kwargs.items():
            ret[k]=v
        # print(ret.keys())
        # print(ret['bboxes_3d_data']['bboxes'].shape)
        # exit()
        return (ret)

    def add_uncond_to_emb(self,prompt_embeds,N_cam,encoder_hidden_states_with_cam):
        # uncond in the front, cond in the tail
        encoder_hidden_states_with_uncond_cam=self.controlnet.add_cam_states(prompt_embeds)
        token_num=encoder_hidden_states_with_cam.shape[1] - encoder_hidden_states_with_uncond_cam.shape[1]
        encoder_hidden_states_with_uncond_cam=self.bbox_embedder.add_n_uncond_tokens(encoder_hidden_states_with_uncond_cam,token_num)
        
        encoder_hidden_states_with_cam=torch.cat([repeat(encoder_hidden_states_with_uncond_cam,'b ... -> (b n) ...', n=N_cam),encoder_hidden_states_with_cam],dim=0)
        return (encoder_hidden_states_with_cam)
    
    def prepare(self,cfg,**kwargs):
        self.bbox_embedder.prepare(cfg,**kwargs)
        
    
    def forward(self,sample: torch.FloatTensor,timestep:Union[torch.Tensor, float, int],camera_param: torch.Tensor,#bev
                bboxes_3d_data: Dict[str, Any],#bev
                encoder_hidden_states:torch.Tensor,controlnet_cond: torch.FloatTensor,
                encoder_hidden_states_uncond: torch.Tensor = None,#bev
                conditioning_scale: float = 1.0,class_labels: Optional[torch.Tensor] = None,timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,cross_attention_kwargs: Optional[Dict[str, Any]] = None,guess_mode: bool = False,
                return_dict: bool = True, **kwargs) -> Union[BEVControlNetOutput, Tuple]:
        
        
        channel_order=self.config.controlnet_conditioning_channel_order
        
        # this is during train 
        # print("bevcontrolNET forward function")
        # print("chanel:",channel_order)
        # print("map image:",controlnet_cond.shape)
        # print("sample",sample.shape)
        # print("timestep:",timestep.shape)
        # print("camera_pram",camera_param.shape)
        # print("encoder_hidden_states:",encoder_hidden_states.shape)
        # print("encoder_hidden_states_uncond:",encoder_hidden_states_uncond.shape)
        # print("controlnet_cond:",controlnet_cond.shape)
        # exit()
        
        
        #this is during val
        print("bevcontrolNET forward function VAL")
        print("chanel:",channel_order)
        print("map image:",controlnet_cond.shape)
        print("sample",sample.shape)
        print("timestep:",timestep.shape)
        print("camera_pram",camera_param.shape)
        print("encoder_hidden_states:",encoder_hidden_states.shape)
        # print("encoder_hidden_states_uncond:",encoder_hidden_states_uncond.shape)
        print("controlnet_cond:",controlnet_cond.shape)
        
        # print("################################## CAMERA AND TEXT EMBEDDING #########################################")
        
        if attention_mask is not None:
            attention_mask=(1-attention_mask.to(sample.dtype)) * -10000.0
            attention_mask=attention_mask.unsqueeze(1)
        
        
        # print("Attention mask:",attention_mask.shape) 
        N_cam=camera_param.shape[1]
        assert N_cam==6
        camera_emb=self._embed_camera(camera_param)
        # print("Camera embeddings:",camera_emb.shape)#bs,n_cam,189
        encoder_hidden_states_with_cam=self.add_cam_states(encoder_hidden_states,camera_emb)
        # print("after concatentating cam and the caption token embedddings:",encoder_hidden_states_with_cam.shape)
        # we may drop the condition during training, but not drop controlnet
        # print("drop cond ratio",self.drop_cond_ratio)
        # print("training",self.training)
        # exit()
        
        
        
        if (self.drop_cond_ratio > 0.0 and self.training):
            if encoder_hidden_states_uncond is not None:
                # print("cpatiom token emebeddings with cam: & caption padding embeddings with uncond cam:",encoder_hidden_states_with_cam.shape,encoder_hidden_states_uncond.shape)
                encoder_hidden_states_with_cam,uncond_mask=self._random_use_uncond_cam(encoder_hidden_states_with_cam, encoder_hidden_states_uncond)
            
            controlnet_cond=controlnet_cond.type(self.dtype)
            # print("isbfjklsfksbfkjsdnfkjsdnfisdfkdsfijlsdfbij")
            controlnet_cond=self._random_use_uncond_map(controlnet_cond)
        else:
            uncond_mask = None
        
        #uncond mask is a random mask for every camera
        # print("encode_hidden_states_with_cam + encoder_uncond_cam in mask==1 locations:",encoder_hidden_states_with_cam.shape)
        # print("uncond mask:",uncond_mask.shape)
        # print("controlnet cond:",controlnet_cond.shape)
        # print("uncond mask:",uncond_mask)
        # exit()
        
        #bbox embeddings
        # print("################################## 3D BBOX EMBEDDING #########################################")
        #bboxes (bs,n_cam or 1,max_len,7)
        # print("bboxes 3d data keys:",bboxes_3d_data.keys())
        # for k in bboxes_3d_data.keys():
        #         print(k,bboxes_3d_data[k].shape)
        
        # print(self.drop_cam_with_box)
        # exit()
        # print(self.drop_cam_with_box)
        if bboxes_3d_data is not None:
            # print(bboxes_3d_data)
            bbox_embedder_kwargs={}
            for k,v in bboxes_3d_data.items():
                bbox_embedder_kwargs[k]=v.clone()
            
            # print(bbox_embedder_kwargs.keys())
            # print(bbox_embedder_kwargs["bboxes"].shape)
            # print(bbox_embedder_kwargs["bboxes"][0])
            # print(bbox_embedder_kwargs["classes"].shape)
            # print(bbox_embedder_kwargs["masks"].shape)
            # exit()
            
            if self.drop_cam_with_box and uncond_mask is not None:
                _,n_box=bboxes_3d_data["bboxes"].shape[:2]
                if n_box!=N_cam:
                    assert n_box==1, "either N_cam or 1."
                    for k in bboxes_3d_data.keys():
                        ori_v=rearrange(bbox_embedder_kwargs[k], 'b n ... -> (b n) ...')
                        new_v=repeat(ori_v, 'b ... -> b n ...', n=N_cam)
                        bbox_embedder_kwargs[k] = new_v
                # here we set mask for dropped boxes to all zero
                masks=bbox_embedder_kwargs['masks']
                masks[uncond_mask>0]=0
                # print("N box:",n_box)
                # print("bbox masks",masks.shape)
                # print("uncond mask:",uncond_mask.shape)
                # exit()
            
            # original flow
            b_box,n_box=bbox_embedder_kwargs["bboxes"].shape[:2]
            # print("b_box:",b_box)
            # print("n_box:",n_box)
            # print(b_box,n_box)
            # exit()
            
            for k in bboxes_3d_data.keys():
                bbox_embedder_kwargs[k]=rearrange(bbox_embedder_kwargs[k],'b n ... -> (b n) ...')
            # for k in bboxes_3d_data.keys():
            #     print(k,bbox_embedder_kwargs[k].shape)
            # exit()
            
            bbox_emb=self.bbox_embedder(**bbox_embedder_kwargs)
            #bbox embedding (take efourier embedding of boxes, pass through mlp, tokenize class-text, pass through mlp , concat, then pass through another MLP)
            # print("bbox embedding output:",bbox_emb.shape)
            # exit()
            if n_box!=N_cam:
                # print(n_box)
                bbox_emb=repeat(bbox_emb,'b ... -> b n ...', n=N_cam)
            else:
                # each view already has its set of bboxes
                bbox_emb=rearrange(bbox_emb,'(b n) ... -> b n ...',n=N_cam)
            print("bbox embedding output:",bbox_emb.shape)
            # print("camera with padding and text emebedding output:",encoder_hidden_states_with_cam.shape)
            encoder_hidden_states_with_cam=torch.cat([encoder_hidden_states_with_cam,bbox_emb],dim=2)
            print("after conccatenting text ,camera and bbox embedding:",encoder_hidden_states_with_cam.shape)
            #(bs,n_cam,n_tokens+1+n_box_max,768)
            # exit()
            #at this point they they concatenated the camera and text embedding with some mask location containing the padding embedding, then concatenate it with the (bbox,cls and mask) embedding.
            
        
        # print("################################## DIFFUSION TIMESTEPS #########################################")
        timesteps=timestep
        # print("timesteps:",timesteps.shape)#timestep of batchsize (random timestep for each sample in batch)
        
        if not torch.is_tensor(timesteps):
            is_mps=sample.device.type=="mps"
            if isinstance(timestep,float):
                dtype=torch.float32 if is_mps else torch.float64
            else:
                dtype=torch.int32 if is_mps else torch.int64
            timesteps=torch.tensor([timesteps],dtype=dtype,device=sample.device)
        elif len(timesteps.shape)==0:
            timesteps=timesteps[None].to(sample.device)
        
        # print(timesteps.shape)
        # exit()
        
        timesteps=timesteps.expand(sample.shape[0])
        # print(timesteps.shape)
        timesteps=timesteps.reshape(-1)  # time_proj can only take 1-D input
        t_emb=self.time_proj(timesteps)
        # print("sinosidual timestep embedding:",t_emb.shape)
        
        # exit()
        
        # # timesteps does not contain any weights and will always return f32 tensors
        # # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb=t_emb.to(dtype=self.dtype)
        # print("timestep cond:",timestep_cond)
        emb=self.time_embedding(t_emb,timestep_cond)
        # print("embed the sinsiudal ttimestep embeddings:",emb.shape)    
        # print("time embedding in controlnet:",emb.shape,timesteps)
        # print("time embedding in controlnet:",emb[0][0])
        # print("time embedding in controlnet:",emb[1][0])
        
        #take 1 timesteps for each batch , then project it into fourier of that integere and then embed it with a linear network
        # exit()
        
        # print("class embedding:",self.class_embedding)
        if self.class_embedding is not None:
            # print(self.config.class_embed_type)
            if class_labels is None:
                raise ValueError("class labels are provide when num_class_embeds>0")
            if self.config.class_embed_type=="timestep":
                class_labels=self.time_proj(class_labels)
            # print("class embedding:",class_labels.shape)
            
            class_emb=self.class_embedding(class_labels).to(dtype=self.dtype)
            emb=emb+class_emb
        
        # print("class labels:",class_labels)
        # exit()
        
        # print("class emebedding")
        # print("sample:",sample.shape) SAMPLE IN NOISY LATENTS
        
        # print("sample:",sample.shape) #(bs,n_cam,4,h//8,w//8)
        sample=rearrange(sample,'b n ... -> (b n) ...')
        # print("noisy latents reshaped:",sample.shape)
        encoder_hidden_states_with_cam=rearrange(encoder_hidden_states_with_cam,'b n ... -> (b n) ...')
        # print("rearrange concatenated camera , text ,bbox embeddings:",encoder_hidden_states_with_cam.shape)
        
        #sample(bs*n_cam,4,h//8,w//8)
        #enocder_hidden_states_with_cam(bs*n_cam,n_tokens+1+n_box_max,768)
        
        # exit()
        
        if len(emb)<len(sample):
            emb=repeat(emb,'b ... -> (b repeat) ...',repeat=N_cam)
        controlnet_cond=repeat(controlnet_cond,'b ... -> (b repeat) ...',repeat=N_cam)
        # print("controlnet conditioning MAP :",controlnet_cond.shape)
    
        #2:PRE-PROCESS
        sample=self.conv_in(sample)
        # print(sample.shape)
        # print("noisy latents through conv:",sample.shape)
        controlnet_cond=self.controlnet_cond_embedding(controlnet_cond)
        # print("embedding in conditioning:",controlnet_cond.shape)
        sample+=controlnet_cond
        print("adding the controlnet map conditioning to the noisy latents:",sample.shape)
        # exit()
        
        # now we have concat text embedding and camera embedding (bs,n_cam,n_tokens+1,768) 
        # then we embed the bbox for every view (bs,n_cam,max_box,768)
        # then we have the controlnet cond MAP image : (bs,n_classes(8),200,200)
        # pass the image latents through a conv : (bs*n_cam,channels,h//8,w//8)
        # pass the controlent cond image though a embedder (bs*n_cam,channels,h//8,w//8)
        
        
        # print("################################## CONTROLNET FORWARD #########################################")
        
        #3:DOWN BLOCKS
        down_block_res_samples=(sample,)
        for downsample_block in self.down_blocks:
            # print("this")
            if (hasattr(downsample_block,"has_cross_attention") and downsample_block.has_cross_attention):
                sample,res_samples = downsample_block(hidden_states=sample,temb=emb,encoder_hidden_states=encoder_hidden_states_with_cam,attention_mask=attention_mask,cross_attention_kwargs=cross_attention_kwargs)
            else:
                # print("here")
                sample, res_samples=downsample_block(hidden_states=sample,temb=emb)
            down_block_res_samples+=res_samples
        # print("DOWN BLOCK RES SAMPLES:",len(down_block_res_samples))# 12
        # print("DOWN BLOCK RES SAMPLES:",sample.shape)
        # print("DOWN BLOCK RES SAMPLES:",down_block_res_samples[0].shape)# torch.Size([bs*n_cam, 320, 28, 50])
        # print("DOWN BLOCK RES SAMPLES:",down_block_res_samples[1].shape)# torch.Size([bs*n_cam, 320, 28, 50])
        # exit()
        
        #4:MID
        if self.mid_block is not None:
            sample=self.mid_block(sample,emb,encoder_hidden_states=encoder_hidden_states_with_cam,attention_mask=attention_mask,cross_attention_kwargs=cross_attention_kwargs)
        
        # print("AFTER MID BLOCK:",sample.shape) #torch.Size([bs*n_cam,1280,4,7])
        
        #5:CONTROLENET BLOCKS
        controlnet_down_block_res_samples=()
        for down_block_res_sample,controlnet_block in zip(down_block_res_samples,self.controlnet_down_blocks):
            down_block_res_sample=controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples+=(down_block_res_sample,)
        
        down_block_res_samples=controlnet_down_block_res_samples
        # print("DOWN BLOCK RES SAMPLES CONTROLNET:",len(down_block_res_samples))#6
        # print("DOWN BLOCK RES SAMPLES CONTROLNET:",down_block_res_samples[0].shape)# torch.Size([bs*n_cam, 320, 28, 50])
        mid_block_res_sample=self.controlnet_mid_block(sample)
        # print("MID BLOCK CONTROLNET:",mid_block_res_sample.shape) #torch.Size([bs*n_cam,1280,4,7])
        #CHECK ARCH AT : https://miro.medium.com/v2/resize:fit:640/format:webp/1*h1Ng0l5rraA0i5ZKa-vj3g.png
        
        
        #6:scaling
        if guess_mode:
            scales=torch.logspace(-1,0,len(down_block_res_samples) + 1)# 0.1 to 1.0
            scales*=conditioning_scale 
            down_block_res_samples=[sample*scale for sample,scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample*=scales[-1]  # last one
        else:
            down_block_res_samples=[sample*conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample*=conditioning_scale

        # scale the controlnet down blocks outpyt and the controlnet mid block output
        # print(guess_mode)
        # print(self.config.global_pool_conditions)
        # print(return_dict)
        # exit()

        if self.config.global_pool_conditions:
            down_block_res_samples=[torch.mean(sample,dim=(2,3),keepdim=True) for sample in down_block_res_samples]
            mid_block_res_sample=torch.mean(mid_block_res_sample,dim=(2,3),keepdim=True)
        
        if not return_dict:
            return (down_block_res_samples,mid_block_res_sample,encoder_hidden_states_with_cam)
        return BEVControlNetOutput(down_block_res_samples=down_block_res_samples,mid_block_res_sample=mid_block_res_sample,encoder_hidden_states_with_cam=encoder_hidden_states_with_cam,)
        

        
