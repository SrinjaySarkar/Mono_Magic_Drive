from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pprint import pprint
import inspect
import collections
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers.configuration_utils import register_to_config
from diffusers.models.unet_2d_condition import UNet2DConditionModel,UNet2DConditionOutput
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D,CrossAttnUpBlock2D,DownBlock2D,UpBlock2D
from diffusers.models.attention import BasicTransformerBlock

from misc.common import _set_module,_get_module
from model.blocks import BasicMultiviewTransformerBlock

# UNet2DConditionModelMultiview
class UNet2DConditionModelMultiview(UNet2DConditionModel):
    _supports_gradient_checkpointing = True
    _WARN_ONCE = 0
    
    @register_to_config
    def __init__(self,sample_size=None,in_channels=4,out_channels=4,center_input_sample=False,flip_sin_to_cos=True,freq_shift=0,down_block_types=("CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"),mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
    up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),only_cross_attention: Union[bool, Tuple[bool]] = False,block_out_channels: Tuple[int] = (320, 640, 1280, 1280), layers_per_block: Union[int, Tuple[int]] = 2,
    downsample_padding: int = 1,mid_block_scale_factor: float = 1,act_fn: str = "silu",norm_num_groups: Optional[int] = 32,norm_eps: float = 1e-5,cross_attention_dim: Union[int, Tuple[int]] = 1280,encoder_hid_dim: Optional[int] = None,encoder_hid_dim_type: Optional[str] = None,
    attention_head_dim: Union[int, Tuple[int]] = 8,dual_cross_attention: bool = False,use_linear_projection: bool = False,class_embed_type: Optional[str] = None, addition_embed_type: Optional[str] = None, num_class_embeds: Optional[int] = None, upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default", resnet_skip_time_act: bool = False,resnet_out_scale_factor: int = 1.0,
    time_embedding_type: str = "positional",time_embedding_dim: Optional[int] = None, time_embedding_act_fn: Optional[str] = None, timestep_post_act: Optional[str] = None,time_cond_proj_dim: Optional[int] = None, conv_in_kernel: int = 3,conv_out_kernel: int = 3,
    projection_class_embeddings_input_dim: Optional[int] = None, class_embeddings_concat: bool = False, mid_block_only_cross_attention: Optional[bool] = None, cross_attention_norm: Optional[str] = None,
    addition_embed_type_num_heads=64,
    # don't chnage params above this line , for pretrained model
    
    trainable_state="only_new",neighboring_view_pair: Optional[dict] = None,neighboring_attn_type: str = "add", zero_module_type: str = "zero_linear",crossview_attn_type: str = "basic",img_size: Optional[Tuple[int, int]] = None):
        
        super().__init__(sample_size=sample_size, in_channels=in_channels,out_channels=out_channels, center_input_sample=center_input_sample,flip_sin_to_cos=flip_sin_to_cos, freq_shift=freq_shift,down_block_types=down_block_types, mid_block_type=mid_block_type,up_block_types=up_block_types,only_cross_attention=only_cross_attention,block_out_channels=block_out_channels,layers_per_block=layers_per_block,downsample_padding=downsample_padding,mid_block_scale_factor=mid_block_scale_factor, act_fn=act_fn,
        norm_num_groups=norm_num_groups, norm_eps=norm_eps,cross_attention_dim=cross_attention_dim,encoder_hid_dim=encoder_hid_dim,
        encoder_hid_dim_type=encoder_hid_dim_type,attention_head_dim=attention_head_dim,dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,class_embed_type=class_embed_type,addition_embed_type=addition_embed_type,
        num_class_embeds=num_class_embeds,upcast_attention=upcast_attention,resnet_time_scale_shift=resnet_time_scale_shift,resnet_skip_time_act=resnet_skip_time_act,resnet_out_scale_factor=resnet_out_scale_factor,time_embedding_type=time_embedding_type,
        time_embedding_dim=time_embedding_dim,time_embedding_act_fn=time_embedding_act_fn,timestep_post_act=timestep_post_act,time_cond_proj_dim=time_cond_proj_dim,conv_in_kernel=conv_in_kernel, conv_out_kernel=conv_out_kernel,projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,class_embeddings_concat=class_embeddings_concat,mid_block_only_cross_attention=mid_block_only_cross_attention,cross_attention_norm=cross_attention_norm,
        addition_embed_type_num_heads=addition_embed_type_num_heads)
        
        self.crossview_attn_type=crossview_attn_type
        self.img_size=[int(s) for s in img_size] if img_size is not None else None
        self._new_module={}
        for name,mod in list(self.named_modules()):
            # print(name,mod)
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
            if isinstance(mod,BasicTransformerBlock):
                # print(name,mod)
                # print(mod._args)
                # exit()
                if crossview_attn_type=="basic":
                    _set_module(self,name,BasicMultiviewTransformerBlock(**mod._args,neighboring_view_pair=neighboring_view_pair,neighboring_attn_type=neighboring_attn_type,zero_module_type=zero_module_type))
                else:
                    raise TypeError(f"Unknown attn type:{crossview_attn_type}")
                for k,v in _get_module(self,name).new_module.items():
                    self._new_module[f"{name}.{k}"]=v
        self.trainable_state=trainable_state   
        
        # print("#####################################")
        # print(self.down_blocks[0])
        # print(self.down_blocks[1]) 
        # print("#####################################")

    @property
    def trainable_module(self)-> Dict[str, nn.Module]:
        if self.trainable_state=="all":
            return {self.__class__:self}
        elif self.trainable_state=="only_new":
            return self._new_module
        else:
            raise ValueError(f"Unknown trainable_state: {self.trainable_state}")
    
    @property
    def trainable_parameters(self)-> List[nn.Parameter]:
        params=[]
        for mod in self.trainable_module.values():
            for param in mod.parameters():
                params.append(param)
        return (params)

    def train(self,mode=True):
        if not isinstance(mode,bool):
            raise ValueError("training mode is expected to be boolean")
        #first set all to false
        super().train(False)
        if mode:
            # ensure gradient_checkpointing is usable, set training = True
            for mod in self.modules():
                if getattr(mod,"gradient_checkpointing",False):
                    mod.training=True
        # then, for some modules, we set according to `mode
        self.training=False
        for mod in self.trainable_module.values():
            if mod is self:
                super().train(mode)
            else:
                mod.train(mode)
        return self
    
    def enable_gradient_checkpointing(self,flag=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        mod_idx=-1
        for module in self.modules():
            if isinstance(module,(CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
                mod_idx+=1
                if flag is not None and not flag[mod_idx]:
                    logging.debug(f"[UNet2DConditionModelMultiview]" f"gradient_checkpointing skip [{module.__class__}]")
                    continue
                logging.debug(f"[UNet2DConditionModelMultiview] set" f"[{module.__class__}] to gradient_checkpointing")
                module.gradient_checkpointing=True
    
    @classmethod
    def from_unet_2d_condition(cls,unet,load_weights_from_unet=True,**kwargs):
        # in this the cls argument is model.unet_2d_condition_multiview.UNet2DConditionModelMultiview' 
        # in this the unet argument is diffuersers.UNet2DConditionModel' 
        # print ("ok")
        # print(cls)
        # print(unet)
        # Instantiate Multiview unet class from UNet2DConditionModel.
        unet_2d_condition_multiview=cls(**unet.config,**kwargs)
        # print(unet)
        if load_weights_from_unet:
            missing_keys,unexpected_keys=unet_2d_condition_multiview.load_state_dict(unet.state_dict(),strict=False)
            # pprint(missing_keys)
            # pprint(unexpected_keys)
            logging.info(f"[UNet2DConditionModelMultiview] load pretrained with " f"missing_keys: {missing_keys}; "f"unexpected_keys: {unexpected_keys}")
        return (unet_2d_condition_multiview)

    def forward(self,sample: torch.FloatTensor,timestep: Union[torch.Tensor, float, int],encoder_hidden_states: torch.Tensor,class_labels: Optional[torch.Tensor] = None,timestep_cond: Optional[torch.Tensor] = None,attention_mask: Optional[torch.Tensor] = None,cross_attention_kwargs: Optional[Dict[str, Any]] = None,down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,mid_block_additional_residual: Optional[torch.Tensor] = None, return_dict: bool = True,):
        
        # noise_pred=self.unet(latent_model_input,t,encoder_hidden_states=encoder_hidden_states_with_cam,**additional_param, cross_attention_kwargs=cross_attention_kwargs,down_block_additional_residuals=down_block_res_samples,mid_block_additional_residual=mid_block_res_sample).sample
        
        
        #according to documentation this takes as input a noisy sample,conditional state ,a time-step and return a sample shpaed output
        
        # print("MVUNet forward function")
        # print("sample:",sample.shape)
        # print("class labels:",class_labels)
        # print("cross attention kwargs:",cross_attention_kwargs)
        # print("timestep:",timestep.shape)
        # print("encoder_hidden_states:",encoder_hidden_states.shape)
        # print("down_block_additional_residuals:",len(down_block_additional_residuals))
        # print("mid_block_additional_residuals:",mid_block_additional_residual.shape)
        # # exit()
        # print("################################# forward pass of mv unet#########################################")
        
        default_overall_up_factor=2**self.num_upsamplers
        forward_upsample_size=False
        upsample_size=None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            if self._WARN_ONCE==0:
                logging.warning("[UNet2DConditionModelMultiview] Forward upsample size to force interpolation output size.")
                self._WARN_ONCE=1
            forward_upsample_size=True
        
        # prepare attention_mask
        if attention_mask is not None:
            attention_mask=(1-attention_mask.to(sample.dtype))*-10000.0
            attention_mask=attention_mask.unsqueeze(1)
        # print("attention mask:",attention_mask)
        # exit()
        # 0. center input if necessary
        if self.config.center_input_sample:
            # print("centering")
            sample=2*sample-1.0
        # exit()
        
        #1:timesteps
        timesteps=timestep
        # print("timesteps",timesteps)
        # exit()
        
        if not torch.is_tensor(timesteps):
            is_mps=sample.device.type=="mps"
            if isinstance(timestep,float):
                dtype=torch.float32 if is_mps else torch.float64
            else:
                dtype=torch.int32 if is_mps else torch.int64
            timesteps=torch.tensor([timesteps],dtype=dtype,device=sample.device)
        elif len(timesteps.shape)==0:
            timesteps=timesteps[None].to(sample.device)
        
        
        timesteps=timesteps.reshape(-1)
        # print("timesteps:",timesteps.shape)
        # print(timesteps.shape)
        t_emb=self.time_proj(timesteps)
        t_emb=t_emb.to(dtype=self.dtype)
        # print("sinosidual timestep embedding:",t_emb.shape)
        # print(timestep_cond)
        # print(self.class_embedding)
        # print(self.config.addition_embed_type)
        # print(self.time_embed_act)
        # print(self.encoder_hid_proj)
        # exit()
        
        emb=self.time_embedding(t_emb,timestep_cond)
        # print("time embedding inside mutliview:",emb.shape,timesteps)
        # print("inside multiview:",emb[0][0])
        # print("inside multiview:",emb[1][0])
        # print("inside multiview:",emb[6][0])
        # print("inside multiview:",emb[7][0])
        # exit()
        
        
        
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError( "class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type=="timestep":
                class_labels=self.time_proj(class_labels)
                class_labels=class_labels.to(dtype=sample.dtype)
            class_emb=self.class_embedding(class_labels).to(dtype=self.dtype)
            # print("class emb:",class_emb.shape)
            if self.config.class_embeddings_concat:
                emb=torch.cat([emb,class_emb],dim=-1)
            else:
                emb=emb+class_emb
        
        if self.config.addition_embed_type=="text":
            aug_emb=self.add_embedding(encoder_hidden_states)
            emb=emb+aug_emb
            # print("addition type:",emb.shape)
        
        if self.time_embed_act is not None:
            emb=self.time_embed_act(emb)
            # print("time embed shape:",emb.shape)
        
        if self.encoder_hid_proj is not None:
            encoder_hidden_states=self.encoder_hid_proj(encoder_hidden_states)
            # print("enco")

        #2:pre-process  
        # print("before conv in:",sample.shape)
        sample=self.conv_in(sample)
        # print("noisy latetns through convolutional",sample.shape)
        # exit()
        
        #3:down blocks
        # print(self.crossview_attn_type)
        # print(len(self.down_blocks))
        # print(self.down_blocks[0])
        # print(inspect.signature(self.down_blocks[0].forward))
        # exit()
        
        
        # down block structure in cross attention 
        # print("crossattndownBlock2d(transformer2dModel x 2,resnet Blocks, downsampler.) ==> Transformer2DModel(norm,proj_in,transformer_2d blocks x2 ) ==> Transformer2dModel(basicMVTransformer)")
        
        down_block_res_samples=(sample,)
        for idx,downsample_block in enumerate(self.down_blocks):
            # print(idx,downsample_block)
            if self.crossview_attn_type=="epipolar":
                cross_attention_kwargs["out_size"]=sample.shape[-2:]
            if hasattr(downsample_block,"has_cross_attention") and downsample_block.has_cross_attention:
                # print("THIS IS THE BLOCKS OF MV TRANSFORMER")
                sample,res_samples=downsample_block(hidden_states=sample,temb=emb,encoder_hidden_states=encoder_hidden_states,attention_mask=attention_mask,cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs)) 
            else:
                sample,res_samples=downsample_block(hidden_states=sample,temb=emb)
            # print("sample:",sample.shape)
            # print("res sample:",len(res_samples))
            down_block_res_samples+=res_samples
        
        # print(len(down_block_res_samples))
        # print(down_block_res_samples[0].shape)
        # print(cross_attention_kwargs)
        
        # exit()
        
        
        
        if down_block_additional_residuals is not None:
            new_down_block_res_samples=()
            for down_block_res_sample,down_block_additional_residual in zip(down_block_res_samples,down_block_additional_residuals):
                down_block_res_sample=down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples+=(down_block_res_sample,)
            down_block_res_samples=new_down_block_res_samples
        
        # print(len(down_block_additional_residuals))
        # print(self.mid_block)
        
        
        #4:mid
        if self.mid_block is not None:
            if self.crossview_attn_type=="epipolar":
                cross_attention_kwargs["out_size"]=sample.shape[-2:]
            sample=self.mid_block(sample,emb,encoder_hidden_states=encoder_hidden_states,attention_mask=attention_mask,cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs))
        if mid_block_additional_residual is not None:
            sample=sample+mid_block_additional_residual

        
        # exit()
            
        #5:up
        # print(self.up_blocks[0])
        # exit()
        for i,upsample_block in enumerate(self.up_blocks):
            is_final_block=i==len(self.up_blocks) - 1
            # print(len(upsample_block.resnets))
            # exit()
            res_samples=down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples=down_block_res_samples[:-len(upsample_block.resnets)]
            
            #this gets 3 blocks at a time since there are 3 layers in each of the 4 boxes.
            
            # print(len(down_block_res_samples))
            # exit()
            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size=down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block,"has_cross_attention") and upsample_block.has_cross_attention:
                if self.crossview_attn_type=='epipolar':
                    cross_attention_kwargs['out_size'] = sample.shape[-2:]
                sample=upsample_block(hidden_states=sample, temb=emb,res_hidden_states_tuple=res_samples,encoder_hidden_states=encoder_hidden_states,cross_attention_kwargs=copy.deepcopy(cross_attention_kwargs),upsample_size=upsample_size, attention_mask=attention_mask,)
            else:
                sample=upsample_block(hidden_states=sample, temb=emb,res_hidden_states_tuple=res_samples,upsample_size=upsample_size)
        
        
        # exit()
        #6:post-process
        if self.conv_norm_out:
            sample=self.conv_norm_out(sample)
            sample=self.conv_act(sample)
        sample=self.conv_out(sample)
        # print("final output of mvunet:",sample.shape)
        # exit()
        if not return_dict:
            return (sample,)
        return (UNet2DConditionOutput(sample=sample)) 
        
        
            
      

        
        