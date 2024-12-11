from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import BasicTransformerBlock, AdaLayerNorm
from diffusers.models.controlnet import zero_module


def _ensure_kv_is_int(view_pair: dict):
    """yaml key can be int, while json cannot. We convert here.
    """
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict


class GatedConnector(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        data=torch.zeros(dim)
        self.alpha=torch.nn.parameter.Parameter(data)
    def forward(self, inx):
        # as long as last dim of input == dim, pytorch can auto-broad
        return (F.tanh(self.alpha)*inx)



class BasicMultiviewTransformerBlock(BasicTransformerBlock):
    def __init__(self,dim: int,num_attention_heads: int,attention_head_dim: int,dropout=0.0,cross_attention_dim: Optional[int] = None,activation_fn: str = "geglu",num_embeds_ada_norm: Optional[int] = None,attention_bias: bool = False,only_cross_attention: bool = False,double_self_attention: bool = False,upcast_attention: bool = False,norm_elementwise_affine: bool = True,norm_type: str = "layer_norm",final_dropout: bool = False,
    #multi-view params
    neighboring_view_pair: Optional[Dict[int, List[int]]] = None,neighboring_attn_type: Optional[str] = "add",zero_module_type="zero_linear"):
        super().__init__(dim,num_attention_heads,attention_head_dim,dropout,cross_attention_dim,activation_fn,num_embeds_ada_norm,attention_bias,only_cross_attention,double_self_attention,upcast_attention,norm_elementwise_affine,norm_type,final_dropout)
        
        self.neighboring_view_pair=_ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type=neighboring_attn_type    
        
        # multiview attention
        self.norm4=(AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else torch.nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine))
        # print("mv attention:",self.norm4)
        
        #cross attention
        self.attn4=Attention(query_dim=dim,cross_attention_dim=dim,heads=num_attention_heads,dim_head=attention_head_dim,dropout=dropout,bias=attention_bias,upcast_attention=upcast_attention)
        
        if zero_module_type=="zero_linear":
            self.connector=zero_module(torch.nn.Linear(dim,dim))
        elif zero_module=="gated":
            self.connector=GatedConnector(dim)
        elif zero_module_type=="none":
            self.connect=lambda x:x
        else:
            raise TypeError(f"Unknown zero module type:{zero_module_type}")
        
        # print("cross attention:",self.attn4)
        # print("zero:",zero_module_type)
    
    
    @property
    def new_module(self):
        ret={"norm4":self.norm4,"attn4":self.attn4}
        
        if isinstance(self.connector,torch.nn.Module):
            ret["connector"]=self.connector
        return (ret)
    
    @property
    def n_cam(self):
        return (len(self.neighboring_view_pair))
    
    def _construct_attn_input(self,norm_hidden_states):
        # print("norm self , norm cross attn then this: ",norm_hidden_states.shape)
        B=len(norm_hidden_states)
        # print(B)
        # exit()
        hidden_states_in1=[]
        hidden_states_in2=[]
        cam_order=[]
        # print(self.neighboring_attn_type)
        # exit()
        
        if self.neighboring_attn_type=="add":
            for key,values in self.neighboring_view_pair.items():
                # 0: [5, 1],1: [0, 2],2: [1, 3],3: [2, 4],4: [3, 5],5: [4, 0]
                # print(key,values)
                for value in values:
                    #[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]
                    # print("key:",key)
                    # print(norm_hidden_states[:,key,...].shape)
                    hidden_states_in1.append(norm_hidden_states[:,key,...])
                    # print("value:",value)
                    # print(norm_hidden_states[:,value,...].shape)
                    #[5,1],[0,2],[1,3],[2,4][3,5],[4,0]
                    hidden_states_in2.append(norm_hidden_states[:,value,...])
                    cam_order += [key]*B
            # N*2*B, H*W, head*dim
            hidden_states_in1=torch.cat(hidden_states_in1,dim=0)#(bs*2*n_cam,n_channel,n_attn_tokens)
            hidden_states_in2=torch.cat(hidden_states_in2,dim=0)#(bs*2*n_cam,n_channel,n_attn_tokens)
            cam_order=torch.LongTensor(cam_order)#(bs*2*n_cam)
            
            # print(hidden_states_in1.shape)#(bs*2*n_cam,n_channel,n_attn_tokens)
            # print(hidden_states_in2.shape)
            # print(cam_order)
            
            # exit()
        
        elif self.neighboring_attn_type=="concat":
            for key,values in self.neighboring_view_pair.items():
                # 0: [5, 1],1: [0, 2],2: [1, 3],3: [2, 4],4: [3, 5],5: [4, 0]
                hidden_states_in1.append(norm_hidden_states[:,key])
                hidden_states_in2.append(torch.cat([norm_hidden_states[:,value] for value in values],dim=1))
                cam_order += [key]*B
            hidden_states_in1=torch.cat(hidden_states_in1,dim=0)
            hidden_states_in2=torch.cat(hidden_states_in2,dim=0)
            cam_order=torch.LongTensor(cam_order)
        
        elif self.neighboring_attn_type=="self":
            hidden_states_in1=rearrange(norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2=None
            cam_order=None
        
        else:
            raise NotImplementedError(f"Unknown type: {self.neighboring_attn_type}")
        return (hidden_states_in1, hidden_states_in2, cam_order)
    
    
    def forward(self,hidden_states,attention_mask=None,encoder_hidden_states=None,encoder_attention_mask=None,timestep=None,cross_attention_kwargs=None,class_labels=None,):
        # print("################################################# MV UNET FORWARD #######################################")

        # print("################################################# inside MVUnet #######################################")
        # print("hidden states:",hidden_states.shape)
        # print("attention mask:",attention_mask)
        # print("encoder hidden states",encoder_hidden_states.shape)
        # print("encoder attention mask:",encoder_attention_mask)
        # print("timestep:",timestep)
        # print("cross attention kwargs:",cross_attention_kwargs)
        # print("class labels:",class_labels)
        # print(self.use_ada_layer_norm)
        # print(self.use_ada_layer_norm_zero)
        # print("neighboring_attn_type:",self.neighboring_attn_type)
        # print("neighboring view pair",self.neighboring_view_pair)
        # exit()
        
        
        if self.use_ada_layer_norm:
            norm_hidden_states=self.norm1(hidden_states,timestep)
            # print("norm hidden states")
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states,gate_msa,shift_mlp,scale_mlp,gate_mlp=self.norm1(hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype)
        else:
            norm_hidden_states=self.norm1(hidden_states)
            # print("normal layer nrom:",self.norm1,norm_hidden_states.shape)
        cross_attention_kwargs=cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # print(cross_attention_kwargs)
        # print("norm layer op:",norm_hidden_states.shape)
        # print("cross attn kwargs:",cross_attention_kwargs)
        # print(self.only_cross_attention)
        attn_output=self.attn1(norm_hidden_states,encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,**cross_attention_kwargs,)
        
        # print("self attn op:",attn_output.shape) 
        if self.use_ada_layer_norm_zero:
            attn_output=gate_msa.unsqueeze(1) * attn_output
        hidden_states=attn_output+hidden_states
        # print("self attn output of image+controlnet cond embeddings and adding it to attn input:",attn_output.shape)
        # print("adding the attention embedding outputs to the camera,text,box embeddings:",hidden_states.shape)
        # exit()
        
        #2:cross attention
        # query:(hidden_states),(key,value):(encoder_hidden_states)
        if self.attn2 is not None:
            norm_hidden_states=(self.norm2(hidden_states,timestep) if self.use_ada_layer_norm else self.norm2(hidden_states))
            attn_output=self.attn2(norm_hidden_states,encoder_hidden_states=encoder_hidden_states,attention_mask=encoder_attention_mask,
                **cross_attention_kwargs)
            hidden_states=attn_output+hidden_states
        # print(self.attn2)
        # print("cross attention between (cam,box,txt) embeddings and image+cntrlnet and adding it to the image embeddings:",hidden_states.shape)
        # exit()
        
        # mv cross attention
        norm_hidden_states=self.norm4(hidden_states,timestep) if self.use_ada_layer_norm else self.norm4(hidden_states)
        # print("norm before multi-view cross attention:",norm_hidden_states.shape)
        # print(self.use_ada_layer_norm)
        # batch dim first,cam dim second
        norm_hidden_states=rearrange(norm_hidden_states, '(b n) ... -> b n ...',n=self.n_cam)
        # print("reshaped layer norm of the addition of camera,text,box embeddings and the previous self attention op:",norm_hidden_states.shape)
        B=len(norm_hidden_states)#bs
        hidden_states_in1,hidden_states_in2,cam_order=self._construct_attn_input(norm_hidden_states)#(bs*2*n_cam,:,:)
        #this tensors have the contents of the camera according to the camera view order.
        #[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]  #[5,1],[0,2],[1,3],[2,4][3,5],[4,0]
        #hidden_states_in1 : [,(norm_hidden_states[0],norm_hidden_states[0]),(norm_hidden_states[1],norm_hidden_states[1])....]()
        #hidden_states_in2 : [(norm_hidden_states[5],norm_hidden_states[1]),(norm_hidden_states[0],norm_hidden_states[2]).....]()
        #cam order: [0,0,0,0,1,1,1,1....] camera for both values in cam view order
        # exit()
        # print("cross view attention query :",hidden_states_in1.shape)#[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]
        # print("cross view attention key and vals :",hidden_states_in2.shape)#[5,1],[0,2],[1,3],[2,4][3,5],[4,0]
        # print("cameras for query:",cam_order.shape)
        # print(cam_order[0],cam_order[6],cam_order[12],cam_order[18],cam_order[24],cam_order[30])
        # print(cam_order[3],cam_order[9],cam_order[15],cam_order[21],cam_order[27],cam_order[33])
        
        #attention
        attn_raw_output=self.attn4(hidden_states_in1,encoder_hidden_states=hidden_states_in2,**cross_attention_kwargs)
        # print("output of mv:",attn_raw_output.shape)#(bs*2*n_cam,:,:)
        # exit()
        
        if self.neighboring_attn_type=="self":
            attn_output=rearrange(attn_raw_output,'b (n l) ... -> b n l ...',n=self.n_cam)
        else:
            attn_output=torch.zeros_like(norm_hidden_states)
            for cam_idx in range(self.n_cam):
                attn_out_mv=rearrange(attn_raw_output[cam_order==cam_idx],'(n b) ... -> b n ...',b=B)
                # print("rearranging  for each camera:",attn_out_mv.shape) #(bs,2,n_channel,h*w) 2 is for the 2 neighboring cameras
                attn_output[:,cam_idx]=torch.sum(attn_out_mv,dim=1) #then add the attention outputs (Attn_{cv}^{r} +Attn_{cv}^{l}).
                # print("adding the left and right camera attention:",attn_out_mv.shape)
            # exit()
        attn_output=rearrange(attn_output,'b n ... -> (b n) ...')
        attn_output=self.connector(attn_output)
        hidden_states=attn_output + hidden_states
        
        # print("adding the cross attention output to mv attention op:",hidden_states.shape)
        #3:feed-forward
        # print(self.norm3)
        norm_hidden_states=self.norm3(hidden_states)
        if self.use_ada_layer_norm_zero:
            norm_hidden_states=norm_hidden_states * (1+scale_mlp[:,None]) + shift_mlp[:,None]
        ff_output=self.ff(norm_hidden_states)
        if self.use_ada_layer_norm_zero:
            ff_output=gate_mlp.unsqueeze(1)*ff_output
        hidden_states=ff_output+hidden_states
        
        # print("output of mv attention:",hidden_states.shape)
        # print("flow:norm,self attention of latents,add(attn_op+attn_ip),norm,cross attention(txt,camera,pose embeddings and latents),norm,mv attention,addition(attn_op+attn_ip),norm,linear layer")
        return (hidden_states)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
        