import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from model.embedder import get_embedder

XYZ_MIN = [-200, -300, -20]
XYZ_RANGE = [350, 650, 80]


def normalizer(mode, data):
    if mode == 'cxyz' or mode == 'all-xyz':
        # data in format of (N, 4, 3):
        mins = torch.as_tensor(
            XYZ_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(
            XYZ_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider
    elif mode == 'owhr':
        raise NotImplementedError(f"wait for implementation on {mode}")
    else:
        raise NotImplementedError(f"not support {mode}")
    return data

class ContinuousBBoxWithTextEmbedding(torch.nn.Module):
    def __init__(self,n_classes,class_token_dim=768,trainable_class_token=False,embedder_num_freq=4,proj_dims=[768,512,512,768],mode="cxyz",minmax_normalize=True,use_text_encoder_init=True,**kwargs):
        super().__init__()
        self.mode=mode
        if self.mode=="cxyz":
            nput_dims=3
            output_num=4 #4 points
        elif self.mode=='all-xyz':
            input_dims=3
            output_num=8  # 8 points
        elif self.mode=='owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {mode}")
        
        self.minmax_normalize=minmax_normalize
        self.use_text_encoder_init=use_text_encoder_init
        self.fourier_embedder=get_embedder(input_dims,embedder_num_freq)
        logging.info(f"[ContinuousBBoxWithTextEmbedding] bbox embedder has"f"{self.fourier_embedder.out_dim} dims.")
        
        self.bbox_proj=torch.nn.Linear(self.fourier_embedder.out_dim*output_num,proj_dims[0])
        self.second_linear=torch.nn.Sequential(torch.nn.Linear(proj_dims[0]+class_token_dim,proj_dims[1]),torch.nn.SiLU(),torch.nn.Linear(proj_dims[1], proj_dims[2]),torch.nn.SiLU(),torch.nn.Linear(proj_dims[2], proj_dims[3]))
        
        #for class token
        self._class_tokens_set_or_warned=not self.use_text_encoder_init
        if trainable_class_token:
            # parameter is trainable, buffer is not
            class_tokens=torch.randn(n_classes,class_token_dim)
            self.register_parameter("_class_tokens",torch.nn.Parameter(class_tokens))
        else:
            class_tokens=torch.randn(n_classes, class_token_dim)
            self.register_buffer("_class_tokens", class_tokens)
            if not self.use_text_encoder_init:
                logging.warn(
                    "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not"
                    " trainable but you set `use_text_encoder_init` to False. "
                    "Please check your config!")
        #null embedding
        self.null_class_feature=torch.nn.Parameter(torch.zeros([class_token_dim]))
        self.null_pos_feature=torch.nn.Parameter(torch.zeros([self.fourier_embedder.out_dim*output_num]))

    @property
    def class_tokens(self):
        if not self._class_tokens_set_or_warned:
            logging.warn("[ContinuousBBoxWithTextEmbedding] Your class_tokens is not trainable and used without initialization. Please check your training code!")
            self._class_tokens_set_or_warned=True
        return (self._class_tokens)
    
    
    def prepare(self,cfg,**kwargs):
        if self.use_text_encoder_init:
            self.set_category_token(kwargs["tokenizer"],kwargs["text_encoder"],cfg.dataset.object_classes)
        else:
            logging.info("[ContinuousBBoxWithTextEmbedding] Your class_tokens initilzed with random.")
    
    @torch.no_grad()
    def set_category_token(self,tokenizer,text_encoder,class_names):
        logging.info("[ContinuousBBoxWithTextEmbedding] Initialzing your class_tokens with text_encoder")
        self._class_tokens_set_or_warned=True
        device=self.class_tokens.device
        for idx, name in enumerate(class_names):
            inputs=tokenizer([name],padding='do_not_pad',return_tensors='pt')
            inputs=inputs.input_ids.to(device)
            # there are two outputs: last_hidden_state and pooler_output
            # we use the pooled version.
            hidden_state=text_encoder(inputs).pooler_output[0]  # 768
            self.class_tokens[idx].copy_(hidden_state)
    
    # def add_n_uncond_tokens(self,hidden_states,token_num):
    def forward_feature(self,pos_emb,cls_emb):
        emb=self.bbox_proj(pos_emb)
        emb=F.silu(emb)
        # print("embedding of fourier featurs of bboxes:",emb.shape)
        
        emb=torch.cat([emb,cls_emb],dim=-1)
        # print("concat bbox embedding with cls embedding:",emb.shape)
        emb=self.second_linear(emb)
        # print("concatenated after passing through after another mlp:",emb.shape)
        return (emb)

    
    def forward(self,bboxes,classes,masks=None,**kwargs):
        
        # print("bbox embedder bboxes:",bboxes.shape)
        # print("bbox embedder classes:",classes.shape)
        # print("bbox embedder masks:",masks.shape)
        # print("bbox masks:",masks[0,:])
        #masks basically tells which of the max_len is box and which is padded .
        
        (B,N)=classes.shape
        # print(B,N)
        bboxes=rearrange(bboxes,'b n ... -> (b n) ...')
        # print("Bounding boxes:",bboxes.shape)
        
        if masks is None:
            masks=torch.ones(len(bboxes))
        else:
            masks=masks.flatten()
        masks=masks.unsqueeze(-1).type_as(self.null_pos_feature)
        # print("masks:",masks.shape)
        
        #box
        if self.minmax_normalize:
            bboxes=normalizer(self.mode,bboxes)
            # print(bboxes.shape)
        pos_emb=self.fourier_embedder(bboxes)
        # print("pos encoding of bboxes:",pos_emb.shape)
        pos_emb=pos_emb.reshape(pos_emb.shape[0],-1).type_as(self.null_pos_feature)
        # print("pos encoding reshaped:",pos_emb.shape)
        pos_emb=pos_emb*masks + self.null_pos_feature[None]*(1-masks)
        # print("pos encoding after adding masks for null positions:",pos_emb.shape)
        # print("masks:",masks[:30])
        
        #class
        cls_emb=torch.stack([self.class_tokens[i] for i in classes.flatten()])
        cls_emb=cls_emb*masks + self.null_class_feature[None]*(1-masks)
        # print("embedding of class tokens;",cls_emb.shape)
        
        #combine
        emb=self.forward_feature(pos_emb,cls_emb)
        emb=rearrange(emb,'(b n) ... -> b n ...', n=N)
        return (emb)
   