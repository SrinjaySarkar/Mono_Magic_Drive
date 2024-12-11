import logging
import os
from pprint import pprint
import contextlib
from omegaconf import OmegaConf
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import (
    ModelMixin,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler
from misc.common import load_module, convert_outputs_to_fp16, move_to
from runner.base_runner import BaseRunner
from runner.utils import smart_param_count


class ControlnetUnetWrapper(ModelMixin):
    def __init__(self,controlnet,unet,weight_dtype=torch.float32,unet_in_fp16=True):
        super().__init__()
        self.controlnet=controlnet
        self.unet=unet
        self.weight_dtype=weight_dtype
        self.unet_in_fp16=unet_in_fp16
    
    def forward(self,noisy_latents,timesteps,camera_param,encoder_hidden_states,encoder_hidden_states_uncond,controlnet_image,**kwargs):
        # print("forward of complete BEVControlNet + MV-UNET")
        
        # print("controlnet:",self.controlnet)# BEVControlNetModel
        # print("unet:",self.unet)#UNet2DConditionModelMultiview    
        N_cam=noisy_latents.shape[1]
        kwargs=move_to(kwargs,self.weight_dtype,lambda x: x.dtype==torch.float32)
        # print(kwargs)
        # print("Cameras ",N_cam)
        # e1=encoder_hidden_states_with_cam[0]
        
        #fmt:off
        down_block_res_samples,mid_block_res_samples,encoder_hidden_states_with_cam=self.controlnet(noisy_latents,#b,n_cam,4,H/8,W/8
        timesteps,camera_param=camera_param, #b,n_cam,3,(4+3)
        encoder_hidden_states=encoder_hidden_states,#b,n_tokens,768(token_embedding)
        encoder_hidden_states_uncond=encoder_hidden_states_uncond,#1,n_token,768
        controlnet_cond=controlnet_image,#b,8,200,200
        return_dict=False,**kwargs)
    
        
        # print("###################################################### END OF CONTROLNET  FORWARD ########################################")        
        # print("down block ops:",len(down_block_res_samples)) # (12 3*4 , there are 4 blocks with 3 layers in them)
        # print("mid block samples:",len(mid_block_res_samples)) # 1 (there is only 1 mid block)
        # print("encoder with cam:",encoder_hidden_states_with_cam.shape) #( bs*n_cam, n_token+n_boxes,768)
        # print("encoder with cam:",len(encoder_hidden_states_with_cam))
        
        
        # e2=encoder_hidden_states_with_cam[0]
        
        #fmt:on
        # starting from here, we use (B n) as batch_size
        noisy_latents=rearrange(noisy_latents,"b n ... -> (b n) ...")
        # print("rearrange to bs*n_cam noisy cams:",noisy_latents.shape)
        
        if timesteps.ndim==1:
            timesteps=repeat(timesteps,"b -> (b n)", n=N_cam)
        # print("replicate timesteps for all cameras:",timesteps.shape)
        
        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        context=contextlib.nullcontext
        context_kwargs={}
        if self.unet_in_fp16:
            context=torch.cuda.amp.autocast
            context_kwargs={"enabled":False}
        
        # print(context_kwargs)
        # exit()
        
        # print("#################### INPUT TO MV UNET #################################")
        # print("noisy latents:",noisy_latents.shape)
        # print("enocder_hidden_states , (combined text,camera,bbox embedding):",encoder_hidden_states_with_cam.shape)
        # print("down block ops:",down_block_res_samples[0].shape)
        # print("down block samples:",len(down_block_res_samples))
        # print("mid block samples:",mid_block_res_samples[0].shape)
        # print("mid block samples:",len(mid_block_res_samples))
        with context(**context_kwargs):
            
            # predict the noisy samples
            model_pred=self.unet(noisy_latents,# bs,4,H/8, W/8
            timesteps.reshape(-1),  # bs
            encoder_hidden_states=encoder_hidden_states_with_cam.to(
            dtype=self.weight_dtype), # bs,len + 1,768
            # TODO: during training, some camera param are masked.
            down_block_additional_residuals=[sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples], # all intermedite have four dims: bs,c,h,w
            mid_block_additional_residual=mid_block_res_samples.to(dtype=self.weight_dtype)).sample
            
        # print("#################### END TO MV UNET #################################")
        # print(model_pred.shape)
        # exit()
            
        # print("Output After running Controlnet and then MV Unet:",model_pred.shape)
        model_pred=rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)
        # print("Output After running Controlnet and then MV Unet then reshape latents:",model_pred.shape)
        return (model_pred)


class MultiviewRunner(BaseRunner):
    def __init__(self,cfg,accelerator,train_set,val_set)->None:
        super().__init__(cfg,accelerator,train_set,val_set)
    
    def _init_fixed_models(self,cfg):
        #fmt:off
        self.tokenizer=CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder=CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae=AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
        self.noise_scheduler=DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")

    def _init_trainable_models(self,cfg):
        # fmt:off
        unet=UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model_name_or_path,subfolder="unet") #pretrained ldm unet(not condition unet) sdv1.5
        # fmt:on
        #this is the multiview unet module defined for this
        model_cls=load_module(cfg.model.unet_module)#magicdrive.networks.unet_2d_condition_multiview.UNet2DConditionModelMultiview
        unet_param=OmegaConf.to_container(self.cfg.model.unet,resolve=True)#oarams of the mv-unet
        self.unet=model_cls.from_unet_2d_condition(unet,**unet_param)#Instantiate Multiview unet class from UNet2DConditionModel.
        # print("############################",self.unet)
        
        model_cls=load_module(cfg.model.model_module)#magicdrive.networks.unet_addon_rawbox.BEVControlNetModel
        controlnet_param=OmegaConf.to_container(self.cfg.model.controlnet,resolve=True)
        self.controlnet=model_cls.from_unet(unet,**controlnet_param) #Instantiate BEVControlnet class from UNet2DConditionModel(pretrained ldm).
    
    def _set_model_trainable_state(self,train=True):
        self.vae.requires_grad_(False) # frozed encoder for inputs
        self.text_encoder.requires_grad_(False) # frozen text encoder
        self.controlnet.train(train) # only finetune ldm init model
        self.unet.requires_grad_(False) # dont finetune the multi-view model  
        for name,mod in self.unet.trainable_module.items():
            logging.debug(f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)
    
    def set_optimizer_schedule(self):
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("To use 8-bit Adam, please install the bitsandbytes library")

            optimizer_class=bnb.optim.AdamW8bit
        else:
            optimizer_class=torch.optim.AdamW
        params_to_optimize=list(self.controlnet.parameters())#only optimize the bevcontrolnet params
        unet_params=self.unet.trainable_parameters
        param_count=smart_param_count(unet_params)
        logging.info(
            f"[MultiviewRunner] add {param_count} params from unet to optimizer.")
        params_to_optimize+= unet_params # optimize the bevcontrolnet and multiview controlnet
        self.optimizer=optimizer_class(params_to_optimize,lr=self.cfg.runner.learning_rate,betas=(self.cfg.runner.adam_beta1,self.cfg.runner.adam_beta2),weight_decay=self.cfg.runner.adam_weight_decay,eps=self.cfg.runner.adam_epsilon)

        self._calculate_steps()
        self.lr_scheduler=get_scheduler(self.cfg.runner.lr_scheduler,optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power)
    
    def prepare_device(self):
        self.controlnet_unet=ControlnetUnetWrapper(self.controlnet,self.unet)# unet is multi-view unet,controlnet is bevcontrolnet full arch
        # Eaccelerator
        ddp_modules=(self.controlnet_unet,self.optimizer,self.train_dataloader,self.lr_scheduler)
        ddp_modules=self.accelerator.prepare(*ddp_modules)
        (self.controlnet_unet,self.optimizer,self.train_dataloader,self.lr_scheduler)=ddp_modules
        
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision=="fp16":
            self.weight_dtype=torch.float16
        elif self.accelerator.mixed_precision=="bf16":
            self.weight_dtype=torch.bfloat16    
        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype==torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                for name,mod in self.unet.trainable_module.items():#multiview unet
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    mod._original_forward=mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward=torch.cuda.amp.autocast(dtype=torch.float16)(mod.forward)
                    # we ensure output is always fp16
                    mod.forward=convert_outputs_to_fp16(mod.forward)
            else:
                 raise TypeError("There is an error/bug in accumulation wrapper, please make all trainable param in fp32.")
        
        controlnet_unet=self.accelerator.unwrap_model(self.controlnet_unet)
        controlnet_unet.weight_dtype=self.weight_dtype
        controlnet_unet.unet_in_fp16=self.cfg.runner.unet_in_fp16            
        
        with torch.no_grad():
            self.accelerator.unwrap_model(self.controlnet).prepare(self.cfg,tokenizer=self.tokenizer,text_encoder=self.text_encoder)
        self._calculate_steps()
    
    def _save_model(self,root=None):
        if root is None:
            root=self.cfg.log_root
        controlnet=self.accelerator.unwrap_model(self.controlnet)
        controlnet.save_pretrained(os.path.join(root, self.cfg.model.controlnet_dir))#controlnet with unet (ldm init)
        unet=self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))#mutli-view unet
        logging.info(f"Save your model to: {root}")
    
    def _train_one_stop(self,batch):
        #check collate_fn in dataset.utils for keys in batch, they are made from the keys in the dataset itself.
        original_dataset=self.train_dataset
        
        # print(original_dataset[0].keys())
        
        
        
        # for keys in original_dataset[0].keys():
        #     if keys=="gt_bboxes_3d":
        #         pass
        #         # print(keys,original_dataset[0][keys].data["gt_boxes_3d"].shape)
        #         # print(keys,original_dataset[0][keys].data["gt_boxes_3d"])
        #     elif keys=="metas":
        #         print(original_dataset[0]["metas"])
        #     else:
        #         print(keys,original_dataset[0][keys].data.shape)
        # gt_bboxes_3d=LiDARInstance3DBoxes(gt_bboxes_3d,box_dim=gt_bboxes_3d.shape[-1],origin=(0.5,0.5,0)).convert_to(self.box_mode_3d)
        # anns_results=dict(gt_bboxes_3d=gt_bboxes_3d,gt_labels_3d=gt_labels_3d,gt_names=gt_names_3d)
        
        self.controlnet_unet.train() #(controlnet+unet)
        # print("%"*100)
        # print("BATCH:",batch.keys())
        # print("bev map aux",batch['bev_map_with_aux'].shape) #(200,200,8)
        # print("camera params",batch['camera_param'].shape) #(3,7)(3 intrinsics + 4extrinsics)
        # print("kwargs",batch['kwargs'].keys()) #(bboxes_data,(6*max_boxes in batch))
        # all boxes are projected to the 6 views , and padded with max_len . 
        #(meta data with (bboxes_3d_data,gt_labels_3d,lidar2ego,lidar2camera,camera2lidar,lidar2image,img_aug_matrix,))  
        # print("pixel values",batch['pixel_values'].shape)#(224,400,3)
        # print("captions",batch['captions']) #(captions list)
        # print("input ids",batch['input_ids'].shape) #(input ids for the word-tokens of the captions)
        # print("uncond ids",batch['uncond_ids'].shape) #(uncond ids for the end and start tokens)
        # print("meta data",batch['meta_data'].keys()) #(timeofday,location,description,camera images filename,scene token)
        
        
        # for keys in batch.keys():
        #     if isinstance(batch[keys],dict):
        #         for key in batch[keys]:
        #             print(keys,key,batch[keys][key].shape)
        #     else:
        #         print(keys,batch[keys].shape)
        
        
        
        with self.accelerator.accumulate(self.controlnet_unet):
            N_cam=batch["pixel_values"].shape[1]
            
            #1 :convert image to latent space
            latents=self.vae.encode(rearrange(batch["pixel_values"],"b n c h w -> (b n) c h w").to(dtype=self.weight_dtype)).latent_dist.sample()
            latents=latents*self.vae.config.scaling_factor
            latents=rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)
            
            #images:(bs,n_view,3,h,w)
            #latents:(bs,n_view,4h//4w//8)
            
            # print("original images:",batch["pixel_values"].shape)
            # print("LATENTS:",latents.shape)
            
            #2: embed camera params (ba,6,3,4+3)4 from (r,t) of cam2lidar + 3 intriniscs, out (B, 6, 189)
            camera_param=batch["camera_param"].to(self.weight_dtype)
            # print("Original cameras:",camera_param.shape)
            # exit()
            #original camera
        
            # 3: sample noise that we'll add to the latents
            noise=torch.randn_like(latents) #(bs,n_cam,4,h//8,w//8)
            # make sure we use same noise for different views, also take the when conditioning on the front image
            if self.cfg.model.train_with_same_noise:
                noise=repeat(noise[:, 0],"b ... -> b r ...",r=N_cam)
            bsz=latents.shape[0]
            # print("noise to add latents:",noise.shape)
            # print("bsz:",bsz)
            
            # exit()
            
            # Sample a random timestep for each image
            if self.cfg.model.train_with_same_t:
                timesteps=torch.randint(0,self.noise_scheduler.config.num_train_timesteps,(bsz,),device=latents.device)
            else:
                timesteps=torch.stack([torch.randint(0,self.noise_scheduler.config.num_train_timesteps,(bsz,),device=latents.device) for _ in range(N_cam)], dim=1)
            timesteps=timesteps.long()
            
            # print("timestep for each sample in batch: same timestep for each view",timesteps.shape)
            # print("timestep for each sample in batch: same timestep for each view",timesteps)
            # exit()
            
            # 4 : add noise
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents=self._add_noise(latents,noise,timesteps)
            # print("Noisy latents:",noisy_latents.shape)
            
            # 5: get text embedding (batch["input_ids"])
            # Get the text embedding for conditioning
            encoder_hidden_states=self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond=self.text_encoder(batch["uncond_ids"])[0]
            
            # print("Caption Tokens",batch["input_ids"].shape)#text embedding.
            # print("Start and end padding tokens",batch["uncond_ids"].shape)#text embedding.
            # print("Caption Tokens",batch["input_ids"])#text embedding.
            # print("Start and end padding tokens",batch["uncond_ids"])#text embedding.
            # check the uncond and cond text embeddings
            # print("token embedding of each token in caption:",encoder_hidden_states.shape)
            # print("token embedding of each padding token (start,end):",encoder_hidden_states_uncond.shape)
            # exit()  
            
            # exit()
            #6 : conditiioning image
            controlnet_image=batch["bev_map_with_aux"].to(dtype=self.weight_dtype)
            # print("bev image:",controlnet_image.shape) #(bs,8 semantic classes, 200,200)
            
            # print(batch["kwargs"])
            # exit()
            #for input to the controlnet we have (6 images encodings , 6 timesteps , 6 camera params , caption token embeddings(both actual captions and padding tokens) , bev map encoding)
            #forward pass thruigh controlnet then the multi-view unet
            model_pred=self.controlnet_unet(noisy_latents,timesteps,camera_param,encoder_hidden_states,encoder_hidden_states_uncond,controlnet_image,**batch["kwargs"])
            # print("MODEL PRED:",model_pred.shape)
            # print(self.noise_scheduler.config.prediction_type) #epsilon
            # exit()  
            
            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type=="epsilon":
                target=noise
            elif self.noise_scheduler.config.prediction_type=="v_prediction":
                target=self.noise_scheduler.get_velocity(latents,noise,timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            loss=F.mse_loss(model_pred.float(),target.float(),reduction='none')
            loss=loss.mean()
            
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip=self.controlnet_unet.parameters()
                self.accelerator.clip_grad_norm_(params_to_clip,self.cfg.runner.max_grad_norm)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=self.cfg.runner.set_grads_to_none)
            # exit()
        
        return (loss)
            
            
            