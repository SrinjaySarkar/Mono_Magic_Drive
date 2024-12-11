from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
import torch
import PIL
import numpy as np
from einops import rearrange
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import BaseOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

#local imports
from misc.common import move_to



@dataclass
class BEVStableDiffusionPipelineOutput(BaseOutput):
    images:Union[List[List[PIL.Image.Image]],np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


class StableDiffusionBEVControlNetPipeline(StableDiffusionControlNetPipeline):
    def __init__(self,vae: AutoencoderKL,text_encoder: CLIPTextModel,unet: UNet2DConditionModel,controlnet,scheduler:KarrasDiffusionSchedulers,tokenizer:CLIPTokenizer,safety_checker:StableDiffusionSafetyChecker=None,feature_extractor:CLIPImageProcessor=None,requires_safety_checker:bool=False):
        super().__init__(vae,text_encoder,tokenizer,unet,controlnet,scheduler,safety_checker,feature_extractor,requires_safety_checker)
        
        assert safety_checker == None, "Please do not use safety_checker."
        self.control_image_process=VaeImageProcessor(vae_scale_factor=self.vae_scale_factor,do_resize=False,do_convert_rgb=False,do_normalize=False)
        
    def numpy_to_pil_double(self,images):
        """need 5_dim to 2dim list"""
        imgs_list=[]
        for imgs in images:
            imgs_list.append(self.numpy_to_pil(imgs))
        return (imgs_list)
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta="eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs={}
        if accepts_eta:
            extra_step_kwargs["eta"]=eta
        # check if the scheduler accepts generator
        accepts_generator="generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            raise RuntimeError("If you fixed the logic for generator, please remove this. Otherwise, please use other sampler.")
            extra_step_kwargs["generator"] = generator
        return (extra_step_kwargs)
    
    def decode_latents(self,latents):
        latents=1/self.vae.config.scaling_factor * latents
        bs=len(latents)
        latents=rearrange(latents,'b c ... -> (b c) ...')
        # print(latents.shape)
        image=self.vae.decode(latents).sample
        image=rearrange(image,'(b c) ... -> b c ...',b=bs)
        # print(image.shape)
        image=(image/2+0.5).clamp(0,1)
        image=rearrange(image.cpu(),'... c h w -> ... h w c').float().numpy()
        # print(image.shape)
        return (image)
    
    
    
    
    @torch.no_grad()
    def __call__(self,prompt: Union[str, List[str]],image: torch.FloatTensor,camera_param: Union[torch.Tensor, None],height: int,width: int,num_inference_steps: int = 50,guidance_scale: float = 7.5,negative_prompt: Optional[Union[str, List[str]]] = None,num_images_per_prompt: Optional[int] = 1,eta: float = 0.0,generator: Optional[torch.Generator] = None,latents: Optional[torch.FloatTensor] = None,prompt_embeds: Optional[torch.FloatTensor] = None,negative_prompt_embeds: Optional[torch.FloatTensor] = None,output_type: Optional[str] = "pil",return_dict: bool = True,callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,callback_steps: int = 1,cross_attention_kwargs: Optional[Dict[str, Any]] = None,controlnet_conditioning_scale: float = 1,guess_mode: bool = False,use_zero_map_as_unconditional: bool = False,bev_controlnet_kwargs = {},bbox_max_length = None,):
        
        p=bev_controlnet_kwargs
        
        print("ok")
        if prompt is not None and isinstance(prompt,str):
            batch_size=1
        elif prompt is not None and isinstance(prompt,list):
            batch_size=len(prompt)
        else:
            batch_size=prompt_embeds.shape[0]
        device=self._execution_device
        do_classifier_free_guidance=guidance_scale > 1.0
        print("batchsize:",batch_size)
        print("device:",device)
        
        ### BEV, check camera_param ###
        if camera_param is None:
            # use uncond_cam and disable classifier free guidance
            N_cam=6  # TODO: hard-coded
            camera_param=self.controlnet.uncond_cam_param((batch_size,N_cam))
            do_classifier_free_guidance=False
        ### done ###
        
        #encode input prompts
        prompt_embeds=self._encode_prompt(prompt,device,num_images_per_prompt,do_classifier_free_guidance,negative_prompt,prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_prompt_embeds)
        print("embedding captions:",prompt_embeds.shape)#(2*bs,76+1,768)
        # print(prompt_embeds[0,0,0])
        # print(prompt_embeds[0,1,0])
        # print(prompt_embeds[0,2,0])
        # print(prompt_embeds[4,0,0])
        # print(prompt_embeds[5,0,0])
        # exit()
        #prep image
        assert not self.control_image_processor.config.do_normalize,"Your controlnet should not normalize the control image."
        
        print(image.shape)
        image=self.prepare_image(image=image,height=height,width=width,batch_size=batch_size*num_images_per_prompt,num_images_per_prompt=num_images_per_prompt,device=device,dtype=self.controlnet.dtype,do_classifier_free_guidance=do_classifier_free_guidance,guess_mode=guess_mode)# (2*bs,c_26,200,200)
        print(image.shape)#(bs*2,8,200,200)
        
        if use_zero_map_as_unconditional and do_classifier_free_guidance:
            _images=list(torch.chunk(image,2))
            _images[0]=torch.zeros_like_(_images[0])
            image=torch.cat(_images)
            # print(image.shape)
            # print("ok")
        
        #prep timesteps
        self.scheduler.set_timesteps(num_inference_steps,device=device)#this gets n_inference timesteps by sampling randomly between (0,n_train_timesteps) 
        # print(self.scheduler.config)
        # exit()
        
        timesteps=self.scheduler.timesteps
        print("timesteps:",timesteps)
        print("timesteps:",timesteps.shape)
        #prep latents
        # print(self.unet)
        num_channels_latents=self.unet.config.in_channels
        print(num_channels_latents)
        latents=self.prepare_latents(batch_size*num_images_per_prompt,num_channels_latents,height,width,prompt_embeds.dtype,device,generator,latents)
        # print(latents.shape)#(bs,4,28,50)
        
        #prep box camera and bev data
        extra_step_kwargs=self.prepare_extra_step_kwargs(generator,eta)
        assert camera_param.shape[0]==batch_size,f"Except {batch_size} camera params, but you have bs={len(camera_param)}"
        N_cam=camera_param.shape[1]
        latents=torch.stack([latents]*N_cam,dim=1)
        camera_param=camera_param.to(self.device)
        
        
        if do_classifier_free_guidance and not guess_mode:
            #uncond then cond
            _images=list(torch.chunk(image,2))
            # print(_images[0].shape)
            print(_images[1].shape)
            # print(bev_controlnet_kwargs)#(3d boxes(8,3),classes,masks)
            kwargs_with_uncond=self.controlnet.add_uncond_to_kwargs(camera_param=camera_param,image=_images[0],max_len=bbox_max_length,**bev_controlnet_kwargs) #(dict_keys(['camera_param', 'bboxes_3d_data', 'image']) image is the conditioning BEV map)
            # print(kwargs_with_uncond.keys())
            kwargs_with_uncond.pop("max_len",None)  # some do not take this.
            # print(kwargs_with_uncond.keys())
            camera_param=kwargs_with_uncond.pop("camera_param")
            _images[0]=kwargs_with_uncond.pop("image")
            image=torch.cat(_images)
            # print(kwargs_with_uncond.keys())
            bev_controlnet_kwargs=move_to(kwargs_with_uncond,self.device)
            # print(bev_controlnet_kwargs.keys())#dict_keys(['bboxes_3d_data'])
            
            #this appends uncond boxes ,uncond bev images,uncond camera params in the beginnning of the cond boxes, cond bev images, cond camera params. so it is the same shape as the prompt embeddings uncond_ids, cond_ids. 
        
        
        #denoising loop
        num_warmup_steps=len(timesteps)-num_inference_steps * self.scheduler.order
        print(num_warmup_steps)
        # print(self.scheduler)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps):
                # print(i,t)   
                #controlnet inference 
                latent_model_input=(torch.cat([latents]*2) if do_classifier_free_guidance else latents)
                #repeat the latents for every cam, then duplicate them again for every batch ,for uncond and cond.
                # print(latent_model_input.shape)                 
                latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)
                controlnet_t=t.unsqueeze(0)
                if guess_mode and do_classifier_free_guidance:
                    controlnet_latent_model_input=latents
                    controlnet_prompt_embeds=prompt_embeds.chunk(2)[1]
                    print(controlnet_latent_model_input.shape)
                    print(controlnet_prompt_embeds.shape)
                else:
                    controlnet_latent_model_input=latent_model_input
                    controlnet_prompt_embeds=prompt_embeds
                
                print("controlnet_latent_model:",latent_model_input.shape)
                print("prompt embeds:",controlnet_prompt_embeds.shape)
                # exit()
                print(image.shape)
                controlnet_t=controlnet_t.repeat(len(controlnet_latent_model_input))
                down_block_res_samples,mid_block_res_sample,encoder_hidden_states_with_cam=self.controlnet(controlnet_latent_model_input,controlnet_t,camera_param,encoder_hidden_states=controlnet_prompt_embeds,controlnet_cond=image,conditioning_scale=controlnet_conditioning_scale,guess_mode=guess_mode,return_dict=False,**bev_controlnet_kwargs)#bev_controlnet_kwargs contains boxes 
                print("down block controlnet op:",down_block_res_samples[0].shape)
                print("mid block controlnet op:",mid_block_res_sample.shape)
                print("encoder hidden_states",encoder_hidden_states_with_cam.shape)
                
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                if guess_mode and do_classifier_free_guidance:
                    down_block_res_samples=[torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample=torch.cat([torch.zeros_like(mid_block_res_sample),mid_block_res_sample])
                    print("down block controlnet op:",down_block_res_samples[0].shape)
                    print("mid block controlnet op:",mid_block_res_sample.shape)
                    
                    encoder_hidden_states_with_cam=self.controlnet.add_uncond_to_emb(prompt_embeds.chunk(2)[0], N_cam,encoder_hidden_states_with_cam)
                    print("encoder hidden_states",encoder_hidden_states_with_cam.shape)

                latent_model_input=rearrange(latent_model_input,'b n ... -> (b n) ...')
                latents=rearrange(latents,'b n ... -> (b n) ...')
                print(latent_model_input.shape)
                print(latents.shape)
                
                # predict the noise residual: 2bxN, 4, 28, 50
                additional_param={}
                noise_pred=self.unet(latent_model_input,t,encoder_hidden_states=encoder_hidden_states_with_cam,**additional_param, cross_attention_kwargs=cross_attention_kwargs,down_block_additional_residuals=down_block_res_samples,mid_block_additional_residual=mid_block_res_sample).sample
                
                print(noise_pred.shape)
                
                if do_classifier_free_guidance:
                    noise_pred_uncond,noise_pred_text=noise_pred.chunk(2)
                    noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)
                    print(noise_pred_uncond.shape)
                    print(noise_pred_text.shape)
                
                # compute the previous noisy sample x_t -> x_t-1
                # NOTE: is the scheduler use randomness, please handle the logic
                # for generator.
                latents=self.scheduler.step(noise_pred,t,latents,**extra_step_kwargs).prev_sample
                latents=rearrange(latents,'(b n) ... -> b n ...',n=N_cam)
                print(latents.shape)
                
                if i==len(timesteps)-1 or ((i+1)>num_warmup_steps and (i+1)%self.scheduler.order==0):
                    progress_bar.update()
                    if callback is not None and i%callback_steps==0:
                        callback(i,t,latents)        
        # exit()
        
        if hasattr(self,"final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()
        
        if output_type=="latent":
            image=latents
            has_nsfw_concept=None
        elif output_type=="pil":
            image=self.decode_latents(latents)
            image,has_nsfw_concept=self.run_safety_checker(image,device,prompt_embeds.dtype)
            image=self.numpy_to_pil_double(image)
            # print("final decoded:",len(image))
            # print(image[0][0].size)
            # print(image[0][1].size)
            # print(image[0][2].size)
            # print(image[0][3].size)
            
            # im1=image[0][1].save("sample.jpg")
            
            # print("here")
            # exit()
        else:
            image=self.decode_latents(latents)
            image,has_nsfw_concept=self.run_safety_checker(image,device,prompt_embeds.dtype)
        
        # Offload last model to CPU
        if hasattr(self,"final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,has_nsfw_concept)
        
        return BEVStableDiffusionPipelineOutput(images=image,nsfw_content_detected=has_nsfw_concept)

