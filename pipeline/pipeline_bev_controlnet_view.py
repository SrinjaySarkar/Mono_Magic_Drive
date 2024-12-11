from dataclasses import dataclass
import sys
sys.path.append("/data/srinjay/magic_drive/")
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
from diffusers import AutoencoderKL

#local imports
from misc.common import move_to
from pipeline.pipeline_bev_controlnet import StableDiffusionBEVControlNetPipeline,BEVStableDiffusionPipelineOutput


class StableDiffusionBEVControlNetGivenViewPipeline(StableDiffusionBEVControlNetPipeline):
    @torch.no_grad()
    def __call__(self,prompt: Union[str, List[str]],image:torch.FloatTensor,camera_param:Union[torch.Tensor,None],height:int,width:int,conditional_latents: List[List[torch.FloatTensor]],conditional_latents_change_every_input=True,# view conditioned latents
    num_inference_steps: int = 50,guidance_scale: float = 7.5,negative_prompt: Optional[Union[str, List[str]]] = None,num_images_per_prompt: Optional[int] = 1,eta: float = 0.0,generator: Optional[torch.Generator] = None,latents: Optional[torch.FloatTensor] = None,prompt_embeds: Optional[torch.FloatTensor] = None,negative_prompt_embeds: Optional[torch.FloatTensor] = None,output_type: Optional[str] = "pil",return_dict: bool = True,callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,callback_steps: int = 1,cross_attention_kwargs: Optional[Dict[str, Any]] = None,controlnet_conditioning_scale: float = 1,guess_mode: bool = False,use_zero_map_as_unconditional: bool = False,bev_controlnet_kwargs = {},bbox_max_length = None):
        print(image.shape)
        print(len(conditional_latents))
        
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
        timesteps=self.scheduler.timesteps
        print("timesteps:",timesteps)
        print("timesteps:",timesteps.shape)
        print(self.scheduler.config)
        # exit()
        num_channels_latents=self.unet.config.in_channels
        print(num_channels_latents)
        latents=self.prepare_latents(batch_size*num_images_per_prompt,num_channels_latents,height,width,prompt_embeds.dtype,device,generator,latents)
        
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
        
        print("LATETNTS:",latents.shape)
        
        # self.vae=AutoencoderKL.from_pretrained("/data/srinjay/magic_drive/pretrained/stable-diffusion-v1-5/", subfolder="vae")
        # self.vae=self.vae.to(device)
        
        
        original_noise=torch.clone(latents)
        if not conditional_latents_change_every_input:
            for i in range(batch_size):
                for j in range(N_cam):
                    print(conditional_latents[i][j].shape)
                    cl=self.vae.encode(conditional_latents[i][j].unsqueeze(0).to(device)).latent_dist.sample()
                    # print(cl.shape,cl.device,latents[i][j].unsqueeze(0).shape)
                    
                    if conditional_latents[i][j] is not None:
                        _timesteps=timesteps[0]
                        noised_latent=self.scheduler.add_noise(cl.to(device),latents[i,j].unsqueeze(0).to(device),_timesteps.to(device))
                        # noised_latent=self.scheduler.add_noise(conditional_latents[i][j].unsqueeze(0).to(device),latents[i,j].unsqueeze(0).to(device),_timesteps.to(device))
                        latents[i,j]=noised_latent.type_as(latents)[0]
        
        num_warmup_steps=len(timesteps)-num_inference_steps * self.scheduler.order
        print(num_warmup_steps)
        # print(self.scheduler)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps):
                
                if conditional_latents_change_every_input:
                    for i in range(batch_size):
                        for j in range(N_cam):
                            if conditional_latents[i][j] is not None:
                                # print(conditional_latents[i][j].shape)
                                cl=self.vae.encode(conditional_latents[i][j].unsqueeze(0).to(device)).latent_dist.sample()
                                # print(cl.shape,cl.device,latents[i][j].unsqueeze(0).shape)
                                noised_latent=self.scheduler.add_noise(cl.to(device),latents[i,j].unsqueeze(0).to(device),t)
                                # noised_latent=self.scheduler.add_noise(conditional_latents[i][j].unsqueeze(0).to(device),latents[i,j].unsqueeze(0).to(device),_timesteps.to(device))
                                latents[i,j]=noised_latent.type_as(latents)[0]
                
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
                
                if do_classifier_free_guidance:
                    noise_pred_uncond,noise_pred_text=noise_pred.chunk(2)
                    noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)
                    print(noise_pred_uncond.shape)
                    print(noise_pred_text.shape)

                if not conditional_latents_change_every_input:
                    noise_pred=rearrange(noise_pred,'(b n) ... -> b n ...', n=N_cam)
                    for i in range(batch_size):
                        for j in range(N_cam):
                            if conditional_latents[i][j] is not None:
                                noise_pred[i,j]=original_noise[i, j]
                    noise_pred=rearrange(noise_pred,'b n ... -> (b n) ...')                

                latents=self.scheduler.step(noise_pred,t,latents,**extra_step_kwargs).prev_sample
                latents=rearrange(latents,'(b n) ... -> b n ...',n=N_cam)
                print(latents.shape)
                
                if i==len(timesteps)-1 or ((i+1)>num_warmup_steps and (i+1)%self.scheduler.order==0):
                    progress_bar.update()
                    if callback is not None and i%callback_steps==0:
                        callback(i,t,latents) 
        
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
        
            



# class StableDiffusionBEVControlNetGivenViewPipeline(
#         StableDiffusionBEVControlNetPipeline):
#     @torch.no_grad()
#     def __call__(
#         self,
#         prompt: Union[str, List[str]],
#         image: torch.FloatTensor,
#         camera_param: Union[torch.Tensor, None],
#         height: int,
#         width: int,
#         # add one param here.
#         # should be BxN list. conditional views are tensor (CxHxW),
#         # unconditional views are None.
#         conditional_latents: List[List[torch.FloatTensor]],
#         conditional_latents_change_every_input = True,
#         # done
#         num_inference_steps: int = 50,
#         guidance_scale: float = 7.5,
#         negative_prompt: Optional[Union[str, List[str]]] = None,
#         num_images_per_prompt: Optional[int] = 1,
#         eta: float = 0.0,
#         generator: Optional[torch.Generator] = None,
#         latents: Optional[torch.FloatTensor] = None,
#         prompt_embeds: Optional[torch.FloatTensor] = None,
#         negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#         output_type: Optional[str] = "pil",
#         return_dict: bool = True,
#         callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
#         callback_steps: int = 1,
#         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#         controlnet_conditioning_scale: float = 1,
#         guess_mode: bool = False,
#         use_zero_map_as_unconditional: bool = False,
#         bev_controlnet_kwargs = {},
#         bbox_max_length = None,
#     ):
# #         r"""
# #         Function invoked when calling the pipeline for generation.

# #         Args:
# #             prompt (`str` or `List[str]`, *optional*):
# #                 The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
# #                 instead.
# #             image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,
# #                     `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
# #                 The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
# #                 the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
# #                 also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
# #                 height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
# #                 specified in init, images must be passed as a list such that each element of the list can be correctly
# #                 batched for input to a single controlnet.
# #             height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
# #                 The height in pixels of the generated image.
# #             width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
# #                 The width in pixels of the generated image.
# #             num_inference_steps (`int`, *optional*, defaults to 50):
# #                 The number of denoising steps. More denoising steps usually lead to a higher quality image at the
# #                 expense of slower inference.
# #             guidance_scale (`float`, *optional*, defaults to 7.5):
# #                 Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
# #                 `guidance_scale` is defined as `w` of equation 2. of [Imagen
# #                 Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
# #                 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
# #                 usually at the expense of lower image quality.
# #             negative_prompt (`str` or `List[str]`, *optional*):
# #                 The prompt or prompts not to guide the image generation. If not defined, one has to pass
# #                 `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
# #                 less than `1`).
# #             num_images_per_prompt (`int`, *optional*, defaults to 1):
# #                 The number of images to generate per prompt.
# #             eta (`float`, *optional*, defaults to 0.0):
# #                 Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
# #                 [`schedulers.DDIMScheduler`], will be ignored for others.
# #             generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
# #                 One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
# #                 to make generation deterministic.
# #             latents (`torch.FloatTensor`, *optional*):
# #                 Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
# #                 generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
# #                 tensor will ge generated by sampling using the supplied random `generator`.
# #             prompt_embeds (`torch.FloatTensor`, *optional*):
# #                 Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
# #                 provided, text embeddings will be generated from `prompt` input argument.
# #             negative_prompt_embeds (`torch.FloatTensor`, *optional*):
# #                 Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
# #                 weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
# #                 argument.
# #             output_type (`str`, *optional*, defaults to `"pil"`):
# #                 The output format of the generate image. Choose between
# #                 [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
# #             return_dict (`bool`, *optional*, defaults to `True`):
# #                 Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
# #                 plain tuple.
# #             callback (`Callable`, *optional*):
# #                 A function that will be called every `callback_steps` steps during inference. The function will be
# #                 called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
# #             callback_steps (`int`, *optional*, defaults to 1):
# #                 The frequency at which the `callback` function will be called. If not specified, the callback will be
# #                 called at every step.
# #             cross_attention_kwargs (`dict`, *optional*):
# #                 A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
# #                 `self.processor` in
# #                 [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
# #             controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
# #                 The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
# #                 to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
# #                 corresponding scale as a list.
# #             guess_mode (`bool`, *optional*, defaults to `False`):
# #                 In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
# #                 you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.

# #         Examples:

# #         Returns:
# #             [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
# #             [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
# #             When returning a tuple, the first element is a list with the generated images, and the second element is a
# #             list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
# #             (nsfw) content, according to the `safety_checker`.
# #         """
# #         # 0. Default height and width to unet
# #         # BEV: we cannot use the size of image
# #         # height, width = self._default_height_width(height, width, None)

# #         # 1. Check inputs. Raise error if not correct
# #         # we do not need this, only some type assertion
# #         # self.check_inputs(
# #         #     prompt,
# #         #     image,
# #         #     height,
# #         #     width,
# #         #     callback_steps,
# #         #     negative_prompt,
# #         #     prompt_embeds,
# #         #     negative_prompt_embeds,
# #         #     controlnet_conditioning_scale,
# #         # )

# #         # 2. Define call parameters
# #         # NOTE: we get batch_size first from prompt, then align with it.
# #         if prompt is not None and isinstance(prompt, str):
# #             batch_size = 1
# #         elif prompt is not None and isinstance(prompt, list):
# #             batch_size = len(prompt)
# #         else:
# #             batch_size = prompt_embeds.shape[0]

# #         device = self._execution_device
# #         # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
# #         # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
# #         # corresponds to doing no classifier free guidance.
# #         do_classifier_free_guidance = guidance_scale > 1.0