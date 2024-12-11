import numpy as np#from nose.tools
from pprint import pprint
from typing import Tuple, Union, List
from omegaconf import OmegaConf
from PIL import Image
from typing import Tuple, Union
import os
import cv2
import copy
import logging
import tempfile
from PIL import Image
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from accelerate.scheduler import AcceleratedScheduler
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera
from mmdet3d.datasets import build_dataset
from diffusers import UniPCMultistepScheduler
from accelerate.utils import set_seed
from functools import partial
from torchvision.transforms.functional import to_pil_image

#local imports
from dataset import collate_fn,ListSetWrapper,FolderSetWrapper
from pipeline.pipeline_bev_controlnet import (StableDiffusionBEVControlNetPipeline,BEVStableDiffusionPipelineOutput)
from pipeline.pipeline_bev_controlnet_view import (StableDiffusionBEVControlNetGivenViewPipeline)
from misc.common import load_module
from runner.utils import (visualize_map,img_m11_to_01,show_box_on_views)


def new_local_seed(global_generator):
    local_seed=torch.randint(0x7ffffffffffffff0,[1],generator=global_generator).item()
    logging.debug(f"Using seed: {local_seed}")
    return (local_seed)

def update_progress_bar_config(pipe, **kwargs):
    if hasattr(pipe,"_progress_bar_config"):
        config=pipe._progress_bar_config
        config.update(kwargs)
    else:
        config=kwargs
    pipe.set_progress_bar_config(**config)

def setup_logger_seed(cfg):
    #### setup logger ####
    # only log debug info to log file
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    # handle log from some packages
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("accelerate.tracking").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.WARN)
    logging.getLogger("PIL").setLevel(logging.WARN)
    logging.getLogger("matplotlib").setLevel(logging.WARN)
    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)
    set_seed(cfg.seed)


def box_center_shift(bboxes: LiDARInstance3DBoxes, new_center):
    raw_data = bboxes.tensor.numpy()
    new_bboxes = LiDARInstance3DBoxes(
        raw_data, box_dim=raw_data.shape[-1], origin=new_center)
    return new_bboxes

def draw_box_on_imgs(cfg, idx, val_input, ori_imgs, transparent_bg=False) -> Tuple[Image.Image, ...]:
    if transparent_bg:
        in_imgs = [Image.new('RGB', img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs
    out_imgs = show_box_on_views(
        OmegaConf.to_container(cfg.dataset.object_classes, resolve=True),
        in_imgs,
        val_input['meta_data']['gt_bboxes_3d'][idx].data,
        val_input['meta_data']['gt_labels_3d'][idx].data.numpy(),
        val_input['meta_data']['lidar2image'][idx].data.numpy(),
        val_input['meta_data']['img_aug_matrix'][idx].data.numpy(),
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(Image.fromarray(
                (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)))
    return out_imgs



def show_box_on_views(classes, images: Tuple[Image.Image, ...],
                      boxes: LiDARInstance3DBoxes, labels, transform,
                      aug_matrix=None):
    # in `third_party/bevfusion/mmdet3d/datasets/nuscenes_dataset.py`, they use
    # (0.5, 0.5, 0) as center, however, visualize_camera assumes this center.
    bboxes_trans = box_center_shift(boxes, (0.5, 0.5, 0.5))

    vis_output = []
    for idx, img in enumerate(images):
        image = np.asarray(img)
        # the color palette for `visualize_camera` is RGB, but they draw on BGR.
        # So we make our image to BGR. This can match their color.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        trans = transform[idx]
        if aug_matrix is not None:
            trans = aug_matrix[idx] @ trans
        # mmdet3d can only save image to file.
        temp_path = tempfile.mktemp(dir=".tmp", suffix=".png")
        img_out = visualize_camera(
            temp_path, image=image, bboxes=bboxes_trans, labels=labels,
            transform=trans, classes=classes, thickness=1,
        )
        img_out = np.asarray(Image.open(temp_path))  # ensure image is loaded
        vis_output.append(Image.fromarray(img_out))
        os.remove(temp_path)
    return vis_output

def build_pipe(cfg,device):
    
    # print("cfg:",cfg)
    # exit()
    weight_dtype=torch.float16
    if cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint=cfg.resume_from_checkpoint[:-1]
    pipe_param={}
    model_cls=load_module(cfg.model.model_module)
    print("MODEL CLS:",model_cls)
    controlnet_path=os.path.join(cfg.resume_from_checkpoint,cfg.model.controlnet_dir)
    print("cntrl path:",controlnet_path)
    
    logging.info(f"Loading controlnet from {controlnet_path} with {model_cls}")
    #Loading controlnet from ./pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/controlnet with <class 'model.unet_addon_rawbox.BEVControlNetModel'>
    controlnet=model_cls.from_pretrained(controlnet_path,torch_dtype=weight_dtype)
    controlnet.eval()# from_pretrained will set to eval mode by default
    pipe_param["controlnet"]=controlnet
    
    if hasattr(cfg.model,"unet_module"):
        # print("adding the mv transoformer block")
        unet_cls=load_module(cfg.model.unet_module)
        unet_path=os.path.join(cfg.resume_from_checkpoint,cfg.model.unet_dir)
        logging.info(f"Loading unet from {unet_path} with {unet_cls}")
        #Loading unet from ./pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/unet with <class 'model.unet_2d_condition_multiview.UNet2DConditionModelMultiview'>
        unet=unet_cls.from_pretrained(unet_path,torch_dtype=weight_dtype)
        unet.eval()
        pipe_param["unet"]=unet
    
    pipe_cls=load_module(cfg.model.pipe_module)
    logging.info(f"Build pipeline with {pipe_cls}")
    # Build pipeline with <class 'pipeline.pipeline_bev_controlnet.StableDiffusionBEVControlNetPipeline'>
    pipe=pipe_cls.from_pretrained(cfg.model.pretrained_model_name_or_path,**pipe_param,safety_checker=None,feature_extractor=None,torch_dtype=weight_dtype)
    
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    if cfg.runner.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    pipe=pipe.to(device)
    return (pipe,weight_dtype)

def build_pipe_view(cfg,device):
    weight_dtype=torch.float16
    if cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint=cfg.resume_from_checkpoint[:-1]
    pipe_param={}
    model_cls=load_module(cfg.model.model_module)
    print("MODEL CLS:",model_cls)
    controlnet_path=os.path.join(cfg.resume_from_checkpoint,cfg.model.controlnet_dir)
    print("cntrl path:",controlnet_path)
    
    logging.info(f"Loading controlnet from {controlnet_path} with {model_cls}")
    #Loading controlnet from ./pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/controlnet with <class 'model.unet_addon_rawbox.BEVControlNetModel'>
    controlnet=model_cls.from_pretrained(controlnet_path,torch_dtype=weight_dtype)
    controlnet.eval()# from_pretrained will set to eval mode by default
    pipe_param["controlnet"]=controlnet
    
    if hasattr(cfg.model,"unet_module"):
        # print("adding the mv transoformer block")
        unet_cls=load_module(cfg.model.unet_module)
        unet_path=os.path.join(cfg.resume_from_checkpoint,cfg.model.unet_dir)
        logging.info(f"Loading unet from {unet_path} with {unet_cls}")
        #Loading unet from ./pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/unet with <class 'model.unet_2d_condition_multiview.UNet2DConditionModelMultiview'>
        unet=unet_cls.from_pretrained(unet_path,torch_dtype=weight_dtype)
        unet.eval()
        pipe_param["unet"]=unet
    
    pipe_cls=load_module(cfg.model.pipe_view_module)
    logging.info(f"Build pipeline with {pipe_cls}")
    # Build pipeline with <class 'pipeline.pipeline_bev_controlnet.StableDiffusionBEVControlNetPipeline'>
    pipe=pipe_cls.from_pretrained(cfg.model.pretrained_model_name_or_path,**pipe_param,safety_checker=None,feature_extractor=None,torch_dtype=weight_dtype)
    
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    if cfg.runner.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    pipe=pipe.to(device)
    return (pipe,weight_dtype)
    

def prepare_all_view(cfg,device="cuda",need_loader=True):
    assert cfg.resume_from_checkpoint is not None,"Please set model to load"
    setup_logger_seed(cfg)
    pipe,weight_dtype=build_pipe_view(cfg,device)
    print(pipe)
    
    update_progress_bar_config(pipe,leave=False)
    if not need_loader:
        return (pipe,weight_dtype)
    if cfg.runner.validation_index=="demo":
        val_dataset=FolderSetWrapper("demo/data")
    else:
        # print("pprint",cfg.dataset)
        # exit()
        val_dataset=build_dataset(OmegaConf.to_container(cfg.dataset.data.val,resolve=True))
        # exit()
        if cfg.runner.validation_index!="all":
            val_dataset=ListSetWrapper(val_dataset,cfg.runner.validation_index)
    # dataloader
    collate_fn_param={"tokenizer":pipe.tokenizer,"template":cfg.dataset.template,"bbox_mode":cfg.model.bbox_mode,"bbox_view_shared":cfg.model.bbox_view_shared,"bbox_drop_ratio":cfg.runner.bbox_drop_ratio,"bbox_add_ratio":cfg.runner.bbox_add_ratio,"bbox_add_num":cfg.runner.bbox_add_num}
    val_dataloader=torch.utils.data.DataLoader(val_dataset,shuffle=False,collate_fn=partial(collate_fn,is_train=False,**collate_fn_param),
    batch_size=cfg.runner.validation_batch_size,num_workers=cfg.runner.num_workers)
    
    return (pipe,val_dataloader,weight_dtype)
    
    
def prepare_all(cfg,device="cuda",need_loader=True):
    # print("cfg:",cfg)
    # exit()
    assert cfg.resume_from_checkpoint is not None,"Please set model to load"
    setup_logger_seed(cfg)
    pipe,weight_dtype=build_pipe(cfg,device)
    # exit()
    # print(pipe)
    update_progress_bar_config(pipe,leave=False)
    
    # print(cfg.dataset)
    # exit()
    
    if not need_loader:
        return (pipe,weight_dtype)
    if cfg.runner.validation_index=="demo":
        val_dataset=FolderSetWrapper("demo/data")
    else:
        # print("pprint",cfg.dataset)
        # exit()
        val_dataset=build_dataset(OmegaConf.to_container(cfg.dataset.data.val,resolve=True))
        # exit()
        if cfg.runner.validation_index!="all":
            print(cfg.runner.validation_index)
            val_dataset=ListSetWrapper(val_dataset,cfg.runner.validation_index)
    # dataloader
    collate_fn_param={"tokenizer":pipe.tokenizer,"template":cfg.dataset.template,"bbox_mode":cfg.model.bbox_mode,"bbox_view_shared":cfg.model.bbox_view_shared,"bbox_drop_ratio":cfg.runner.bbox_drop_ratio,"bbox_add_ratio":cfg.runner.bbox_add_ratio,"bbox_add_num":cfg.runner.bbox_add_num}
    
    val_dataloader=torch.utils.data.DataLoader(val_dataset,shuffle=False,collate_fn=partial(collate_fn,is_train=False,**collate_fn_param),
    batch_size=cfg.runner.validation_batch_size,num_workers=cfg.runner.num_workers)
    
    return (pipe,val_dataloader,weight_dtype)

def run_one_batch_pipe_view(cfg,pipe:StableDiffusionBEVControlNetGivenViewPipeline,pixel_values:torch.FloatTensor,captions:Union[str,List[str]],bev_map_with_aux:torch.FloatTensor,camera_param:Union[torch.Tensor,None],bev_controlnet_kwargs:dict,global_generator=None):
    pass
    print("run one batch pipe view")
    if isinstance(captions,str):
        batch_size=1
    else:
        batch_size=len(captions)
    print(captions)
    # let different prompts have the same random seed
    if cfg.seed is None:
        generator=None
    else:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator=[]
                for _ in range(batch_size):
                    local_seed=new_local_seed(global_generator)
                    generator.append(torch.manual_seed(local_seed))
            else:
                local_seed=new_local_seed(global_generator)
                generator=torch.manual_seed(local_seed)
        else:
            print(1)
            if cfg.fix_seed_within_batch:
                print(2)
                generator=[torch.manual_seed(cfg.seed) for _ in range(batch_size)]
            else:
                print(3)
                generator=torch.manual_seed(cfg.seed)

    gen_imgs_list=[[] for _ in range(batch_size)]
    for ti in range(cfg.runner.validation_times):
        print(ti,bev_map_with_aux.shape,pipe)
        print("pixel_values:",pixel_values.shape)
        cond=[[pixel_values]]
        image : BEVStableDiffusionPipelineOutput=pipe(prompt=captions,image=bev_map_with_aux,camera_param=camera_param,height=cfg.dataset.image_size[0],width=cfg.dataset.image_size[1],conditional_latents=pixel_values,generator=generator,bev_controlnet_kwargs=bev_controlnet_kwargs,**cfg.runner.pipeline_param)
        
        # print("FINAL:",image.shape)
        image:List[List[Image.Image]]=image.images
        for bi,imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return (gen_imgs_list)
    

def run_one_batch_pipe(cfg,pipe:StableDiffusionBEVControlNetPipeline,pixel_values:torch.FloatTensor,captions:Union[str,List[str]],bev_map_with_aux:torch.FloatTensor,camera_param:Union[torch.Tensor,None],bev_controlnet_kwargs:dict,global_generator=None):
    if isinstance(captions,str):
        batch_size=1
    else:
        batch_size=len(captions)
    # print(captions)
    # let different prompts have the same random seed
    if cfg.seed is None:
        generator=None
    else:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator=[]
                for _ in range(batch_size):
                    local_seed=new_local_seed(global_generator)
                    generator.append(torch.manual_seed(local_seed))
            else:
                local_seed=new_local_seed(global_generator)
                generator=torch.manual_seed(local_seed)
        else:
            print(1)
            if cfg.fix_seed_within_batch:
                print(2)
                generator=[torch.manual_seed(cfg.seed) for _ in range(batch_size)]
            else:
                print(3)
                generator=torch.manual_seed(cfg.seed)
    # print(generator)
    gen_imgs_list=[[] for _ in range(batch_size)]
    for ti in range(cfg.runner.validation_times):
        print(ti,bev_map_with_aux.shape)
        image : BEVStableDiffusionPipelineOutput=pipe(prompt=captions,image=bev_map_with_aux,camera_param=camera_param,height=cfg.dataset.image_size[0],width=cfg.dataset.image_size[1],generator=generator,bev_controlnet_kwargs=bev_controlnet_kwargs,**cfg.runner.pipeline_param)
        
        # print("FINAL:",image.shape)
        image:List[List[Image.Image]]=image.images
        for bi,imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return (gen_imgs_list)
        

def run_one_batch(cfg,pipe,val_input,weight_dtype,global_generator=None,run_one_batch_pipe_func=run_one_batch_pipe,transparent_bg=False,map_size=400):
    bs=len(val_input['meta_data']['metas'])
    print("bs:",bs)
    
    #all 6 images for all batches
    image_names=[val_input["meta_data"]["metas"][i].data["filename"] for i in range(bs)]
    print(len(image_names))
    print(f"Caption: {val_input['captions']}")
    logging.debug(f"Caption: {val_input['captions']}")
    # print(val_input["bev_map_with_aux"].shape)
    # exit()
    # map
    map_imgs=[]
    for i,bev_map in enumerate(val_input["bev_map_with_aux"]):
        map_img_np=visualize_map(cfg, bev_map,target_size=map_size)
        # print(map_img_np.shape)
        # cv2.imwrite("./sample_map_{}.png".format(i),map_img_np)
        map_imgs.append(Image.fromarray(map_img_np))
    
    ori_imgs=[None for bi in range(bs)]
    ori_imgs_with_box=[None for bi in range(bs)]
    if val_input["pixel_values"] is not None:
        ori_imgs=[[to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i])) for i in range(6)]for bi in range(bs)]
        if cfg.show_box:
            ori_imgs_with_box=[draw_box_on_imgs(cfg,bi,val_input,ori_imgs[bi],transparent_bg=transparent_bg) for bi in range(bs)]
    
    camera_param=val_input["camera_param"].to(weight_dtype)
    # 3-dim list: B,Times,views
    print(val_input.keys())
    gen_imgs_list=run_one_batch_pipe_func(cfg,pipe,val_input['pixel_values'],val_input['captions'],val_input['bev_map_with_aux'],camera_param, val_input['kwargs'],global_generator=global_generator)
    # import pdb
    # pdb.set_trace()
    print(len(gen_imgs_list))
    # save gen with box
    gen_imgs_wb_list=[]
    if cfg.show_box:
        for bi,images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append([draw_box_on_imgs(cfg,bi,val_input,images[ti],transparent_bg=transparent_bg) for ti in range(len(images))])
    else:
        for bi,images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append(None)
    return (map_imgs,ori_imgs,ori_imgs_with_box,gen_imgs_list,gen_imgs_wb_list)


def run_one_batch_view(cfg,pipe,val_input,weight_dtype,global_generator=None,run_one_batch_pipe_func=run_one_batch_pipe_view,transparent_bg=False,map_size=400):
    bs=len(val_input['meta_data']['metas'])
    # print("bs:",bs)
    # print(run_one_batch_pipe_func)
    
    #all 6 images for all batches
    image_names=[val_input["meta_data"]["metas"][i].data["filename"] for i in range(bs)]
    # print(len(image_names))
    print(f"Caption: {val_input['captions']}")
    logging.debug(f"Caption: {val_input['captions']}")
    # print(val_input["bev_map_with_aux"].shape)
    # exit()
    # map
    map_imgs=[]
    for i,bev_map in enumerate(val_input["bev_map_with_aux"]):
        map_img_np=visualize_map(cfg, bev_map,target_size=map_size)
        # print(map_img_np.shape)
        # cv2.imwrite("./sample_map_{}.png".format(i),map_img_np)
        map_imgs.append(Image.fromarray(map_img_np))
    
    ori_imgs=[None for bi in range(bs)]
    ori_imgs_with_box=[None for bi in range(bs)]
    if val_input["pixel_values"] is not None:
        print("there are images")
        ori_imgs=[[to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i])) for i in range(6)]for bi in range(bs)]
        if cfg.show_box:
            ori_imgs_with_box=[draw_box_on_imgs(cfg,bi,val_input,ori_imgs[bi],transparent_bg=transparent_bg) for bi in range(bs)]
    
    camera_param=val_input["camera_param"].to(weight_dtype)
    # 3-dim list: B,Times,views
    print(val_input.keys())
    gen_imgs_list=run_one_batch_pipe_func(cfg,pipe,val_input['pixel_values'],val_input['captions'],val_input['bev_map_with_aux'],camera_param, val_input['kwargs'],global_generator=global_generator)
    
    
    print(len(gen_imgs_list))
    # save gen with box
    gen_imgs_wb_list=[]
    if cfg.show_box:
        for bi,images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append([draw_box_on_imgs(cfg,bi,val_input,images[ti],transparent_bg=transparent_bg) for ti in range(len(images))])
    else:
        for bi,images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append(None)
    return (map_imgs,ori_imgs,ori_imgs_with_box,gen_imgs_list,gen_imgs_wb_list)
    


