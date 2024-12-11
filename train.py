# print("ok")
import os
import cv2
import sys
import logging
from pprint import pprint
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torchvision
from mmdet3d.datasets import build_dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore",category=ShapelyDeprecationWarning)
# print("sasaijdisajdnisnadsidakdniasndiajsdnasjdnajdn")
#local imports
sys.path.append("/data/srinjay/magic_drive/")
import dataset.pipeline
from runner.utils import load_module
# from utils import *
print("imports done")

@hydra.main(version_base=None,config_path="/data/srinjay/magic_drive/configs",config_name="config")
def main(cfg: DictConfig):
    # pprint(cfg)
    if cfg.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
        
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler) or cfg.try_run:
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
        
    # multi process context
    # since our model has randomness to train the uncond embedding, we need this.
    ddp_kwargs=DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator=Accelerator(gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,mixed_precision=cfg.accelerator.mixed_precision,log_with=cfg.accelerator.report_to,project_dir=cfg.log_root,kwargs_handlers=[ddp_kwargs])
    
    # pprint(cfg.dataset.data.train)    
    train_dataset=build_dataset(OmegaConf.to_container(cfg.dataset.data.train,resolve=True))
    # print(train_dataset[0]["bev_map_with_aux"].shape)
    # exit()
    # pprint(OmegaConf.to_container(cfg.dataset.data.train,resolve=True))
    # pprint(train_dataset[0].keys())
    # exit()
    # pprint("$"*50)
    # sample=train_dataset[0]
    # pprint(sample.keys())
    # print("sdnsaidnj")
    # print(sample["metas"])
    # # for key,_ in sample.items():
    # #     print(key,sample[key])c
    # print("iandiasndaind")
    val_dataset=build_dataset(OmegaConf.to_container(cfg.dataset.data.val,resolve=True))
    # pprint(train_dataset[0].keys())
    # pprint(len(train_dataset))
    # pprint(val_dataset[0].keys())
    # pprint(len(val_dataset))
    # pprint(train_dataset[0]["img_aug_matrix"])
    # exit()
    # print(cfg.model.runner_module)
    # print(val_dataset[0]["metas"])
    # print(val_dataset[0].keys())
    # print(val_dataset[0]["gt_masks_bev"].shape)
    # cv2.imwrite("1.png",val_dataset[0]["gt_masks_bev"][:3,:,:])
    # exit()
    if cfg.resume_from_checkpoint and cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint=cfg.resume_from_checkpoint[:-1]
    runner_cls=load_module(cfg.model.runner_module)
    # print(runner_cls)
    runner=runner_cls(cfg,accelerator,train_dataset,val_dataset)
    # print(runner)
    
    runner.set_optimizer_schedule()
    runner.prepare_device()
    # tracker
    logging.debug("Current config:\n"+OmegaConf.to_yaml(cfg,resolve=True))
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(f"tb-{cfg.task_id}",config=None)
    logging.debug("start!")
    runner.run()
    
    # for batch_idx,batch in enumerate(train_dataset):
    #     if batch_idx<=200:
    #         print(batch.keys())
    #         # dict_keys(['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks_bev', 'gt_aux_bev', 'camera_intrinsics', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'metas'])
    #         print(type(batch))
    #         mv_img=batch["img"].data
    #         img_aug_matrix=batch["img_aug_matrix"].data
    #         camera_K=batch["camera_intrinsics"].data
    #         liadr2ego=batch["lidar2ego"].data
    #         lidar2camera=batch["lidar2camera"].data
    #         camera2lidar=batch["camera2lidar"].data
    #         lidar2image=batch["lidar2image"].data
    #         img_aug_matrix=batch["img_aug_matrix"].data
    #         gt_bboxes_3d=batch["gt_bboxes_3d"].data
    #         gt_labels_3d=batch["gt_labels_3d"].data
                        
    #         print(camera_K.shape)
    #         print(mv_img.shape)
    #         print(img_aug_matrix.shape)
    #         print(liadr2ego.shape)
    #         print(lidar2camera.shape)
    #         print(camera2lidar.shape)
    #         print(lidar2image.shape)
    #         print(img_aug_matrix.shape)
    #         print(gt_bboxes_3d.tensor.shape)
    #         print(gt_labels_3d)
            
    #         mv_img+=1
    #         mv_img/=2
    #         # mv_img*=255
    #         print(torch.max(mv_img))
    #         print(torch.min(mv_img))
            
    #         rendered_img=torch.randn(mv_img.shape)
    #         print(rendered_img.shape)
    #         rendered_img[1,:,:,:]=mv_img[1,:,:,:]
    #         avg=torch.mean(mv_img[0],dim=1).unsqueeze(0)
    #         avg=torch.mean(avg,dim=2)
    #         avg=avg.unsqueeze(2).unsqueeze(3)
    #         avg=avg.repeat(1,1,224,400)
    #         # avg=avg.repeat(1,3,1,1)
    #         print(avg.shape)
    #         # print(torch.mean(mv_img[0],dim=0).shape)
    #         # exit()
            
    #         for idx in range(mv_img.shape[0]):
    #             if idx!=1:
    #                 avg=torch.mean(mv_img[idx],dim=1).unsqueeze(0)
    #                 avg=torch.mean(avg,dim=2)
    #                 avg=avg.unsqueeze(2).unsqueeze(3)
    #                 avg=avg.repeat(1,1,224,400)
    #                 rnd=torch.randn(avg.shape)
    #                 # avg=torch.mean(mv_img[idx],dim=0).unsqueeze(0)
    #                 # avg=avg.repeat(1,3,1,1)
    #                 rendered_img[idx,:,:,:]=0.95*(avg[0,:,:,:])+0.05*(rnd[0,:,:,:])
                
            
    #         for idx in range(mv_img.shape[0]):
    #             torchvision.utils.save_image(rendered_img,"/data/srinjay/magic_drive/rendered_images/"+str(batch_idx)+".png",make_grid=True)
            
    #         # exit()
            
    #         # for idx in range(mv_img.shape[0]):
    #         #     torchvision.utils.save_image(mv_img,"/data/srinjay/magic_drive/datalaodaer_images/"+str(batch_idx)+".png",make_grid=True)
            
            
    #         # print(batch.image_paths)
    #         # print(batch.sweeps)
    #         # print(batch["image_paths"])
    #     else:
    #         break

if __name__ == "__main__":
    main()