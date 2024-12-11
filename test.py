import os
import sys
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm
import torch
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

#local imports
sys.path.append("/data/srinjay/magic_drive/")
from misc.test_utils import prepare_all,run_one_batch
from runner.utils import concat_6_views

transparent_bg = True
target_map_size = 400

@hydra.main(version_base=None,config_path="/data/srinjay/magic_drive/configs",config_name="test_config")
def main(cfg: DictConfig):
    if cfg.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')
    
    output_dir=to_absolute_path(cfg.resume_from_checkpoint)
    original_overrides=OmegaConf.load(os.path.join(output_dir, "hydra/overrides.yaml"))
    current_overrides=HydraConfig.get().overrides.task
    
    config_name=HydraConfig.get().job.config_name
    overrides=original_overrides + current_overrides
    cfg=hydra.compose(config_name, overrides=overrides)
    logging.info(f"Your validation index: {cfg.runner.validation_index}")
    
    #### setup everything ####
    pipe,val_dataloader,weight_dtype=prepare_all(cfg)
    OmegaConf.save(config=cfg,f=os.path.join(cfg.log_root,"run_config.yaml"))
    total_num=0
    progress_bar=tqdm(range(len(val_dataloader)*cfg.runner.validation_times),desc="Steps")
    # print(progress_bar)
    print(len(val_dataloader))
    # exit()
    for val_idx,val_input in enumerate(val_dataloader):
        total_num=val_idx
        # print(val_input)
        return_tuples=run_one_batch(cfg,pipe,val_input,weight_dtype,transparent_bg=transparent_bg,map_size=target_map_size)
        print(os.path.join(cfg.log_root,f"{total_num}_map.png"))
        
        for map_img,ori_imgs,ori_imgs_wb,gen_imgs_list,gen_imgs_wb_list in zip(*return_tuples):
            # map_img.save(os.path.join(cfg.log_root, f"{total_num}_map.png"))
            # if ori_imgs is not None:
            #     ori_img=concat_6_views(ori_imgs)
            #     ori_img.save(os.path.join(cfg.log_root, f"{total_num}_ori.png"))
            # save gen
            for ti, gen_imgs in enumerate(gen_imgs_list):
                gen_img=concat_6_views(gen_imgs)
                gen_img.save(os.path.join(cfg.log_root,f"{total_num}_gen{ti}.png"))
            
            # if cfg.show_box:
            #     # save ori with box
            #     if ori_imgs_wb is not None:
            #         ori_img_with_box=concat_6_views(ori_imgs_wb)
            #         ori_img_with_box.save(os.path.join(cfg.log_root,f"{total_num}_ori_box.png"))
            #     # save gen with box
            #     for ti, gen_imgs_wb in enumerate(gen_imgs_wb_list):
            #         gen_img_with_box=concat_6_views(gen_imgs_wb)
            #         gen_img_with_box.save(os.path.join(cfg.log_root,f"{total_num}_gen{ti}_box.png"))
    
if __name__ == "__main__":
    main()

    
    