from typing import Any, Dict
import numpy as np
from pyquaternion import Quaternion
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets import build_dataset
import numpy as np
import numba

__all__ = ["one_hot_encode","one_hot_decode"]

@numba.njit
def one_hot_encode(data: np.ndarray):
    data=data.transpose(1, 2, 0)
    assert data.ndim == 3
    n=data.shape[2]
    assert n<= 30  # ensure int32 does not overflow
    for x in np.unique(data):
        assert x in [0,1]
    # shift = np.arange(n, np.int32)[None, None]
    shift=np.zeros((1,1,n),np.int32)
    shift[0,0,:]=np.arange(0,n,1,np.int32)
    binary=(data>0)  # bool
    # after shift, numpy keeps int32, numba change dtype to int64
    binary=(binary<<shift).sum(-1)  # move bit to left and combine to one
    binary=binary.astype(np.int32)
    return (binary)

@numba.njit
def one_hot_decode(data:np.ndarray,n:int):
    shift=np.zeros((1,1,n),np.int32)
    shift[0,0,:]=np.arange(0,n,1,np.int32)
    x=np.zeros((*data.shape,1),data.dtype)
    x[...,0]=data
    x=(x>>shift)&1  # only keep the lowest bit, for each n
    x=x.transpose(2,0,1)
    return (x)

@DATASETS.register_module()
class NuScenesDatasetM(NuScenesDataset):
    def __init__(self,ann_file,pipeline=None,dataset_root=None,object_classes=None,map_classes=None,load_interval=1,with_velocity=True,modality=None,box_type_3d="LiDAR",filter_empty_gt=True,test_mode=False,eval_version="detection_cvpr_2019",use_valid_flag=False,force_all_boxes=False):
        # self.dataset_root="/data/sunny/MagicDrive/data/nuscenes"
        # print(self.dataroot)
        
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(object_classes)
        
        self.force_all_boxes=force_all_boxes
        super().__init__(ann_file=ann_file,pipeline=pipeline,dataset_root=dataset_root,object_classes=object_classes,map_classes=map_classes,load_interval=load_interval,with_velocity=with_velocity,modality=modality,box_type_3d=box_type_3d,filter_empty_gt=filter_empty_gt,test_mode=test_mode,eval_version=eval_version,use_valid_flag=use_valid_flag)
        # print(self.dataset_root)
        # print(ann_file)
    
    def get_cat_ids(self,idx):
        info=self.data_infos[idx]
        if self.use_valid_flag and not self.force_all_boxes:
            mask=info["valid_flag"]
            gt_names=set(info["gt_names"][mask])
        else:
            gt_names=set(info["gt_names"])
        cat_ids=[]
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return (cat_ids)
    
    def get_data_info(self,index):
        info=self.data_infos[index]
        
        data=dict(token=info["token"],sample_idx=info["token"],lidar_path=info["lidar_path"],sweeps=info["sweeps"],timestamp=info["timestamp"],location=info["location"])
        add_key=["description","timeofday","visibility","flip_gt"]
        
        for key in add_key:
            if key in info:
                data[key]=info[key]
        
        #ego2global
        ego2global=np.eye(4).astype(np.float32)
        ego2global[:3,:3]=Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3,3]=info["ego2global_translation"]
        data["ego2global"]=ego2global
        
        #lidar2ego
        lidar2ego=np.eye(4).astype(np.float32)
        lidar2ego[:3,:3]=Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3,3]=info["lidar2ego_translation"]
        data["lidar2ego"]=lidar2ego
        
        if self.modality["use_camera"]:
            data["image_paths"]=[]
            data["lidar2camera"]=[]
            data["lidar2image"]=[]
            data["camera2ego"]=[]
            data["camera_intrinsics"]=[]
            data["camera2lidar"]=[]
            
            for _,camera_info in info["cams"].items():
                # print("$$$$$$$$$$$$$$$$$$")
                old_path=camera_info["data_path"]
                # new_path=old_path
                old_path=old_path.split("/")
                # print(old_path)
                old_path=old_path[2:]
                new_path="/data/sunny/MagicDrive/data/"
                for idx,items in enumerate(old_path):
                    if idx!=len(old_path)-1:
                        new_path+=items+"/"
                    else:
                        new_path+=items
                
                camera_info["data_path"]=new_path
                # print("new path",new_path)
                # exit()

                data["image_paths"].append(camera_info["data_path"])
                
                #lidat2camera
                lidar2camera_r=np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t=(camera_info["sensor2lidar_translation"]@lidar2camera_r.T)
                lidar2camera_rt=np.eye(4).astype(np.float32)
                lidar2camera_rt[:3,:3]=lidar2camera_r.T
                lidar2camera_rt[3,:3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)
                
                # camera intrinsics
                camera_intrinsics=np.eye(4).astype(np.float32)
                camera_intrinsics[:3,:3]=camera_info["camera_intrinsics"]
                data["camera_intrinsics"].append(camera_intrinsics)
                
                # lidar2image
                lidar2image=camera_intrinsics@lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)
                
                # camera2ego
                camera2ego=np.eye(4).astype(np.float32)
                camera2ego[:3,:3]=Quaternion(camera_info["sensor2ego_rotation"]).rotation_matrix
                camera2ego[:3,3]=camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)
                
                #camaers2lidar
                camera2lidar=np.eye(4).astype(np.float32)
                camera2lidar[:3,:3]=camera_info["sensor2lidar_rotation"]
                camera2lidar[:3,3]=camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar) 
                
        annos,mask=self.get_ann_info(index)
        if "visibility" in data:
            data["visibility"]=data["visibility"][mask]
        data["ann_info"]=annos
        return (data)        

    def get_ann_info(self,index):
        # Get annotation info according to the given index
        info=self.data_infos[index]
        #filter out bbox containing no points
        if self.force_all_boxes:
            mask=np.ones_like(info["valid_flag"])
        elif self.use_valid_flag:
            mask=info["valid_flag"]
        else:
            mask=info["num_lidar_points"] > 0
        gt_bboxes_3d=info["gt_boxes"][mask]
        gt_names_3d= info["gt_names"][mask]
        gt_labels_3d=[]
        
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d=np.array(gt_labels_3d)
        if self.with_velocity:
            gt_velocity=info["gt_velocity"][mask]
            nan_mask=np.isnan(gt_velocity[:,0])
            gt_velocity[nan_mask]=[0.0,0.0]
            gt_bboxes_3d=np.concatenate([gt_bboxes_3d,gt_velocity],axis=-1)
        
        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d=LiDARInstance3DBoxes(gt_bboxes_3d,box_dim=gt_bboxes_3d.shape[-1],origin=(0.5,0.5,0)).convert_to(self.box_mode_3d)
        anns_results=dict(gt_bboxes_3d=gt_bboxes_3d,gt_labels_3d=gt_labels_3d,gt_names=gt_names_3d)
        return (anns_results,mask)

