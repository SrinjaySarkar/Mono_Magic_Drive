import os
import numpy as np
import pickle
import json

root="/data/sunny/MagicDrive/data/nuscenes_mmdet3d_2/"
root1="/data/sunny/MagicDrive/data/nusc_mmdet3d_2/"
root2="/data/sunny/MagicDrive/data/nuscenes/nuscenes_dbinfos_train.pkl"
root3="/data/sunny/MagicDrive/data/nuscenes/maps/expansion/singapore-onenorth.json"

with open(os.path.join(root,'nuscenes_infos_train.pkl'),'rb') as f:
    data=pickle.load(f)
print(data.keys())

print(data["infos"].keys())

exit()

with open(os.path.join(root1,"nuscenes_infos_train.pkl"),"rb") as f:
    data1=pickle.load(f)
print(data1.keys()[0])

# with open(os.path.join(root,'nuscenes_dbinfos_train.pkl'),'rb') as f:
#     data3=pickle.load(f)
# print(data3.keys())


# print(len(data["infos"]))
# print(len(data1["infos"]))

# print(data["infos"][0])
# print(data1["infos"][0])

# print(data3["truck"])
# with open(root3,"r") as file:
#     data=json.load(file)
# print(data)

# print(subset_data["infos"][0].keys())
# print(subset_data["infos"][1].keys())
# print(subset_data["infos"][2].keys())
# print(subset_data["infos"][3].keys())
# print(subset_data["infos"][4].keys())
# print(subset_data["infos"][5].keys())



# @PIPELINES.register_module()
# class LoadBEVSegmentation:
#     def __init__(
#         self,
#         dataset_root: str,
#         xbound: Tuple[float, float, float],
#         ybound: Tuple[float, float, float],
#         classes: Tuple[str, ...],
#     ) -> None:
#         super().__init__()
#         patch_h = ybound[1] - ybound[0]
#         patch_w = xbound[1] - xbound[0]
#         canvas_h = int(patch_h / ybound[2])
#         canvas_w = int(patch_w / xbound[2])
#         self.patch_size = (patch_h, patch_w)
#         self.canvas_size = (canvas_h, canvas_w)
#         self.classes = classes

#         self.maps = {}
        
#         print("LOCATIONS:",LOCATIONS)
        
#         for location in LOCATIONS:
#             self.maps[location] = NuScenesMap(dataset_root, location)

#     def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         lidar2point = data["lidar_aug_matrix"]
#         point2lidar = np.linalg.inv(lidar2point)
#         lidar2ego = data["lidar2ego"]
#         ego2global = data["ego2global"]
#         lidar2global = ego2global @ lidar2ego @ point2lidar

#         map_pose = lidar2global[:2, 3]
#         patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

#         rotation = lidar2global[:3, :3]
#         v = np.dot(rotation, np.array([1, 0, 0]))
#         yaw = np.arctan2(v[1], v[0])
#         patch_angle = yaw / np.pi * 180

#         mappings = {}
#         for name in self.classes:
#             if name == "drivable_area*":
#                 mappings[name] = ["road_segment", "lane"]
#             elif name == "divider":
#                 mappings[name] = ["road_divider", "lane_divider"]
#             else:
#                 mappings[name] = [name]

#         layer_names = []
#         for name in mappings:
#             layer_names.extend(mappings[name])
#         layer_names = list(set(layer_names))

#         location = data["location"]
#         masks = self.maps[location].get_map_mask(
#             patch_box=patch_box,
#             patch_angle=patch_angle,
#             layer_names=layer_names,
#             canvas_size=self.canvas_size,
#         )
#         # masks = masks[:, ::-1, :].copy()
#         masks = masks.transpose(0, 2, 1)
#         masks = masks.astype(np.bool)

#         num_classes = len(self.classes)
#         labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
#         for k, name in enumerate(self.classes):
#             for layer_name in mappings[name]:
#                 index = layer_names.index(layer_name)
#                 labels[k, masks[index]] = 1

#         data["gt_masks_bev"] = labels
#         return data