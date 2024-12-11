from typing import Tuple, List
from functools import partial
import random
import cv2
import torch
import numpy as np
from transformers import CLIPTokenizer
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from .box_utils import trans_boxes_to_views


META_KEY_LIST = [
    "gt_bboxes_3d",
    "gt_labels_3d",
    "camera_intrinsics",
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
    "metas",
]


def _tokenize_captions(examples, template, tokenizer=None, is_train=True):
    captions = []
    # print(" ")
    for example in examples:
        # print("example:",example["metas"])
        caption = template.format(**example["metas"].data)
        # print("caption",caption)
        captions.append(caption)
    
    # print (len(caption))
        
    captions.append("")
    if tokenizer is None:
        return None, captions

    # pad in the collate_fn function
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="do_not_pad",
        truncation=True,
    )
    
    #this will have the tokens for the start token , end token and the tokens for all the words in the caption . this has 2 lists ()  
    # print("inputs tokens:",inputs,len(inputs))

    # exit()
    
    input_ids = inputs.input_ids
    # pad to the longest of current batch (might differ between cards)
    padded_tokens = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids
    
    # print("padded tokens:",padded_tokens)
    # exit()
    
    # print("length of captions:",len(captions[0]))
    # print("length of captions:",len(captions[1]))
    # print("captions 0:",captions[0])
    # p=captions[0].split(" ")
    # print(len(p))
    # # print("captions 0:",captions[0][0])
    # # print("captions 0:",captions[0][1])
    # # print("captions 0:",captions[0][2]) 
    # print("captions 1:",captions[1])
    # print("length of tokens:",len(padded_tokens[0]))
    # print("length of tokens:",len(padded_tokens[1]))
    # print("padded tokens 0:",padded_tokens[0])
    # print("padded tokens 1:",padded_tokens[1])
    # exit()
    
    # print("padded  tokens:",padded_tokens)
    # print("captions:", captions)
    # exit()
    
    return padded_tokens, captions


def ensure_canvas(coords, canvas_size: Tuple[int, int]):
    """Box with any point in range of canvas should be kept.

    Args:
        coords (_type_): _description_
        canvas_size (Tuple[int, int]): _description_

    Returns:
        np.array: mask on first axis.
    """
    (h, w) = canvas_size
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    w_mask = np.any(np.logical_and(
        coords[..., 0] > 0, coords[..., 0] < w), axis=1)
    h_mask = np.any(np.logical_and(
        coords[..., 1] > 0, coords[..., 1] < h), axis=1)
    c_mask = np.logical_and(c_mask, np.logical_and(w_mask, h_mask))
    return c_mask


def ensure_positive_z(coords):
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    return c_mask


def random_0_to_1(mask: np.array, num):
    assert mask.ndim == 1
    inds = np.where(mask == 0)[0].tolist()
    random.shuffle(inds)
    mask = np.copy(mask)
    mask[inds[:num]] = 1
    return mask


def _transform_all(examples, matrix_key, proj):
    """project all bbox to views, return 2d coordinates.

    Args:
        examples (List): collate_fn input.

    Returns:
        2-d list: List[List[np.array]] for B, N_cam. Can be [].
    """
    # print("matrix key:",matrix_key)
    # print(proj)
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    # print("Number of boxes in batch:",len(gt_bboxes_3d))
    # print("Number of boxes in batch:",len(gt_bboxes_3d[0]))
    # lidar2image (np.array): lidar to image view transformation
    trans_matrix = np.stack([example[matrix_key].data.numpy()  # (6,4,4)
                            for example in examples], axis=0)
    # print("###########################",trans_matrix.shape)#(6,4,4)
    # exit()
    # img_aug_matrix (np.array): augmentation matrix
    img_aug_matrix = np.stack([example['img_aug_matrix'].data.numpy()
                               for example in examples], axis=0)
    # print("###########################",trans_matrix.shape) #(6,4,4)
    
    B, N_cam = trans_matrix.shape[:2] #(1,6)
    bboxes_coord = []
    # for each keyframe set
    for idx in range(B):
        # if zero, add empty list
        if len(gt_bboxes_3d[idx]) == 0:
            # keep N_cam dim for convenient
            bboxes_coord.append([None for _ in range(N_cam)])
            continue
        # print("pass to trans boxes to view:",trans_matrix[idx].shape)
        # print("pass to trans boxes to view:",img_aug_matrix[idx].shape)
        coords_list = trans_boxes_to_views(
            gt_bboxes_3d[idx], trans_matrix[idx], img_aug_matrix[idx], proj)
        # print(len(coords_list)) #(6)
        # print("bbox corners in camera coordinate system:",coords_list[0].shape)
        # print("bbox corners in camera coordinate system:",coords_list[0])
        # print("bbox corners in camera coordinate system:",coords_list[1].shape)
        # print("bbox corners in camera coordinate system:",coords_list[1])
        # print("bbox corners in camera coordinate system:",coords_list[2].shape)
        # print("bbox corners in camera coordinate system:",coords_list[2])
        # print("bbox corners in camera coordinate system:",coords_list[3].shape)
        # print("bbox corners in camera coordinate system:",coords_list[3])
        # print("bbox corners in camera coordinate system:",coords_list[4].shape)
        # print("bbox corners in camera coordinate system:",coords_list[4])
        # print("bbox corners in camera coordinate system:",coords_list[5].shape)
        # print("bbox corners in camera coordinate system:",coords_list[5])
        bboxes_coord.append(coords_list)
    # print("PROJECTED BOXES")
    # print(len(bboxes_coord)) #(1)
    # print(len(bboxes_coord[0]))#(6)
    # print(len(bboxes_coord[0][0]))#(n_boxes in sample)
    # exit()
    
    return bboxes_coord


def _preprocess_bbox(bbox_mode, canvas_size, examples, is_train=True,
                     view_shared=False, use_3d_filter=True, bbox_add_ratio=0,
                     bbox_add_num=0, bbox_drop_ratio=0):
    """Pre-processing for bbox
    .. code-block:: none

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Args:
        bbox_mode (str): type of bbox raw data.
            cxyz -> x1y1z1, x1y0z1, x1y1z0, x0y1z1;
            all-xyz -> all 8 corners xyz;
            owhr -> center, l, w, h, z-orientation.
        canvas_size (2-tuple): H, W of input images
        examples: collate_fn input
        view_shared: if enabled, all views share same set of bbox and output
            N_cam=1; otherwise, use projection to keep only visible bboxes.
    Return:
        in form of dict:
            bboxes (Tensor): B, N_cam, max_len, ...
            classes (LongTensor): B, N_cam, max_len
            masks: 1 for data, 0 for padding
    """
    # init data
    bboxes = []
    classes = []
    max_len = 0
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    gt_labels_3d: List[torch.Tensor] = [
        example["gt_labels_3d"].data for example in examples]

    # params
    B = len(gt_bboxes_3d)
    N_cam = len(examples[0]['lidar2image'].data.numpy())
    N_out = 1 if view_shared else N_cam
    
    # print("batch:",B)
    # print("n_cam:",N_cam)
    # print("n_out",N_out)
    # print("number of GT boxes insied collate",len(gt_bboxes_3d[0]))
    # print("number of GT boxes insied collate",len(gt_labels_3d[0]))

    bboxes_coord = None
    if not view_shared and not use_3d_filter:
        bboxes_coord = _transform_all(examples, 'lidar2image', True)
    elif not view_shared:
        # for normal conditions this is getting executed
        bboxes_coord_3d = _transform_all(examples, 'lidar2camera', False)
        # this is a list of n_views . Inside each list there are n_boxes . basically each list contains the boxes transformed in it coordinate system. 
    # print("coord")
    # print("coords:",len(bboxes_coord_3d))
    # for idx in bboxes_coord_3d:
    #     print("coords inside list:",idx[0].shape)
    
    #at this point all the boxes are in the cameras
    
    # for each keyframe set
    for idx in range(B):
        bboxes_kf = gt_bboxes_3d[idx]
        classes_kf = gt_labels_3d[idx]
        
        # print(len(bboxes_kf))  (n_box)
        # print(len(classes_kf)) (n_box)
        # exit()

        # if zero, add zero length tensor (for padding).
        if len(bboxes_kf) == 0 or (
                random.random() < bbox_drop_ratio and is_train):
            bboxes.append([None] * N_out)
            classes.append([None] * N_out)
            continue

        # whether share the boxes across views, filtered by 2d projection.
        if not view_shared:
            index_list = []  # each view has a mask
            if use_3d_filter:
                coords_list = bboxes_coord_3d[idx]
                filter_func = ensure_positive_z
            else:
                # filter bbox according to 2d projection on image canvas
                coords_list = bboxes_coord[idx]
                # judge coord by cancas_size
                filter_func = partial(ensure_canvas, canvas_size=canvas_size)
            # we do not need to handle None since we already filter for len=0
            
            # print("filter func:",filter_func) #ensure_postive_z
            # print("coords list:",coords_list[0].shape) #(n_box,8,3)
            # print("coords list:",len(coords_list)) #(6)
            # print("coords list:",len(coords_list[0])) #(n_box)
            
            # coords list if a list of view-wise boxes. it has 6 lists each of which has n_boxes in the camera coordinate system. 
            # exit()
            
            for coords in coords_list:
                c_mask = filter_func(coords)
                # print("c mask",c_mask)
                if random.random() < bbox_add_ratio and is_train: #always true during training.
                    c_mask = random_0_to_1(c_mask, bbox_add_num)
                    # print("coords",len(coords))
                    # print("c mask:",c_mask)
                    # exit()
                index_list.append(c_mask) 
                #basically take all boxes where all the corners of the box have z>0 in front of camera 
                max_len = max(max_len, c_mask.sum())

            # print("index list",len(index_list))#(6)
            # print("index list",len(index_list[0]))#(n_box)
            # print("max len:",max_len)
            # exit()
                
        else:
            # we use as mask, torch.bool is important
            index_list = [torch.ones(len(bboxes_kf), dtype=torch.bool)]
            max_len = max(max_len, len(bboxes_kf))
        
        # index_list.append(c_mask) 
        # #basically take all boxes where all the corners of the box have z>0 in front of camera 
        # max_len = max(max_len, c_mask.sum())
        # print("number of positive boxes:",len(index_list))
        
        # here index list is a list of booleans of which boxes to accept ore reject
        # max len is is the length of all good boxes 
        # print("index list",len(index_list))#(6)
        # print("index list",len(index_list[0]))#(n_box)
        # print("max len:",max_len)
        # print("bbox mode:",bbox_mode)
        # exit()
        # exit()

        # construct data
        if bbox_mode == 'cxyz':
            # x1y1z1, x1y0z1, x1y1z0, x0y1z1
            bboxes_pt = bboxes_kf.corners[:, [6, 5, 7, 2]]
        elif bbox_mode == 'all-xyz':
            bboxes_pt = bboxes_kf.corners  # n x 8 x 3
        elif bbox_mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {bbox_mode}")
        
        # print("bboxes kf:",len(bboxes_kf))
        # print("bboxes pt:",bboxes_pt.shape)
        
        # for ind in index_list:
        #     print("ind len",len(ind))
        #     print("ind true:",sum(ind==True))
        #     print("ind false:",sum(ind==False))
        #     print("ind:",ind)
        
        bboxes.append([bboxes_pt[ind] for ind in index_list])
        #right now the list has n_view elements (inside every view there is a list of boxes in that view which have all corners z>0)
        classes.append([classes_kf[ind] for ind in index_list])
        bbox_shape = bboxes_pt.shape[1:]

        # print("list inside list",len(bboxes))
        # print("number of views list",len(bboxes[0]))
        # print("boxes inside first view",len(bboxes[0][0]))
        # print("list inside list",len(classes))
        # print("number of views list",len(classes[0]))
        # print("boxes inside first view",len(classes[0][0]))
        # print(bbox_shape)
        # exit()
        
        # print("this list is of corners of all positive z >0 boxes in camera coord system:",len(bboxes[0]))
        # print(bbox_shape)
        # exit()
    
    # for b in range(len(bboxes)):
    #     print(b)
        
    # there is no (visible) boxes in this batch
    if max_len == 0:
        return None, None

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape)
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    #make all the 
    
    for _b in range(B):
        _bboxes = bboxes[_b]
        # print("number of positive visible boxes:",len(_bboxes))
        # exit()
        _classes = classes[_b]
        for _n in range(N_out):
            if _bboxes[_n] is None:
                continue  # empty for this batch
            this_box_num = len(_bboxes[_n])
            ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
            ret_classes[_b, _n, :this_box_num] = _classes[_n]
            ret_masks[_b, _n, :this_box_num] = True

    # assemble as input format
    
    # these are tensors of shape (ns,N_view,max_len,8,3) with all positive bbox corners.
    # but all views will not have equal number of positive boxes so others are padded with 0 both for boxes,classes and masks 
    
    
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    # exit()
    return ret_dict, bboxes


def collate_fn(
    examples: Tuple[dict, ...],
    template: str,
    tokenizer: CLIPTokenizer = None,
    is_train: bool = True,
    bbox_mode: str = None,
    bbox_view_shared: bool = False,
    bbox_drop_ratio: float = 0,
    bbox_add_ratio: float = 0,
    bbox_add_num: int = 3,
):
    """
    We need to handle:
    1. make multi-view images (img) into tensor -> [N, 6, 3, H, W]
    2. make masks (gt_masks_bev, gt_aux_bev) into tensor
        -> [N, 25 = 8 map + 10 obj + 7 aux, 200, 200]
    3. make caption (location, desctiption, timeofday) and tokenize, padding
        -> [N, pad_length]
    4. extract camera parameters (camera_intrinsics, camera2lidar)
        camera2lidar: A @ v_camera = v_lidar
        -> [N, 6, 3, 7]
    We keep other meta data as original.
    """
    # print("examples:",examples)
    # print("gt_aux_bev",examples[0]["gt_aux_bev"].shape)
    # print("gt_masks_bev",examples[0]["gt_masks_bev"].shape)
    # exit()
    
    # print("bbox add ratio:",bbox_add_ratio)
    # print("is_train:",is_train)
    
    if bbox_add_ratio > 0 and is_train:
        assert bbox_view_shared == False, "You cannot add any box on view shared."
    # print(examples[0].keys())
    # for keys,vals in examples[0].items():
    #     print(keys)
    # exit()
    # print("inside collate fn:",examples[0].keys())
    # mask
    if "gt_aux_bev" in examples[0] and examples[0]["gt_aux_bev"] is not None:
        keys = ["gt_masks_bev", "gt_aux_bev"]
        assert bbox_drop_ratio == 0, "map is not affected in bbox_drop"
    else:
        keys = ["gt_masks_bev"]
    # print("keys to add 1 :",keys)
    
    # # fmt: off
    # print(examples[0]["gt_masks_bev"].shape)
    # bev_map_with_aux = torch.stack([torch.from_numpy(np.concatenate([ example[key] for key in keys], axis=0)).float() for example in examples], dim=0)  # float32
    # print("after concatenting the separate channel:",bev_map_with_aux.shape)
    
    # exit()
    
    
    # bev map with aux takes the bev segmentation map and adds a batch dimension
    bev_map_with_aux = torch.stack([torch.from_numpy(np.concatenate([
        example[key] for key in keys  # np array, channel-last
    ], axis=0)).float() for example in examples], dim=0)  # float32
    # fmt: on
    # print("after concatenting the separate channel:",bev_map_with_aux.shape)
    # exit()

    # camera param
    # TODO: camera2lidar should be changed to lidar2camera
    # fmt: off
    
    
    # print("cam intrinsics:",examples[0]["camera_intrinsics"].data.shape)
    # print("cam 2 lidar:",examples[0]["camera2lidar"].data.shape)
    
    
    # the take the 3x3 intrinsic matrix and concate it with the 3x4 transform matrix . 
    camera_param = torch.stack([torch.cat([
        example["camera_intrinsics"].data[:, :3, :3],  # 3x3 is enough
        example["camera2lidar"].data[:, :3],  # only first 3 rows meaningful
    ], dim=-1) for example in examples], dim=0)
    # fmt: on
    
    # print("camera paramters after collate:",camera_param.shape)
    ret_dict = {
        "bev_map_with_aux": bev_map_with_aux,
        "camera_param": camera_param,
        "kwargs": {},
    }
    
    # print(examples[0]["img"].data.shape)
    # exit()

    #change the memory layout , nothing else in collate for images
    if "img" in examples[0]:
        # multi-view images
        pixel_values = torch.stack(
            [example["img"].data for example in examples])
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()
        ret_dict["pixel_values"] = pixel_values
    elif is_train:
        raise RuntimeError("For training, you should provide gt images.")
    # print(pixel_values.shape)
    # exit()
    
    # bboxes_3d, convert to tensor
    # here we consider:
    # 1. do we need to filter bboxes for each view? use `view_shared`
    # 2. padding for one batch of data if need (with zero), and output mask.
    # 3. what is the expected output format? dict of kwargs to bbox embedder
    
    canvas_size = pixel_values.shape[-2:]#(224,400)
    # print("input to preprocess_box:")
    # print("bbox mode:",bbox_mode)
    # print("canvas size",canvas_size)
    # # print(exam)
    # print("view shared:",bbox_view_shared)
    # print("add ratio",bbox_add_ratio)
    # print("add num",bbox_add_num)
    # print("drop ratio",bbox_drop_ratio)
    
    if bbox_mode is not None:
        # NOTE: both can be None
        bboxes_3d_input, bbox_view_coord = _preprocess_bbox(
            bbox_mode, canvas_size, examples, is_train=is_train,
            view_shared=bbox_view_shared, bbox_add_ratio=bbox_add_ratio,
            bbox_add_num=bbox_add_num, bbox_drop_ratio=bbox_drop_ratio)
        
        # bbox_mode, canvas_size, examples, is_train=True,
        #              view_shared=False, use_3d_filter=True, bbox_add_ratio=0,
        #              bbox_add_num=0, bbox_drop_ratio=0
        
        
        ret_dict["kwargs"]["bboxes_3d_data"] = bboxes_3d_input
        #so the dict contains positive boxes of max_len in all views (bs,n_view,max_len,8,3) but only positive(z>0) ones are filled rest are 0.
        # print("all bboxes in n_views:",bboxes_3d_input["bboxes"][0,0,:,:,:])
        # print("list of positive boxes:",bbox_view_coord[0][0])
    else:
        bbox_view_coord = None
    
    # print("template:",template)
    # print("tokenizer:",tokenizer)
    

    # captions: one real caption with one null caption
    
    input_ids_padded, captions = _tokenize_captions(
        examples, template, tokenizer, is_train)
    ret_dict["captions"] = captions[:-1]  # list of str
    
    # print("captions:",captions[:-1])
    
    
    if tokenizer is not None:
        # real captions in head; the last one is null caption
        # we omit "attention_mask": padded_tokens.attention_mask, seems useless
        ret_dict["input_ids"] = input_ids_padded[:-1]
        ret_dict["uncond_ids"] = input_ids_padded[-1:]

    # print("input ids padded:",input_ids_padded)
    # print(input_ids_padded[:-1])
    # print(input_ids_padded[-1:])
    
    # this returns ret["input_ids"] which is the tokens of the actual caption, then there is the ret["uncond ids"] which is of equal length as ret["input_ids"] but filled with the end caption token.
    

    # exit()

    # other meta data
    meta_list_dict = dict()
    for key in META_KEY_LIST:
        try:
            meta_list = [example[key] for example in examples]
            meta_list_dict[key] = meta_list
        except KeyError:
            continue
    ret_dict['meta_data'] = meta_list_dict
    
    # print(ret_dict.keys())
    # print(ret_dict["kwargs"])
    # print("$$"*100)
    
    # print(ret_dict["meta_data"])
    
    # for keys,vals in ret_dict.items():
    #     print(keys,vals)

    # exit()

    return ret_dict
