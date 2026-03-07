import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
sys.path.insert(0, osp.join('..', 'main', 'transformer_utils'))
from config import cfg
import cv2
from tqdm import tqdm
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC = PROJECT_ROOT / 'src'
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from avatar_gs.segmentation import estimate_person_mask_modnet
from utils.inference_utils import non_max_suppression


def _save_obj(mesh, faces, file_name):
    with open(file_name, 'w', encoding='utf-8') as obj_file:
        for v in mesh:
            obj_file.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for f in faces:
            obj_file.write(
                f'f {f[0] + 1}/{f[0] + 1} {f[1] + 1}/{f[1] + 1} {f[2] + 1}/{f[2] + 1}\n'
            )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--pretrained_model', type=str, default=0)
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', type=str, default='na')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--demo_dataset', type=str, default='na')
    parser.add_argument('--demo_scene', type=str, default='all')
    parser.add_argument('--show_verts', action="store_true")
    parser.add_argument('--show_bbox', action="store_true")
    parser.add_argument('--save_mesh', action="store_true")
    parser.add_argument('--multi_person', action="store_true")
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--bbox_thr', type=int, default=50)
    parser.add_argument('--person_threshold', type=float, default=0.5)
    args = parser.parse_args()
    return args


def _component_boxes_from_mask(mask_u8, multi_person=True):
    mask = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    if mask.size == 0 or int(mask.sum()) <= 0:
        return []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if int(area) <= 0:
            continue
        score = float(area) / float(mask.shape[0] * mask.shape[1])
        boxes.append([float(x), float(y), float(x + w), float(y + h), score])
    boxes = sorted(boxes, key=lambda b: (b[4], (b[2] - b[0]) * (b[3] - b[1])), reverse=True)
    if not multi_person and len(boxes) > 1:
        return boxes[:1]
    return boxes


def _detect_person_boxes_modnet(img_path, multi_person=True, threshold=0.5):
    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image_bgr is None:
        return []
    mask = estimate_person_mask_modnet(
        image_bgr,
        ref_size=512,
        threshold=float(threshold),
    )
    return _component_boxes_from_mask(mask, multi_person=multi_person)

def main():

    args = parse_args()
    config_stem = str(args.pretrained_model)
    config_path = osp.join('./config', f'config_{config_stem}.py')
    if not osp.exists(config_path) and config_stem.endswith('_correct'):
        config_stem = config_stem[:-len('_correct')]
        config_path = osp.join('./config', f'config_{config_stem}.py')
    ckpt_path = osp.join('../pretrained_models', f'{args.pretrained_model}.pth.tar')

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(args.testset, args.agora_benchmark, shapy_eval_split=None, 
                            pretrained_model_path=ckpt_path, use_cache=False)
    cfg.update_config(args.num_gpus, args.exp_name)
    cudnn.benchmark = True

    # load model
    from base import Demoer
    from utils.preprocessing import load_img, process_bbox, generate_patch_image
    from utils.human_models import smpl_x
    try:
        from utils.vis import render_mesh
    except Exception:
        render_mesh = None
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()
    
    start = int(args.start)
    end = start + int(args.end)
    multi_person = args.multi_person
            

    for frame in tqdm(range(start, end)):
        img_path = os.path.join(args.img_path, f'{int(frame):06d}.jpg')

        # prepare input image
        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        os.makedirs(args.output_folder, exist_ok=True)

        ## person detection via MODNet
        det_boxes = _detect_person_boxes_modnet(
            img_path,
            multi_person=True,
            threshold=float(args.person_threshold),
        )
        
        # save original image if no bbox
        if len(det_boxes) < 1:
            # save rendered image
            frame_name = img_path.split('/')[-1]
            save_path_img = os.path.join(args.output_folder, 'img')
            os.makedirs(save_path_img, exist_ok= True)
            cv2.imwrite(os.path.join(save_path_img, f'{frame_name}'), vis_img[:, :, ::-1])
            continue
        
        if not multi_person:
            # only select the largest bbox
            num_bbox = 1
            det_boxes = det_boxes[:1]
        else:
            # keep bbox by NMS with iou_thr
            det_boxes = non_max_suppression(det_boxes, args.iou_thr)
            num_bbox = len(det_boxes)
        
        ## loop all detected bboxes
        for bbox_id in range(num_bbox):
            mmdet_box_xywh = np.zeros((4))
            mmdet_box_xywh[0] = det_boxes[bbox_id][0]
            mmdet_box_xywh[1] = det_boxes[bbox_id][1]
            mmdet_box_xywh[2] =  abs(det_boxes[bbox_id][2]-det_boxes[bbox_id][0])
            mmdet_box_xywh[3] =  abs(det_boxes[bbox_id][3]-det_boxes[bbox_id][1]) 

            # skip small bboxes by bbox_thr in pixel
            if mmdet_box_xywh[2] < args.bbox_thr or mmdet_box_xywh[3] < args.bbox_thr * 3:
                continue

            # for bbox visualization 
            start_point = (int(det_boxes[bbox_id][0]), int(det_boxes[bbox_id][1]))
            end_point = (int(det_boxes[bbox_id][2]), int(det_boxes[bbox_id][3]))   

            bbox = process_bbox(mmdet_box_xywh, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            ## save mesh
            if args.save_mesh:
                save_path_mesh = os.path.join(args.output_folder, 'mesh')
                os.makedirs(save_path_mesh, exist_ok= True)
                _save_obj(mesh, smpl_x.face, os.path.join(save_path_mesh, f'{frame:05}_{bbox_id}.obj'))

            ## save single person param
            smplx_pred = {}
            smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy()
            smplx_pred['leye_pose'] = np.zeros((1, 3))
            smplx_pred['reye_pose'] = np.zeros((1, 3))
            smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10).cpu().numpy()
            smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10).cpu().numpy()
            smplx_pred['transl'] =  out['cam_trans'].reshape(-1,3).cpu().numpy()
            save_path_smplx = os.path.join(args.output_folder, 'smplx')
            os.makedirs(save_path_smplx, exist_ok= True)

            npz_path = os.path.join(save_path_smplx, f'{frame:05}_{bbox_id}.npz')
            np.savez(npz_path, **smplx_pred)

            ## render single person mesh
            focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
            princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
            if render_mesh is not None:
                vis_img = render_mesh(
                    vis_img,
                    mesh,
                    smpl_x.face,
                    {'focal': focal, 'princpt': princpt},
                    mesh_as_vertices=args.show_verts,
                )
            if args.show_bbox:
                vis_img = cv2.rectangle(vis_img, start_point, end_point, (255, 0, 0), 2)

            ## save single person meta
            meta = {'focal': focal, 
                    'princpt': princpt, 
                    'bbox': bbox.tolist(), 
                    'bbox_detector': mmdet_box_xywh.tolist(),
                    'bbox_mmdet': mmdet_box_xywh.tolist(),
                    'detector': 'modnet',
                    'bbox_id': bbox_id,
                    'img_path': img_path}
            json_object = json.dumps(meta, indent=4)

            save_path_meta = os.path.join(args.output_folder, 'meta')
            os.makedirs(save_path_meta, exist_ok= True)
            with open(os.path.join(save_path_meta, f'{frame:05}_{bbox_id}.json'), "w") as outfile:
                outfile.write(json_object)

        ## save rendered image with all person
        frame_name = img_path.split('/')[-1]
        save_path_img = os.path.join(args.output_folder, 'img')
        os.makedirs(save_path_img, exist_ok= True)
        cv2.imwrite(os.path.join(save_path_img, f'{frame_name}'), vis_img[:, :, ::-1])


if __name__ == "__main__":
    main()
