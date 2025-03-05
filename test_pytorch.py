import argparse
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import cv2
import onnxruntime as ort

from videoanalyst.config.config import cfg, specify_task
from videoanalyst.model import builder as model_builder


def make_parser():
    parser = argparse.ArgumentParser(
        description="press s to select the target box,\n \
                        then press enter or space to confirm it or press c to cancel it,\n \
                        press c to stop track and press q to exit program")
    parser.add_argument(
        "-cfg",
        "--config",
        default="experiments/stmtrack/test/lasot/stmtrack-googlenet-lasot.yaml",
        type=str,
        help='experiment configuration')

    return parser


def get_model(args):  # reference to demo/main/video/sot_video.py
    root_cfg = cfg
    root_cfg.merge_from_file(args.config)

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    # build model
    model = model_builder.build(task, task_cfg.model)

    return model

class FeatureExtractionMemory(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, im_memory, mask_memory):  # (1,3,289,289), (1,1,289,289)
        im_memory = im_memory.float()
        mask_memory = mask_memory.float()

        fm = self.model.memorize(im_memory, mask_memory)
        return fm
    
class FeatureExtractionQuery(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, im_query):  # (1,3,289,289)
        im_query = im_query.float()
        fq = self.model.basemodel_q(im_query)
        fq = model.neck_q(fq)  
        return fq
    
class ReadMemoryAndHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.q_size = 289  # fixed 

    def forward(self, fm_init, fq):
        # fm = torch.cat([fm_init, fm_prev], dim=2)
        # fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = \
        #     self.model.head(fm, fq, self.q_size)
        fm = fm_init
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = \
            self.model.head(fm, fq, self.q_size)
        # apply sigmoid
        fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
        fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
        # apply centerness correction
        fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final

        return fcos_score_final, fcos_bbox_final
    

def convertToONNX_FeatureExtractionMemory(model):
    feature_extraction_memory = FeatureExtractionMemory(model)
    input_names = ["memory_image", "memory_mask"]
    output_names = ["memory"]
    # dummy_memory_im_input = torch.rand(1, 3, 289, 289).cuda()
    # dummy_memory_mask_input = torch.rand(1, 3, 289, 289).cuda()
    dummy_memory_im_input = torch.randint(0, 256, (1, 3, 289, 289)).to(torch.uint8).cuda()
    dummy_memory_mask_input = torch.randint(0, 256, (1, 1, 289, 289)).to(torch.uint8).cuda()
    torch.onnx.export(
        feature_extraction_memory,
        (dummy_memory_im_input, dummy_memory_mask_input),
        "/home/tengjunwan/project/ObjectTracking/STMTrack-main/STMTrack_FeatureExtractionMemory.onnx",
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )


def convertToONNX_FeatureExtractionQuery(model):
    feature_extraction_query = FeatureExtractionQuery(model)
    input_names = ["search_image"]
    output_names = ["query"]
    # dummy_search_im_input = torch.rand(1, 3, 289, 289).cuda()
    dummy_search_im_input = torch.randint(0, 256, (1, 3, 289, 289)).to(torch.uint8).cuda()
    torch.onnx.export(
        feature_extraction_query,
        (dummy_search_im_input,),
        "/home/tengjunwan/project/ObjectTracking/STMTrack-main/STMTrack_FeatureExtractionQuery.onnx",
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )



def convertToONNX_ReadMemoryAndHead(model):
    read_memory_and_head = ReadMemoryAndHead(model)
    # input_names = ["init_memory", "prev_memory", "query"]
    input_names = ["init_memory", "query"]
    output_names = ["score", "bbox"]
    dummy_init_memory = torch.rand(1, 512, 1, 25, 25).cuda()
    # dummy_prev_memory = torch.rand(1, 512, 1, 25, 25).cuda()
    dummy_query = torch.rand(1, 512, 25, 25).cuda()
    torch.onnx.export(
        read_memory_and_head,
        (dummy_init_memory, dummy_query),
        "/home/tengjunwan/project/ObjectTracking/STMTrack-main/STMTrack_ReadMemoryAndHead.onnx",
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )


def load_om_result(filename, dtype=np.float32):
    with open(filename, "rb") as f:
        result = np.frombuffer(f.read(), dtype=dtype)
    
    return result


def test_FeatureExtractionQuery(model):
    print("==================test consistency of query model==================")
    # pytorch model
    feature_extraction_query = FeatureExtractionQuery(model)
    test_img_path = "./om_result/query/resize_query289_black.png"
    input_image = cv2.imread(test_img_path)
    print(f"loading img:{test_img_path}, shape={input_image.shape}")
    pytorch_im_input = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).cuda() # (289,289,3) ->(1,3,289,289)
    print("==================test pytorch model==================")
    print(f"pytorch_im_input: dtype={pytorch_im_input.dtype}, shape={pytorch_im_input.shape}")
    with torch.no_grad():
        pytorch_result = feature_extraction_query(pytorch_im_input)

    print(f"pytorch_result: dtype={pytorch_result.dtype}, shape={pytorch_result.shape}")
    print("====pytorch_result first 20 elements====")
    print(pytorch_result.flatten()[:20])

    # onnx model
    print("==================test onnx model==================")
    backbone_session = ort.InferenceSession("STMTrack_FeatureExtractionQuery.onnx")
    onnx_im_input = pytorch_im_input.detach().cpu().numpy()  # （1,3,289,289）
    onnx_inputs = {
    "search_image": onnx_im_input,  
    }   
    onnx_results = backbone_session.run(None, onnx_inputs)
    onnx_result = onnx_results[0]
    print(f"onnx_result: dtype={onnx_result.dtype}, shape={onnx_result.shape}")
    print("====onnx_result first 20 elements====")
    print(onnx_result.flatten()[:20])
    


    # load om result 
    print("==================test om model(load from .result file)==================")
    om_result_path = "./om_result/query/query_om_black_1.result"
    # om_result_path = "./om_result/head/head_fq_input.result"
    om_result = load_om_result(om_result_path, dtype=np.float32)
    om_result = om_result.reshape(*onnx_result.shape)
    print(f"om_result: dtype={om_result.dtype}, shape={om_result.shape}")
    print("====om_result first 20 elements====")
    print(onnx_result.flatten()[:20])
    print("test_FeatureExtractionQuery done")


def test_FeatureExtractionMemory(model):
    print("==================test consistency of memory model==================")
    # pytorch model
    feature_extraction_memory = FeatureExtractionMemory(model)
    test_img_path = "./om_result/memory/resize_memory289_black.png"
    test_mask_path = "./om_result/memory/resize_mask289_black.png"
    input_image = cv2.imread(test_img_path)
    input_mask = cv2.imread(test_mask_path)
    input_mask = input_mask[:, :, [0]]  # (289,289,3) -> (289,289,1)
    print(f"loading img:{test_img_path}, shape={input_image.shape}")
    print(f"loading img:{test_mask_path}, shape={input_mask.shape}")
    pytorch_im_input = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).cuda() # (289,289,3) ->(1,3,289,289)
    pytorch_mask_input = torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0).cuda() # (289,289,1) ->(1,1,289,289)
    print("==================test pytorch model==================")
    print(f"pytorch_im_input: dtype={pytorch_im_input.dtype}, shape={pytorch_im_input.shape}")
    print(f"pytorch_mask_input: dtype={pytorch_mask_input.dtype}, shape={pytorch_mask_input.shape}")
    with torch.no_grad():
        pytorch_result = feature_extraction_memory(pytorch_im_input, pytorch_mask_input)

    print(f"pytorch_result: dtype={pytorch_result.dtype}, shape={pytorch_result.shape}")
    print("====pytorch_result first 20 elements====")
    print(pytorch_result.flatten()[:20])

    # onnx model
    print("==================test onnx model==================")
    backbone_session = ort.InferenceSession("STMTrack_FeatureExtractionMemory.onnx")
    onnx_im_input = pytorch_im_input.detach().cpu().numpy()  # （1,3,289,289）
    onnx_mask_input = pytorch_mask_input.detach().cpu().numpy()  # （1,1,289,289）
    onnx_inputs = {
        "memory_image": onnx_im_input,  
        "memory_mask": onnx_mask_input,
    }   
    onnx_results = backbone_session.run(None, onnx_inputs)
    onnx_result = onnx_results[0]
    print(f"onnx_result: dtype={onnx_result.dtype}, shape={onnx_result.shape}")
    print("====onnx_result first 20 elements====")
    print(onnx_result.flatten()[:20])
    


    # load om result 
    print("==================test om model(load from .result file)==================")
    om_result_path = "./om_result/memory/memory_om_black_1.result"
    # om_result_path = "./om_result/head/head_fm_input.result"
    om_result = load_om_result(om_result_path, dtype=np.float32)
    om_result = om_result.reshape(*onnx_result.shape)
    print(f"om_result: dtype={om_result.dtype}, shape={om_result.shape}")
    print("====om_result first 20 elements====")
    print(onnx_result.flatten()[:20])
    print("test_FeatureExtractionMemory done")


def test_ReadMemoryAndHead(model):
    print("==================test consistency of head model==================")
    # pytorch model
    feature_extraction_query = FeatureExtractionQuery(model)
    feature_extraction_memory = FeatureExtractionMemory(model)
    feature_extraction_head = ReadMemoryAndHead(model)

    # test data
    # test_query_img_path = "./om_result/query/resize_query289_black.png"
    # test_memory_img_path = "./om_result/memory/resize_memory289_black.png"
    # test_memory_mask_path = "./om_result/memory/resize_mask289_black.png"
    test_query_img_path = "./img_q_crop.png"
    test_memory_img_path = "./im_m_crop.png"
    test_memory_mask_path = "./mask.png"
    input_query_image = cv2.imread(test_query_img_path)
    input_memory_image = cv2.imread(test_memory_img_path)
    input_memory_mask = cv2.imread(test_memory_mask_path)
    input_memory_mask = input_memory_mask[:, :, [0]]  # (289,289,3) -> (289,289,1)
    input_memory_mask = (input_memory_mask / 255).astype(np.uint8)  # normalize to 0-1
    print(f"loading query img:{test_query_img_path}, shape={input_query_image.shape}")
    print(f"loading memory img:{test_memory_img_path}, shape={input_memory_image.shape}")
    print(f"loading memory mask img:{test_memory_mask_path}, shape={input_memory_mask.shape}")
    pytorch_query_img_input = torch.from_numpy(input_query_image).permute(2, 0, 1).unsqueeze(0).cuda() # (289,289,3) ->(1,3,289,289)
    pytorch_memory_img_input = torch.from_numpy(input_memory_image).permute(2, 0, 1).unsqueeze(0).cuda() # (289,289,3) ->(1,3,289,289)
    pytorch_memory_mask_input = torch.from_numpy(input_memory_mask).permute(2, 0, 1).unsqueeze(0).cuda() # (289,289,1) ->(1,1,289,289)
    print("==================test pytorch model==================")
    print(f"pytorch_query_img_input: dtype={pytorch_query_img_input.dtype}, shape={pytorch_query_img_input.shape}")
    print(f"pytorch_memory_img_input: dtype={pytorch_memory_img_input.dtype}, shape={pytorch_memory_img_input.shape}")
    print(f"pytorch_memory_mask_input: dtype={pytorch_memory_mask_input.dtype}, shape={pytorch_memory_mask_input.shape}")
    with torch.no_grad():
        fq = feature_extraction_query(pytorch_query_img_input)
        fm = feature_extraction_memory(pytorch_memory_img_input, pytorch_memory_mask_input)
        pytorch_score, pytorch_bbox = feature_extraction_head(fm, fq)

    print(f"pytorch_score: dtype={pytorch_score.dtype}, shape={pytorch_score.shape}")
    print(f"pytorch_bbox: dtype={pytorch_bbox.dtype}, shape={pytorch_bbox.shape}")
    print("====pytorch_score first 20 elements====")
    print(pytorch_score.flatten()[:20])
    print("====pytorch_bbox first 20 elements====")
    print(pytorch_bbox.flatten()[:20])

    # onnx model
    print("==================test onnx model==================")
    query_session = ort.InferenceSession("STMTrack_FeatureExtractionQuery.onnx")
    memory_session = ort.InferenceSession("STMTrack_FeatureExtractionMemory.onnx")
    head_session = ort.InferenceSession("STMTrack_ReadMemoryAndHead.onnx")
    onnx_query_img_input = pytorch_query_img_input.detach().cpu().numpy()  # （1,3,289,289）
    onnx_memory_img_input = pytorch_memory_img_input.detach().cpu().numpy()  # （1,3,289,289）
    onnx_memory_mask_input = pytorch_memory_mask_input.detach().cpu().numpy()  # （1,1,289,289）
    onnx_fq = query_session.run(None, {"search_image": onnx_query_img_input})[0]
    onnx_fm = memory_session.run(None, {"memory_image": onnx_memory_img_input, 
                                        "memory_mask": onnx_memory_mask_input})[0]
    onnx_score, onnx_bbox = head_session.run(None, {"init_memory": onnx_fm,
                                                    "query": onnx_fq})


    print(f"onnx_score: dtype={onnx_score.dtype}, shape={onnx_score.shape}")
    print(f"onnx_bbox: dtype={onnx_bbox.dtype}, shape={onnx_bbox.shape}")
    print("====onnx_score first 20 elements====")
    print(onnx_score.flatten()[:20])
    print("====onnx_bbox first 20 elements====")
    print(onnx_bbox.flatten()[:20])

    onnx_best_score_index = onnx_score.flatten().argmax()
    onnx_best_score = onnx_score.flatten()[onnx_best_score_index]
    onnx_best_bbox = onnx_bbox[0, onnx_best_score_index]
    print(f"onnx_best_score_index: {onnx_best_score_index}")
    print(f"onnx_best_score: {onnx_best_score}")
    print(f"onnx_best_bbox: {onnx_best_bbox}")

    

    # load om result 
    print("==================test om model(load from .result file)==================")
    om_score_path = "./om_result/pic/head_1.result"
    om_score = load_om_result(om_score_path, dtype=np.float32)
    om_score = om_score.reshape(*onnx_score.shape)
    om_bbox_path = "./om_result/pic/head_2.result"
    om_bbox = load_om_result(om_bbox_path, dtype=np.float32)
    om_bbox = om_bbox.reshape(*onnx_bbox.shape)
    print(f"om_score: dtype={om_score.dtype}, shape={om_score.shape}")
    print(f"om_bbox: dtype={om_bbox.dtype}, shape={om_bbox.shape}")
    print("====om_score first 20 elements====")
    print(om_score.flatten()[:20])
    print("====om_bbox first 20 elements====")
    print(om_bbox.flatten()[:20])

    om_best_score_index = om_score.flatten().argmax()
    om_best_score = om_score.flatten()[om_best_score_index]
    om_best_bbox = om_bbox[0, om_best_score_index]
    print(f"om_best_score_index: {om_best_score_index}")
    print(f"om_best_score: {om_best_score}")
    print(f"om_best_bbox: {om_best_bbox}")


    

    print("test_ReadMemoryAndHead done")





def next_multiple_of_4(x):
    return (x // 4 + 1) * 4


def get_crop(state, search_area_factor):
    search_size = search_area_factor * (state["w"] * state["h"])**0.5
    # maybe it must be multiples of 4 
    search_size = next_multiple_of_4(search_size)
    state["scale"] = search_size / model_input_size  

    crop_x = int(state["cx"] - search_size * 0.5)
    crop_y = int(state["cy"] - search_size * 0.5)
    crop_w = int(search_size)
    crop_h = int(search_size)
    return [crop_x, crop_y, crop_w, crop_h]

def safe_crop(image, crop):
    """
    crop a region from the image, allowing out-of-bounds crop areas
    Args:
        image:(H,W,C) or (H,C)
        crop: [x, y, w, h]
    """
    x, y, w, h = crop

    if len(image.shape) == 3:
        cropped_area =  np.zeros((h, w, image.shape[2]), dtype=image.dtype)
    else:
        cropped_area =  np.zeros((h, w), dtype=image.dtype)

    # determine the area in the source image that overlaps with the crop
    src_x1 = max(0, x)
    src_y1 = max(0, y)
    src_x2 = min(image.shape[1], x + w)
    src_y2 = min(image.shape[0], y + h)

    # determin the area in the cropped area that receive data
    dst_x1 = max(0, -x)
    dst_y1 = max(0, -y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # copy
    cropped_area[dst_y1: dst_y2, dst_x1: dst_x2] = image[src_y1: src_y2, src_x1: src_x2]

    return cropped_area





def resize(image, model_size):
    resize_image = cv2.resize(image, (model_size, model_size), 
                              interpolation = cv2.INTER_NEAREST)
    return resize_image


def get_size(w, h): # for post-processing
    pad = (w + h) * 0.5
    sz = np.sqrt(((w + pad) * (h + pad)))
    return sz





if __name__ == "__main__":
    # prepare model
    parser = make_parser()
    args = parser.parse_args()
    model = get_model(args)
    model.eval()
    model.cuda()
    feature_extraction_query = FeatureExtractionQuery(model)
    feature_extraction_memory = FeatureExtractionMemory(model)
    feature_extraction_head = ReadMemoryAndHead(model)


    # prepare data(images)
    image_dir = Path("/home/tengjunwan/project/ObjectTracking/STMTrack-main/test_images/polo")
    image_paths = sorted(list(image_dir.glob("*")))

    # constant params
    model_input_size = 289
    model_output_size = 25

    # hyper params
    search_area_factor = 4.0
    penalty_k = 0.04
    window = np.outer(np.hanning(model_output_size), np.hanning(model_output_size)).flatten()  # (625,) 
    window_influence = 0.21
    test_lr = 0.95

    # state used in stmtrack_tracker
    state = {"cx": 0,
             "cy": 0,
             "w": 0,
             "h": 0,
             "scale": 0}

    # ============================model inference logics in one piece=======================

    # ====part 0: crop resize and preprocess====
    # init setting(only called once)
    init_frame = cv2.imread(str(image_paths[0]))
    init_xywh = (670, 234, 145, 113) # xy means top left
    state["cx"] = init_xywh[0] + 0.5 * init_xywh[2]
    state["cy"] = init_xywh[1] + 0.5 * init_xywh[3]
    state["w"] = init_xywh[2]
    state["h"] = init_xywh[3]
    
    
    memory_crop = get_crop(state, search_area_factor)  # expand crop area according to bbox
    memory_image = safe_crop(init_frame, memory_crop)  # actual crop 
    
    # memory image resize
    memory_image_resize = resize(memory_image, model_input_size)

    # memory mask
    mask = np.zeros((model_input_size, model_input_size), dtype=np.uint8)
    mask_x1 = int(model_input_size / 2 - state["w"] * 0.5 / state["scale"])
    mask_y1 = int(model_input_size / 2 - state["h"] * 0.5 / state["scale"])
    mask_x2 = mask_x1 + int(state["w"] / state["scale"])
    mask_y2 = mask_y1 + int(state["h"] / state["scale"])
    mask[mask_y1: mask_y2, mask_x1: mask_x2] = 1

    # run memory model
    with torch.no_grad():
        memory_image_input = torch.from_numpy(memory_image_resize).cuda()  # (289, 289, 3)
        memory_image_input = memory_image_input.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 289, 289)
        mask_input = torch.from_numpy(mask).cuda()  # (289, 289)
        mask_input = mask_input.unsqueeze(0).unsqueeze(0)  # (1, 1, 289, 289)
        fm = feature_extraction_memory(memory_image_input, mask_input)

    # query image
    for i in range(1, len(image_paths)):
    # for i in range(1, 5):
        frame = cv2.imread(str(image_paths[i]))
        query_crop = get_crop(state, search_area_factor)
        query_image = safe_crop(frame, query_crop)
        query_image_resize = resize(query_image, model_input_size)

        # run query model
        with torch.no_grad():
            query_image_input = torch.from_numpy(query_image_resize).cuda()  # (289, 289, 3)
            query_image_input = query_image_input.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 289, 289)
            fq = feature_extraction_query(query_image_input)

        # run head model
        with torch.no_grad():
            score, bbox = feature_extraction_head(fm, fq)  # (1, 625, 1), (1, 625, 4)

        # reshape
        score = score[0, :, 0]  # (625,)
        bbox = bbox[0]  # (625, 4)

        # back to numpy
        score = score.detach().cpu().numpy()
        bbox = bbox.detach().cpu().numpy()

        # ========post-process========
        # 1) penalty over score
        # size change
        bbox_w = bbox[..., 2] - bbox[..., 0]  # (625,)
        bbox_h = bbox[..., 3] - bbox[..., 1]
        prev_size = get_size(state["w"] / state["scale"], state["h"] / state["scale"])  
        current_size = get_size(bbox_w, bbox_h)  # (625,)
        size_change = np.maximum(prev_size / current_size, current_size / prev_size)

        # ratio change
        prev_ratio = state["w"] / state["h"]
        current_ratio = bbox_w / bbox_h
        ratio_change = np.maximum(prev_ratio / current_ratio, current_ratio / prev_ratio)

        # penalty(due to deformation)
        penalty = np.exp((1 - size_change * ratio_change) * penalty_k)  # (625,)
        
        pscore = penalty * score

        # reduce pscore by rapid position change
        pscore = pscore * (1 - window_influence) + window * window_influence

        # get best pscore id
        best_pscore_id = np.argmax(pscore)

        # 2) get bbox (update by EMA)
        # choose best pscore prediction
        bbox_best = bbox[best_pscore_id]  # (4,) xyxy

        # back to original scale
        bbox_best = bbox_best * state["scale"]

        # xyxy to cxcywh
        pred_cx = (bbox_best[2] + bbox_best[0]) * 0.5 
        pred_cx = pred_cx + state["cx"] - (model_input_size / 2) * state["scale"]  # back to global coordinate
        pred_cy = (bbox_best[3] + bbox_best[1]) * 0.5 
        pred_cy = pred_cy + + state["cy"] - (model_input_size / 2) * state["scale"]
        pred_w = bbox_best[2] - bbox_best[0]
        pred_h = bbox_best[3] - bbox_best[1]

        # update wh by EMA
        lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        pred_w = state["w"] * (1 - lr) + pred_w * lr
        pred_h = state["h"] * (1 - lr) + pred_h * lr
        
        # =======final results=======
        final_score = pscore[best_pscore_id]
        final_bbox = [pred_cx, pred_cy, pred_w, pred_h]

        # udpate state for next frame tracking
        state["cx"] = pred_cx
        state["cy"] = pred_cy
        state["w"] = pred_w
        state["h"] = pred_h


        # visualization
        vis_save_dir = Path("/home/tengjunwan/project/ObjectTracking/STMTrack-main/test_images/polo_output")
        save_path = vis_save_dir / image_paths[i].name
        frame_disp = frame.copy()
        
        cv2.rectangle(frame_disp, 
                      (int(state["cx"] - state["w"] * 0.5), int(state["cy"] - state["h"] * 0.5)),
                      (int(state["cx"] + state["w"] * 0.5), int(state["cy"] + state["h"] * 0.5)), 
                      (0, 0, 255), thickness=3)
        cv2.putText(frame_disp,  f"{final_score:.2f}", 
                    (int(state["cx"] - state["w"] * 0.5), int(state["cy"] - state["h"] * 0.5)), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                    (0, 0, 255), thickness=3)
        cv2.imwrite(str(save_path), frame_disp)
        print("done")

        

    print("logic in one piece done!")




    # convert to ONNX
    convert_flag = False
    if convert_flag:
        # print("converting STMTrack_FeatureExtractionMemory.onnx...")
        # convertToONNX_FeatureExtractionMemory(model)
        # print("done.")

        # print("converting STMTrack_FeatureExtractionQuery.onnx...")
        # convertToONNX_FeatureExtractionQuery(model)
        # print("done.")

        # print("converting STMTrack_ReadMemoryAndHead.onnx...")
        # convertToONNX_ReadMemoryAndHead(model)
        # print("done.")
        pass
    

    # test consistency
    test_consistency_flag = False
    if test_consistency_flag:
        # test_FeatureExtractionQuery(model)

        # test_FeatureExtractionMemory(model)

        # test_ReadMemoryAndHead(model)
        pass


    

