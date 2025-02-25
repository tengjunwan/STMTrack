import argparse

import torch
import numpy as np
import torch.nn as nn

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

    def forward(self, im_memory, mask_memory):
        fm = self.model.memorize(im_memory, mask_memory)
        return fm
    
class FeatureExtractionQuery(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, im_query):
        fq = self.model.basemodel_q(im_query)
        fq = model.neck_q(fq)  
        return fq
    
class ReadMemoryAndHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.q_size = 289  # fixed 

    def forward(self, fm_init, fm_prev, fq):
        fm = torch.cat([fm_init, fm_prev], dim=2)
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
    dummy_memory_im_input = torch.rand(1, 3, 289, 289).cuda()
    dummy_memory_mask_input = torch.rand(1, 1, 289, 289).cuda()
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
    dummy_search_im_input = torch.rand(1, 3, 289, 289).cuda()
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
    input_names = ["init_memory", "prev_memory", "query"]
    output_names = ["score", "bbox"]
    dummy_init_memory = torch.rand(1, 512, 1, 25, 25).cuda()
    dummy_prev_memory = torch.rand(1, 512, 1, 25, 25).cuda()
    dummy_query = torch.rand(1, 512, 25, 25).cuda()
    torch.onnx.export(
        read_memory_and_head,
        (dummy_init_memory, dummy_prev_memory, dummy_query),
        "/home/tengjunwan/project/ObjectTracking/STMTrack-main/STMTrack_ReadMemoryAndHead.onnx",
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )



if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    model = get_model(args)
    model.eval()
    model.cuda()

    # model inference logics

    # part 0: crop resize and preprocess
    # TODO

    with torch.no_grad():
        # part 1-1: test "memorize phase": memory image embedding
        # input image: (1, 3, 289, 289) + input mask: (1, 1, 289, 289)
        dummy_im_input = torch.rand(1, 3, 289, 289).cuda()
        dummy_mask = torch.rand(1, 1, 289, 289).cuda()
        fm = model.memorize(dummy_im_input, dummy_mask)  # basemodel_m + neck_m -> (B=1, C=512, T=1, H=25, W=25)
        # memorize is equivalent to:
        #       fm = self.basemodel_m(im_crop, fg_bg_label_map)
        #       fm = self.neck_m(fm)
        #       fm = fm.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # B, C, T, H, W

        # part 1-2: test search image embedding(query part)
        dummy_search_im_input = torch.rand(1, 3, 289, 289).cuda()
        fq = model.basemodel_q(dummy_search_im_input)
        fq = model.neck_q(fq)  # B=1, C=512, H=25, W=25

        # part 2: attention(memory_read) and prediction head(classification+centerness+box)
        fm_only_two_frames = torch.cat([fm, fm], dim=2)  # (B=1, C=512, T=2, H=25, W=25)
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = \
            model.head(fm_only_two_frames, fq, q_size=dummy_search_im_input.size(-1))
        # apply sigmoid
        fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
        fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
        # apply centerness correction
        fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final

        score = fcos_score_final
        bbox = fcos_bbox_final

    # part 4: post-process(no models involved)
    # TODO


    # convert to ONNX
    convert_flag = False
    if convert_flag:
        print("converting STMTrack_FeatureExtractionMemory.onnx...")
        convertToONNX_FeatureExtractionMemory(model)
        print("done.")

        print("converting STMTrack_FeatureExtractionQuery.onnx...")
        convertToONNX_FeatureExtractionQuery(model)
        print("done.")

        print("converting STMTrack_ReadMemoryAndHead.onnx...")
        convertToONNX_ReadMemoryAndHead(model)
        print("done.")
    



    

