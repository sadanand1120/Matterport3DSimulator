import base64
import requests
from easydict import EasyDict as edict
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
import os
import sys
import supervision as sv
from segment_anything import sam_hq_model_registry, SamPredictor
from groundingdino.util.inference import Model


class FastGSAM:
    def __init__(self, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.6, ann_thickness=2, ann_text_scale=0.3, ann_text_thickness=1, ann_text_padding=5):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GROUNDING_DINO_CONFIG_PATH = "thirdparty/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT_PATH = "thirdparty/Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth"
        self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH, device=self.DEVICE)
        self.SAM_ENCODER_VERSION = "vit_h"
        self.SAM_CHECKPOINT_PATH = "thirdparty/Grounded-Segment-Anything/weights/sam_hq_vit_h.pth"
        self.sam = sam_hq_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.SAM_CHECKPOINT_PATH)
        self.sam.to(device=self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam)
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.NMS_THRESHOLD = nms_threshold
        self.box_annotator = sv.BoxAnnotator(
            thickness=ann_thickness,
            text_scale=ann_text_scale,
            text_thickness=ann_text_thickness,
            text_padding=ann_text_padding
        )
        self.mask_annotator = sv.MaskAnnotator()

    def predict_on_image(self, img, text_prompts, do_nms=True, box_threshold=None, text_threshold=None, nms_threshold=None):
        """
        Performs zero-shot object detection using grounding DINO on image.
        img: A x B x 3 np cv2 BGR image
        text_prompts: list of text prompts / classes to predict
        Returns:
            annotated_frame: cv2 BGR annotated image with boxes and labels
            detections:
                - xyxy: (N, 4) boxes (float pixel locs) in xyxy format
                - confidence: (N, ) confidence scores
                - class_id: (N, ) class ids, i.e., idx of text_prompts
        """
        if box_threshold is None:
            box_threshold = self.BOX_THRESHOLD
        if text_threshold is None:
            text_threshold = self.TEXT_THRESHOLD
        if nms_threshold is None:
            nms_threshold = self.NMS_THRESHOLD
        detections = self.grounding_dino_model.predict_with_classes(
            image=img,
            classes=text_prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        # print(f"Detected {len(detections.xyxy)} boxes")
        if do_nms:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                nms_threshold
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            # print(f"After NMS: {len(detections.xyxy)} boxes")
        labels = [
            f"{text_prompts[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_frame = self.box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
        return annotated_frame, detections

    @staticmethod
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=False,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    @staticmethod
    def get_per_class_mask(img, masks, class_ids, num_classes):
        """
        Create a per-class segmentation mask.
        Parameters:
            masks: N x H x W array, where N is the number of masks
            class_ids: (N,) array of corresponding class ids
            num_classes: Total number of classes, C
        Returns:
            per_class_mask: C x H x W array
        """
        H, W = img.shape[0], img.shape[1]
        per_class_mask = np.zeros((num_classes, H, W), dtype=bool)
        if len(masks) == 0:
            return per_class_mask
        for i in range(num_classes):
            class_idx = np.where(class_ids == i)[0]
            if class_idx.size > 0:
                per_class_mask[i] = np.any(masks[class_idx], axis=0)
        return per_class_mask

    @torch.inference_mode()
    def predict_and_segment_on_image(self, img, text_prompts, do_nms=True, box_threshold=None, text_threshold=None, nms_threshold=None):
        """
        Performs zero-shot object detection using grounding DINO and segmentation using HQ-SAM on image.
        img: H x W x 3 np cv2 BGR image
        text_prompts: list of text prompts / classes to predict
        Returns:
            annotated_frame: cv2 BGR annotated image with boxes and labels and segment masks
            detections: If there are N detections,
                - xyxy: (N, 4) boxes (int pixel locs) in xyxy format
                - confidence: (N, ) confidence scores
                - class_id: (N, ) class ids, i.e., idx of text_prompts
                - mask: (N, H, W) boolean segmentation masks, i.e., True at locations belonging to corresponding class
        """
        _, detections = self.predict_on_image(img, text_prompts, do_nms=do_nms, box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold)
        detections.mask = FastGSAM.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        labels = [
            f"{text_prompts[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_image = self.mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        detections.xyxy = detections.xyxy.astype(np.int32).reshape((-1, 4))
        return annotated_image, detections, FastGSAM.get_per_class_mask(img, detections.mask, detections.class_id, len(text_prompts))


def encode_image(img, mode="path"):
    if mode == "path":
        with open(img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif mode == "pil":
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError(f"Invalid encode mode: {mode}")


def get_ideal_intrinsic_matrix(width: int, height: int, vfov_deg: float):
    """
    Get the ideal intrinsic matrix for a pinhole camera model.
    Parameters:
        width: Image width
        height: Image height
        vfov_deg: vertical field of view in degrees
    Returns:
        K: 3x3 intrinsic matrix
    """
    vfov = np.deg2rad(vfov_deg)
    hfov = 2 * np.arctan(np.tan(vfov / 2) * width / height)
    fx = width / (2 * np.tan(hfov / 2))
    fy = height / (2 * np.tan(vfov / 2))
    cx = width / 2
    cy = height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


if __name__ == "__main__":
    # print numpy arrays without scientific notation
    np.set_printoptions(suppress=True)
    print(get_ideal_intrinsic_matrix(1280, 1024, 50.8))
