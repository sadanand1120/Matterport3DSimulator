import openai
import os
import sys
openai.api_key = os.environ.get("OPENAI_API_KEY")
from openai import OpenAI
import base64
import requests
import PIL
from PIL import Image
from easydict import EasyDict as edict
sys.path.append('/root/mount/Matterport3DSimulator')
os.chdir('/root/mount/Matterport3DSimulator')
import cv2
import torch
import open3d as o3d
from tasks.R2R.ns.backend import *
from thirdparty.DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

GSAM = None


def gpt4(context: str, prompt: str, temperature: float = 0.2, max_tokens: int = None, stop: str = "END", seed: int = 0) -> str:
    openai_client = OpenAI()
    model = "gpt-4"
    openai_response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        stop=stop,
        seed=seed,
        max_tokens=max_tokens,
    )
    text = openai_response.choices[0].message.content.strip()
    return text, openai_response  # check openai_response.choices[0].finish_reason


def gpt4v(context: str, prompt: str, images: list, model: str = "gpt-4o-mini", img_detail: str = "auto", img_mode: str = "pil", temperature: float = 0.2, max_tokens: int = None, stop: str = "END", seed: int = 0) -> str:
    """
    model: "gpt-4o-mini" / "gpt-4o" / "gpt-4-turbo"
    img_detail: "low" / "auto" / "high"
    img_mode: "pil" / "path"
    """
    base64_images = [encode_image(img, mode=img_mode) for img in images]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    if img_mode == "pil":
        img_ext = "png"
    elif img_mode == "path":
        img_ext = images[0].split(".")[-1].lower()
        if img_ext in ["jpg", "jpeg"]:
            img_ext = "jpeg"
    else:
        raise ValueError(f"Invalid image mode: {img_mode}")
    user_content = []
    user_content.append({"type": "text", "text": prompt})
    for i, base64_image in enumerate(base64_images):
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{img_ext};base64,{base64_image}", "detail": img_detail}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if max_tokens is None:
        payload.pop("max_tokens")
    openai_response = edict(requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json())
    text = openai_response.choices[0].message.content.strip()
    return text, openai_response  # check openai_response.choices[0].finish_reason


@torch.inference_mode()
def groundedsam(classes: str, image: cv2.imread):
    global GSAM
    if GSAM is None:
        GSAM = FastGSAM()
    ann_img, detections, per_class_mask = GSAM.predict_and_segment_on_image(img=image,
                                                                            text_prompts=classes)
    return ann_img, detections, per_class_mask


@torch.inference_mode()
def depth(image: cv2.imread):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb'
    dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20  # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(f'thirdparty/DepthAnythingV2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    depth = model.infer_image(image)  # HxW depth map in meters in numpy
    return depth


def pc_from_depth(image: cv2.imread, z: np.ndarray, vfov_deg: float = 60):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = z.shape
    K = get_ideal_intrinsic_matrix(W, H, vfov_deg)
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = (x - K[0, 2]) / K[0, 0]
    y = (y - K[1, 2]) / K[1, 1]
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.asarray(image).reshape(-1, 3) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # # o3d.visualization.draw_geometries([pcd])  # visualize the point cloud
    # # pcd = o3d.io.read_point_cloud("pointcloud.ply")  # read the point cloud
    # # o3d.io.write_point_cloud("pointcloud.ply", pcd)  # save the point cloud
    return z, pcd


if __name__ == "__main__":
    image_path = "src/driver/rgb.png"
    # pil_img = Image.open(image_path)
    # text, openai_response = gpt4v(
    #     context="You are a good VQA assistant. You always answer questions in very very detail. You always end your answers with 'END'.",
    #     prompt="Describe the image, especially the objects and pathways with regards to navigation. Clearly demarcate three different areas: left, middle and right, and describe the environment in each with regards to navigation.",
    #     images=[pil_img],
    #     model="gpt-4o-mini",
    #     img_detail="auto",
    #     img_mode="pil"
    # )
    # print(text)
    # print(openai_response.choices[0].finish_reason)

    # cv2_img = cv2.imread(image_path)
    # groundedsam(classes=["door", "bar chairs"], image=cv2_img)

    cv2_img = cv2.imread(image_path)
    dd = depth(cv2_img)
    cv2.imshow("Depth", (dd*4000).astype(np.uint16))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    read_dd = cv2.imread("src/driver/depth.png", cv2.IMREAD_UNCHANGED) / 4000
