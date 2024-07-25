import openai
import os
import sys
openai.api_key = os.environ.get("OPENAI_API_KEY")
from openai import OpenAI
import base64
import requests
from easydict import EasyDict as edict
sys.path.append('/root/mount/Matterport3DSimulator')
os.chdir('/root/mount/Matterport3DSimulator')
from tasks.R2R.ns.backend import *

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


def groundedsam(classes: str, image: cv2.imread):
    global GSAM
    if GSAM is None:
        GSAM = FastGSAM()
    ann_img, detections, per_class_mask = GSAM.predict_and_segment_on_image(img=image,
                                                                            text_prompts=classes)
    cv2.imshow("GroundedSAM", ann_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ann_img, detections, per_class_mask


if __name__ == "__main__":
    image_path = "test_image2.png"
    pil_img = Image.open(image_path)
    text, openai_response = gpt4v(
        context="You are a good VQA assistant. You always answer questions in very very detail. You always end your answers with 'END'.",
        prompt="Describe the image, especially the objects and pathways with regards to navigation. Clearly demarcate three different areas: left, middle and right, and describe the environment in each with regards to navigation.",
        images=[pil_img],
        model="gpt-4o-mini",
        img_detail="auto",
        img_mode="pil"
    )
    print(text)
    print(openai_response.choices[0].finish_reason)

    # cv2_img = cv2.imread(image_path)
    # groundedsam(classes=["door", "bar chairs"], image=cv2_img)
