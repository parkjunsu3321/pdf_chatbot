import os
import base64
import re
from openai import OpenAI
from tqdm import tqdm
from config import Config

config = Config()
client = OpenAI(api_key=config.OPENAI_API_KEY)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(image_path: str, image_type: str = "image") -> str:
    """Vision LLM으로 이미지 설명 생성"""
    base64_image = encode_image(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".")
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    if image_type == "table":
        prompt = "이 표 이미지의 내용을 텍스트로 정확히 설명해주세요. 표의 행과 열 데이터를 빠짐없이 포함해주세요."
    else:
        prompt = "이 이미지의 내용을 자세히 설명해주세요."

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}},
            ],
        }],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def get_page_descriptions(table_dir: str, img_dir: str) -> dict:
    """페이지별 이미지 설명을 딕셔너리로 반환. {page_num: [설명1, 설명2, ...]}"""
    page_pattern = re.compile(r"page_(\d+)_")
    descriptions = {}

    # 처리할 파일 목록 수집
    tasks = []
    if os.path.exists(table_dir):
        for fname in sorted(os.listdir(table_dir)):
            match = page_pattern.match(fname)
            if match:
                tasks.append((os.path.join(table_dir, fname), int(match.group(1)), "table"))

    if os.path.exists(img_dir):
        for fname in sorted(os.listdir(img_dir)):
            match = page_pattern.match(fname)
            if match:
                tasks.append((os.path.join(img_dir, fname), int(match.group(1)), "image"))

    # Vision LLM으로 설명 생성
    for filepath, page_num, img_type in tqdm(tasks, desc="Generating image descriptions"):
        try:
            desc = describe_image(filepath, image_type=img_type)
            label = "표 설명" if img_type == "table" else "이미지 설명"
            descriptions.setdefault(page_num, []).append(f"[{label}] {desc}")
        except Exception as e:
            print(f"  Warning: {os.path.basename(filepath)} 설명 생성 실패 - {e}")

    return descriptions
