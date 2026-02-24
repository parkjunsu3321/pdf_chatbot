import torch
import os
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from pdf2image import convert_from_path
from tqdm import tqdm
from config import Config

config = Config()


class TableExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DetrImageProcessor.from_pretrained(config.TABLE_DET_MODEL)
        self.model = TableTransformerForObjectDetection.from_pretrained(config.TABLE_DET_MODEL).to(self.device)

    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def extract_tables(self, pdf_path, output_folder="extracted_tables", padding=20):
        os.makedirs(output_folder, exist_ok=True)
        images = convert_from_path(pdf_path, dpi=300)

        for page_idx, image in enumerate(tqdm(images, desc="Extracting Tables")):
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs, threshold=0.98, target_sizes=target_sizes)[0]

            boxes = results["boxes"].tolist()
            scores = results["scores"].tolist()

            # NMS (중복 박스 제거)
            keep_indices = []
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            while sorted_indices:
                current = sorted_indices.pop(0)
                keep_indices.append(current)
                sorted_indices = [i for i in sorted_indices if self.compute_iou(boxes[current], boxes[i]) < 0.5]

            for i, idx in enumerate(keep_indices):
                box = boxes[idx]
                x_min, y_min, x_max, y_max = box
                width, height = image.size
                cropped = image.crop((
                    max(0, x_min - padding), max(0, y_min - padding),
                    min(width, x_max + padding), min(height, y_max + padding),
                ))
                cropped.save(os.path.join(output_folder, f"page_{page_idx + 1}_table_{i + 1}.png"))
