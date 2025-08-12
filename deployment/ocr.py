import os
import tempfile
from io import BytesIO

import numpy as np
import requests
import torch
from crnn import CRNN
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import Response
from PIL import Image
from torchvision import transforms

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

import matplotlib.pyplot as plt

# Constants
TEXT_DEC_MODEL_PATH = "/Users/phamtu/Documents/Project_AI/Scene_Text_Recognition/Scene_Text_Recognition/weights/best.pt"
OCR_MODEL_PATH = "/Users/phamtu/Documents/Project_AI/Scene_Text_Recognition/Scene_Text_Recognition/weights/ocr_model.pt"

# Character set configuration
CHARS = "\n0123456789abcdefghijklmnopqrstuvwxyz-"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

# Model configuration
HIDDEN_SIZE = 256
N_LAYER = 3
UNFREEZE_LAYERS = 3
DROPOUT_PRO = 0.2

app = FastAPI()


class APIIngress:
    def __init__(self, ocr_handle):
        self.handle = ocr_handle

    async def process_image(self, image_data: bytes) -> Response:
        try:
            print(f"Processing image of size: {len(image_data)} bytes")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(image_data)
                temp_file_path = temp_file.name

            # Request OCR results using the temp file path
            predictions = self.handle.process_image.remote(temp_file_path)

            # Load the image and draw predictions
            image = Image.open(temp_file_path)
            annotated_img = await self.handle.draw_predictions.remote(
                image,
                predictions
            )

            file_stream = BytesIO()
            annotated_img.save(file_stream, format='PNG')
            file_stream.seek(0)

            os.unlink(temp_file_path)

            return Response(
                content=file_stream.getvalue(),
                media_type="image/png",
                headers={"X-Predictions": str(predictions)}
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing image {e}")

    @app.get('/ocr')
    async def ocr_url(self, image_url: str = Query(...)):
        """Endpoint for processing images from  URLs"""
        try:
            response = requests.get(url=image_url)
            print(response.status_code, response.text)
            response.raise_for_status()
            return await self.process_image(response.content)

        except requests.RequestException as re:
            raise HTTPException(
                status_code=400, detail=f"Error downloading image {re}")

    @app.post('/ocr/upload')
    async def ocr_upload(self, file: UploadFile = File(...)):
        try:
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, detail=f"File must be an image")
            content = await file.read()
            return await self.process_image(content)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing uploaded file: {e}")


class OCRService:
    def __init__(self, det_model, reg_model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det_model = det_model.to(self.device)
        self.reg_model = reg_model.to(self.device)
        self.data_transform = transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )

    def text_detection(self, image_path):
        text_detected = self.det_model(image_path, verbose=False)[0]

        return (
            text_detected.bboxes.xyxy.tolist(),
            text_detected.bboxes.cls.tolist(),
            text_detected.bboxes.conf.tolist(),
            text_detected.names
        )

    def text_recognition(self, image):
        transformed_image = self.data_transform(
            image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.reg_model(transformed_image).cpu()
        text = self.decode(logits.permute(1, 0, 2).argmax(2), IDX_TO_CHAR)
        return text

    def decode(self, encoded_sequences, idx_to_char, blank_char="-"):
        decoded_sequences = []

        for seq in encoded_sequences:
            decoded_label = []
            prev_char = None  # To track the previous character

            for token in seq:
                if token != 0:  # Ignore padding (token = 0)
                    char = idx_to_char[token.item()]
                    # Append the character if it's not a blank or the same as the previous character
                    if char != blank_char:
                        if char != prev_char or prev_char == blank_char:
                            decoded_label.append(char)
                    prev_char = char  # Update previous character

            decoded_sequences.append("".join(decoded_label))

        return decoded_sequences

    def process_image(self, image_path: str):

        try:
            bboxes, classes, confs, names = self.text_detection(image_path)

            # Load image
            image = Image.open(image_path)
            predictions = []

            for bbox, cls, conf in zip(bboxes, classes, confs):
                x1, y1, x2, y2 = bbox
                name = names[int(cls)]

                cropped_image = image.crop((x1, y1, x2, y2))
                transcribed_text = self.text_recognition(cropped_image)
                predictions.append((bbox, name, conf, transcribed_text))
            return predictions

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing image (ocr): {e}, image path: {image_path}")

    def draw_predictions(self, image, predictions):
        image_array = np.array(image)

        annotator = Annotator(image_array, font="Arial.ttf", pil=False)

        predictions = sorted(
            predictions, key=lambda x: x[0][1]
        )

        for bbox, class_name, conf, text in predictions:
            x1, y1, x2, y2 = [int(coor) for coor in bbox]
            color = colors(hash(class_name) % 20, True)
            label = f"{class_name[:3]} ({conf:.2}): {text}"

            annotator.box_label(
                [x1, y1, x2, y2], label, color, txt_color=(255, 255, 255)
            )
        return Image.fromarray(annotator.result())


detection_model = YOLO(TEXT_DEC_MODEL_PATH)
recognition_model = CRNN(
    vocab_size=len(CHARS),
    hidden_size=HIDDEN_SIZE,
    num_layer=N_LAYER,
    unfreeze_layers=UNFREEZE_LAYERS,
    dropout=DROPOUT_PRO
)

recognition_model.load_state_dict(torch.load(
    OCR_MODEL_PATH, map_location=torch.device('cpu')))
recognition_model.eval()


entrypoint = APIIngress(
    OCRService(
        reg_model=recognition_model,
        det_model=detection_model
    )
)
