from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

processor = DonutProcessor.from_pretrained("./donut_sensitive_model")
model = VisionEncoderDecoderModel.from_pretrained("./donut_sensitive_model")
model.eval()

image = Image.open("some_test_image.jpg").convert("RGB")
task_prompt = "<s_sensitivedetect>"

inputs = processor(image, return_tensors="pt").pixel_values
decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids

with torch.no_grad():
    outputs = model.generate(inputs, decoder_input_ids=decoder_input_ids, max_length=512)
    prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print("Prediction:", prediction)