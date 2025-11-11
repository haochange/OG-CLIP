import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint,device="cpu",))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = Image.open(r"./demo/P0000.png")
    predictor.set_image(image)
    masks, _, _ = predictor.predict()
    print(masks)