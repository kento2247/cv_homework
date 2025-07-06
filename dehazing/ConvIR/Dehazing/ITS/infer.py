import torch
from torchvision.transforms import functional as F
import time
import torch.nn.functional as f
from PIL import Image
import torchvision.transforms as transforms


def infer_single_image(model, model_path: str, image_path: str, save_path: str = None):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    h, w = input_tensor.shape[2], input_tensor.shape[3]
    factor = 32
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = f.pad(input_tensor, (0, padw, 0, padh), "reflect")

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        pred = model(input_tensor)[2]
        pred = pred[:, :, :h, :w]

    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print(f"Inference time: {elapsed:.4f} seconds")

    pred_clip = torch.clamp(pred, 0, 1)

    if save_path:
        pred_clip += 0.5 / 255
        result_image = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")

        # Load original image for side-by-side comparison
        original_image = Image.open(image_path).convert("RGB")

        # Ensure both images have the same size
        if result_image.size != original_image.size:
            result_image = result_image.resize(
                original_image.size, Image.Resampling.LANCZOS
            )

        # Create side-by-side image
        total_width = original_image.width + result_image.width
        max_height = max(original_image.height, result_image.height)

        side_by_side = Image.new("RGB", (total_width, max_height))
        side_by_side.paste(original_image, (0, 0))
        side_by_side.paste(result_image, (original_image.width, 0))

        side_by_side.save(save_path)
        print(f"Side-by-side comparison saved to: {save_path}")

    return pred_clip.squeeze(0).cpu()
