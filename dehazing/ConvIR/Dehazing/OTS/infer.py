import torch
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as transforms
import time
import torch.nn.functional as f

# ---------------------------------------------------


def infer_single_image(model, model_path: str, image_path: str, save_path: str = None):
    """
    Perform inference on a single image without using dataloader

    Args:
        model: The ConvIR model
        model_path: Path to the trained model checkpoint
        image_path: Path to the input hazy image
        save_path: Path to save the dehazed result (optional)
    """
    # Load model weights
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load and preprocess image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Get original dimensions
    h, w = input_tensor.shape[2], input_tensor.shape[3]

    # Pad to multiple of 32
    factor = 32
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = f.pad(input_tensor, (0, padw, 0, padh), "reflect")

    # Inference
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        pred = model(input_tensor)[2]
        pred = pred[:, :, :h, :w]  # Remove padding

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # Post-process
    pred_clip = torch.clamp(pred, 0, 1)

    print(f"Inference time: {elapsed:.4f} seconds")

    # Save result if path provided
    if save_path:
        pred_clip += 0.5 / 255
        result_image = F.to_pil_image(pred_clip.squeeze(0).cpu(), "RGB")

        # Create a side-by-side comparison with original image
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
