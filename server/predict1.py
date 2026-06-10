import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import io
import torch.nn.functional as F

# ============================================================
# LOAD MODEL
# ============================================================

ckpt = torch.load("best_mobilevit.pth", map_location="cpu")

classes = ckpt["classes"]
num_classes = len(classes)

model = timm.create_model(
    "mobilevit_s",
    pretrained=False,
    num_classes=num_classes
)

model.load_state_dict(ckpt["model_state"])
model.eval()

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

CONFIDENCE_THRESHOLD = 0.4

# ============================================================
# PREDICT
# ============================================================

def predict_disease(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Không thể đọc ảnh. Hãy thử chụp lại.")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)

    confidence, predicted_idx = torch.max(probabilities, 1)

    confidence_val = confidence.item()
    disease_name = classes[predicted_idx.item()]

    if confidence_val < CONFIDENCE_THRESHOLD:
        return (
            f"{disease_name} "
            f"(độ tin cậy thấp: {confidence_val:.0%} — hãy chụp lại ảnh rõ hơn)"
        )

    return disease_name


def predict_disease_detailed(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Không thể đọc ảnh.")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]

    top3_probs, top3_idx = torch.topk(
        probabilities,
        min(3, num_classes)
    )

    top3 = [
        {
            "disease": classes[idx.item()],
            "confidence": prob.item()
        }
        for prob, idx in zip(top3_probs, top3_idx)
    ]

    best = top3[0]

    return {
        "disease": best["disease"],
        "confidence": best["confidence"],
        "low_confidence": best["confidence"] < CONFIDENCE_THRESHOLD,
        "top3": top3
    }
