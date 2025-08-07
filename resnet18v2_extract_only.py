import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Model Definitions -----------

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.features_dim = base_model.fc.in_features

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)


class ClassifierHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.intermediate = nn.Linear(in_features, 4)  # 4-neuron FC layer
        self.final = nn.Linear(4, 1)
        self.hidden_output = None

    def forward(self, x):
        x = self.intermediate(x)
        self.hidden_output = x
        x = self.final(x)
        return x

# ----------- Load Model -----------

feature_extractor = ResNetFeatureExtractor(pretrained=False)
head = ClassifierHead(in_features=feature_extractor.features_dim)
model = nn.Sequential(feature_extractor, head).to(DEVICE)

# Load trained weights
model.load_state_dict(torch.load("resnet18_fc4_lr0.0005_20250807_111526.pt", map_location=DEVICE))
model.eval()

# ----------- Transforms -----------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------- Prediction Function -----------

@torch.no_grad()
def predict_image(model, image_path, transform, task="classification"):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    output = model(input_tensor)

    fc4_vector = model[1].hidden_output.detach().cpu().numpy().flatten()

    if task == "classification":
        prob = torch.sigmoid(output).view(-1)
        pred_class = (prob >= 0.5).int().item()
        confidence = prob.item()
        return pred_class, confidence, fc4_vector
    else:
        score = output.item()
        return score, fc4_vector

# ----------- Run Inference on Folder -----------

extract_path = "./images_for_feature_extraction"
task_type = "classification"

extracted_features = pd.DataFrame(columns=['filename', 'pred_class', 'confidence', 'f1', 'f2', 'f3', 'f4'])

for img in os.listdir(extract_path):
    if not img.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(extract_path, img)
    pred_class, confidence, fc4_vector = predict_image(model, img_path, transform, task=task_type)
    extracted_features.loc[len(extracted_features)] = [img, pred_class, confidence] + list(fc4_vector)

# Save to CSV
extracted_features.to_csv("extracted_features_fc4.csv", index=False)
print("✅ Feature extraction completed and saved to extracted_features_fc4.csv")
