import os
import pandas as pd
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim import Adam
import datetime

dataset_source = 'dataset_reduced_pre_post'

# ----------- Configuration -----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 150
LEARNING_RATE = 5e-4
CLASS_MODE = True  # True = classification, False = regression
# VAL_RATIO = 0.2

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# -------------------------------------


# ----------- Transforms -----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# ----------------------------------


# ----------- Dataset Setup -----------
# Wrap a transform that remaps the target
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

class RemappedImageFolder(Dataset):
    def __init__(self, imagefolder_dataset, label_map):
        self.dataset = imagefolder_dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        remapped_label = self.label_map[label]
        return img, remapped_label



if CLASS_MODE:
    full_dataset = datasets.ImageFolder(root=dataset_source, transform=transform)

    def get_patient_id(path):
        filename = os.path.basename(path)
        return filename.split('_')[0]

    # Map patient to indices
    patient_to_indices = {}
    for idx in range(len(full_dataset)):
        path, _ = full_dataset.samples[idx]
        pid = get_patient_id(path)
        patient_to_indices.setdefault(pid, []).append(idx)

    # Split patients
    all_patients = list(patient_to_indices.keys())
    train_ids, testval_ids = train_test_split(all_patients, test_size=0.3, random_state=SEED)
    val_ids, test_ids = train_test_split(testval_ids, test_size=0.5, random_state=SEED)

    def get_indices(pids):
        return [i for pid in pids for i in patient_to_indices[pid]]

    train_indices = get_indices(train_ids)
    val_indices = get_indices(val_ids)
    test_indices = get_indices(test_ids)

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    # Remap labels: pre = 0, post = 1
    original_map = full_dataset.class_to_idx
    label_remap = {original_map["pre"]: 0, original_map["post"]: 1}

    # Apply remapping to subsets
    train_dataset = RemappedImageFolder(train_subset, label_remap)
    val_dataset = RemappedImageFolder(val_subset, label_remap)
    test_dataset = RemappedImageFolder(test_subset, label_remap)

    # Inspect filenames and remapped labels
    print("Class-to-Index Mapping:", full_dataset.class_to_idx)
    for i in range(5):
        idx_in_full = train_indices[i]
        filename = os.path.basename(full_dataset.samples[idx_in_full][0])
        original_label = full_dataset.samples[idx_in_full][1]
        remapped = label_remap[original_label]
        print(f"{filename} — Remapped Label: {remapped}")

else:
    class ImageRegressionDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None):
            self.df = pd.read_csv(csv_file)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
            image = Image.open(img_path).convert("RGB")
            label = float(self.df.iloc[idx, 1])
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)

    full_dataset = ImageRegressionDataset(csv_file='labels.csv', img_dir='dataset', transform=transform)

    # Random split (regression doesn't need patient logic)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

# ----------- DataLoaders -----------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



# ---------------------------------------------


# ----------- Model Definitions -----------

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
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


# class ClassifierHead(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()
#         self.fc = nn.Linear(in_features, 1)

#         for param in self.fc.parameters():
#             param.requires_grad = True


#     def forward(self, x):
#         return self.fc(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.intermediate = nn.Linear(in_features, 4)  # New 6-neuron FC layer
        self.final = nn.Linear(4, 1)                   # Final output layer (binary classification)
        self.hidden_output = None                      # Store the 6-neuron activations

    def forward(self, x):
        x = self.intermediate(x)                       # [batch_size, 6]
        self.hidden_output = x                         # Store for extraction
        x = self.final(x)                              # [batch_size, 1]
        return x


class RegressorHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x)

# ---------------------------------------------


# ----------- Training & Validation Functions -----------

def train(model, loader, criterion, optimizer, task="classification"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_probs = [], []

    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        if task == "classification":
            labels = labels.float().view(-1)  # Ensure shape is [B]
        else:
            labels = labels.view(-1, 1)

        outputs = model(inputs).view(-1)  # Flatten output to [B]
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if task == "classification":
            probs = torch.sigmoid(outputs).view(-1)           # [B]
            preds = (probs >= 0.5).float()            # [B] in float
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)

    if task == "classification":
        accuracy = correct / total * 100
        auc = roc_auc_score(all_labels, all_probs)
        return avg_loss, accuracy, auc
    else:
        return avg_loss, None



@torch.no_grad()
def validate(model, loader, criterion, task="classification"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_probs, v_fpr, v_tpr, v_roc_auc = [], [], [], [], []

    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        if task == "classification":
            labels = labels.float().view(-1)  # Ensure shape is [B]
        else:
            labels = labels.view(-1, 1)

        outputs = model(inputs).view(-1)  # Flatten to [B] for BCE
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        if task == "classification":
            probs = torch.sigmoid(outputs).view(-1)           # [B]
            preds = (probs >= 0.5).float()            # Binary predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    if task == "classification":
        accuracy = correct / total * 100
        v_fpr, v_tpr, _ = roc_curve(all_labels, all_probs)
        v_roc_auc = auc(v_fpr, v_tpr)
        val_auc = roc_auc_score(all_labels, all_probs)
        return avg_loss, accuracy, val_auc, v_roc_auc, v_fpr, v_tpr
    else:
        return avg_loss, None, None, None, None
 
# ---------------------------------------------


# ----------- Test Function -----------
@torch.no_grad()
def test(model, loader, criterion, task="classification"):
    print("\n--- Testing on Test Set ---")
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_confidences = []
    all_filenames = []
    t_fpr, t_tpr, t_roc_auc = [], [], []
    all_fc6_outputs = []

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        if task == "classification":
            labels = labels.float().view(-1, 1)
        else:
            labels = labels.view(-1, 1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        if task == "classification":
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs >= 0.5).float()
            confidences = probs.detach().cpu().numpy()
            all_confidences.extend(confidences)
        else:
            preds = outputs.view(-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        fc6_batch = model[1].hidden_output.detach().cpu().numpy()
        all_fc6_outputs.extend(fc6_batch)    

        # Extract filenames (for RemappedImageFolder wrapping Subset of ImageFolder)
        try:
            remapped_dataset = loader.dataset  # RemappedImageFolder
            subset_dataset = remapped_dataset.dataset  # Subset
            base_dataset = subset_dataset.dataset  # ImageFolder

            if isinstance(subset_dataset, Subset):
                start = batch_idx * loader.batch_size
                end = start + len(inputs)
                batch_indices = subset_dataset.indices[start:end]
                batch_filenames = [base_dataset.samples[i][0] for i in batch_indices]
                all_filenames.extend(batch_filenames)

        except Exception as e:
            print("❗ Failed to extract filenames:", e)

    avg_loss = total_loss / len(loader)
    print(f"Test Loss: {avg_loss:.4f}")

    if task == "classification":
        report_dict = classification_report(all_labels, all_preds, target_names=["pre", "post"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f"resnet18_{condition}_bce_remap_classification_report_{timestamp}.csv", index=True)          
        print(classification_report(all_labels, all_preds, target_names=["pre", "post"]))
    else:
        print(f"MSE: {mean_squared_error(all_labels, all_preds):.4f}")
        print(f"R²: {r2_score(all_labels, all_preds):.4f}")

    # Save to CSV
    df = pd.DataFrame({
        "filename": all_filenames,
        "actual": all_labels,
        "predicted": all_preds
    })
    if task == "classification":
        df["confidence"] = all_confidences

    # Add FC6 output vectors to CSV
    fc6_df = pd.DataFrame(all_fc6_outputs, columns=[f"fc6_neuron_{i+1}" for i in range(4)])
    df = pd.concat([df, fc6_df], axis=1)

    df.to_csv(f"resnet18_{condition}_bce_remap_test_results_remap_{timestamp}.csv", index=False)
    print("✅ Test results saved to test_results.csv")

    if task == "classification":
        acc = accuracy_score(all_labels, all_preds) * 100
        try:
            test_auc = roc_auc_score(all_labels, all_confidences)
            t_fpr, t_tpr, _ = roc_curve(all_labels, all_confidences)
            t_roc_auc = auc(t_fpr, t_tpr)
        except ValueError:
            test_auc = None
            t_roc_auc, t_fpr, t_tpr = None, [], []
    else:
        acc = test_auc = None

    return all_preds, all_labels, avg_loss, acc, test_auc, t_roc_auc, t_fpr, t_tpr
# ---------------------------------------------


# ----------- Build Model -----------

if __name__ == "__main__":

    condition = 'lr' + str(LEARNING_RATE)


    feature_extractor = ResNetFeatureExtractor(pretrained=True).to(DEVICE)

    if CLASS_MODE:
        head = ClassifierHead(feature_extractor.features_dim).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        task_type = "classification"
    else:
        head = RegressorHead(feature_extractor.features_dim).to(DEVICE)
        criterion = nn.MSELoss()
        task_type = "regression"

    model = nn.Sequential(feature_extractor, head).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------

    # ----------- Run Training Loop -----------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics = {
        "epoch": [], "train_loss": [], "train_acc": [], "train_auc": [],
        "val_loss": [], "val_acc": [], "val_auc": [], "val_roc_auc":[], "val_fpr": [], "val_tpr":[], 
        "test_loss": [], "test_acc": [], "test_auc": [], "test_roc_auc":[], "test_fpr": [], "test_tpr":[]
    }


    for epoch in range(EPOCHS):
        train_loss, train_acc, train_auc = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc, val_roc_auc, val_fpr, val_tpr = validate(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["train_auc"].append(train_auc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["val_auc"].append(val_auc)
        metrics["val_roc_auc"].append(val_roc_auc)
        metrics["val_fpr"].append(val_fpr)
        metrics["val_tpr"].append(val_tpr)    
    

    # ---------------------------------------------

    # After training
    test_preds, test_labels, test_loss, test_acc, test_auc, test_roc_auc, test_fpr, test_tpr  = test(model, test_loader, criterion, task=task_type)

    # Extend to match length of training epochs
    for _ in range(len(metrics["epoch"])):
        metrics["test_loss"].append(test_loss)
        metrics["test_acc"].append(test_acc)
        metrics["test_auc"].append(test_auc)
        metrics["test_roc_auc"].append(test_roc_auc)
        metrics["test_fpr"].append(test_fpr)
        metrics["test_tpr"].append(test_tpr)      

    df_metrics = pd.DataFrame(metrics)

    df_metrics.to_csv(f"resnet18_{condition}_bce_remap_training_metrics_{timestamp}.csv", index=False)
    print("✅ Metrics saved to training_metrics.csv")

    save_path = f"resnet18_fc4_{condition}_{timestamp}.pt"
    torch.save(model.state_dict(), save_path)
    torch.save(model[0].state_dict(), "resnet18_fc4_{condition}_feature_extractor.pt")

    print(f"? Model saved to {save_path}")

# ----------- Inference Function -----------

@torch.no_grad()
def predict_image(model, image_path, transform, task="classification"):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # shape (1, C, H, W)

    output = model(input_tensor)

    # Get the 6-neuron FC output
    fc6_vector = model[1].hidden_output.detach().cpu().numpy().flatten()  # shape: (6,)    

    if task == "classification":
        prob = torch.sigmoid(output).view(-1)  # binary classification
        pred_class = (prob >= 0.5).int().item()
        confidence = prob.item()
        print(f"[Prediction] Class: {pred_class} | Confidence: {confidence:.2f}")
        print(f"[FC6 Vector]: {fc6_vector}")
        return pred_class, confidence, fc6_vector
    else:
        score = output.item()
        print(f"[Prediction] Score: {score:.3f}")
        print(f"[FC6 Vector]: {fc6_vector}")
        return score, fc6_vector

extract_path = "./images_for_feature_extraction"

extracted_features = pd.DataFrame(columns = ['filename', 'pred_class', 'confidence', 'f1', 'f2', 'f3', 'f4'])

for img in os.listdir(extract_path):
    img_path = os.path.join(extract_path, img)
    pred_class, confidence, fc6_vector = predict_image(model, img_path, transform, task=task_type)
    extracted_features.loc[len(extracted_features)] = [img, pred_class, confidence, fc6_vector[0], fc6_vector[1], fc6_vector[2], fc6_vector[3]]

extracted_features.to_csv("extracted_features_fc4.csv", index = False)


# ---------------------------------------------
