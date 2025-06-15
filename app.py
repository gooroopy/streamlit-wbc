import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import load_model
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch.nn.functional as F_torch # Renamed to avoid conflict with `functional as F`

# --- Functions (from your original script, slightly modified) ---

# This needs to be defined if it's used in get_object_detection_model
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2

# Define fastrcnn_loss_modified (it's used by get_object_detection_model)
# You need to ensure torch is imported as well
def fastrcnn_loss_modified(class_logits, box_regression, labels, regression_targets):
    # type: (torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]
    """
    Computes the loss for Faster R-CNN.
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    w = torch.Tensor([1.0, 1.0, 24.0]).to(class_logits.device) # Move weight to device
    classification_loss = F_torch.cross_entropy(class_logits, labels, w)

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F_torch.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def get_object_detection_model(num_classes = 3,
                               feature_extraction = True):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    if feature_extraction == True:
        for p in model.parameters():
            p.requires_grad = False
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats,
                                                   num_classes)
    model.roi_heads.fastrcnn_loss = fastrcnn_loss_modified
    return model

IMAGE_SIZE = 224
CLASS_NAMES = {1: "rbc", 2: "wbc", 0: "background"}
wbc_class_labels ={0: 'Basophil', 1: 'Eosinophil', 2: 'Lymphocyte', 3: 'Monocyte', 4: 'Neutrophil'}

# Function to preprocess the input image
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image_tensor = F.to_tensor(image)
    return image, image_tensor.unsqueeze(0)

# Load the trained model
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_trained_model(model_path, num_classes, device):
    model = get_object_detection_model(num_classes=num_classes, feature_extraction=False)
    map_location = torch.device('cpu') if device == 'cpu' else None
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.to(device)
    model.eval()
    return model

@st.cache_resource # Cache the model
def load_wbc_classification_model(model_path):
    return load_model(model_path)


# Perform prediction
def predict_image(model, image_tensor, device, confidence_threshold=0.5):
    with torch.no_grad():
        predictions = model(image_tensor.to(device))
    return predictions[0]

# Plot the results
def plot_predictions(image, predictions, wbc_model_local, confidence_threshold=0.5, margin=0.1):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    img_width, img_height = image.size
    stats = {'RBC': 0, 'WBC:Basophil': 0, 'WBC:Eosinophil': 0, 'WBC:Lymphocyte': 0, 'WBC:Monocyte': 0, 'WBC:Neutrophil': 0, 'UNK': 0}

    for box, score, label in zip(
        predictions['boxes'], predictions['scores'], predictions['labels']
    ):
        if score >= confidence_threshold:
            class_name = CLASS_NAMES.get(label.item(), "unknown")
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            width, height = xmax - xmin, ymax - ymin

            xmin = max(0, xmin - margin * width)
            ymin = max(0, ymin - margin * height)
            xmax = min(img_width, xmax + margin * width)
            ymax = min(img_height, ymax + margin * height)

            if class_name == 'wbc':
                cropped_image = image.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                cropped_image_np = np.array(cropped_image)
                cropped_image_resized = cv2.resize(cropped_image_np, (IMAGE_SIZE, IMAGE_SIZE))
                cropped_image_array = cropped_image_resized / 255.0
                cropped_image_array = np.expand_dims(cropped_image_array, axis=0)

                predictions_wbc = wbc_model_local.predict(cropped_image_array, verbose=0) # Added verbose=0 to suppress Keras output
                predicted_class = np.argmax(predictions_wbc)
                predicted_label = wbc_class_labels.get(predicted_class, "unknown")
                subclass_score = predictions_wbc[0][predicted_class]
                stats[f'WBC:{predicted_label}'] += 1
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(
                    xmin, ymin - 10,
                    f"{class_name}: {predicted_label} ({subclass_score:.2f})",
                    color='red',
                    fontsize=12,
                    bbox=dict(facecolor='yellow', alpha=0.5)
                )
            else:
                if class_name == 'rbc':
                    stats['RBC'] += 1
                else:
                    stats['UNK'] += 1
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(
                    xmin, ymin - 10,
                    f"{class_name}: {score:.2f}",
                    color='red',
                    fontsize=12,
                    bbox=dict(facecolor='yellow', alpha=0.5)
                )
    return fig, stats

# --- Streamlit App ---

st.title("Blood Cell Classification by AI model")

st.write("Upload an image to classify blood cells.")

# Load models (adjust paths as needed)
WBC_MODEL_PATH = "models/resnet50_wbc_model.h5" # Assuming 'models' directory is in the same place as your script
BCD_MODEL_PATH = "models/faster_rcnn_bcd_model.pth"

try:
    wbc_model = load_wbc_classification_model(WBC_MODEL_PATH)
    st.success("White Blood Cell Classification Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading WBC Classification Model: {e}")
    st.info(f"Please ensure '{WBC_MODEL_PATH}' exists in the correct location.")
    st.stop() # Stop the app if model fails to load

device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    bcd_model = load_trained_model(BCD_MODEL_PATH, num_classes=3, device=device)
    st.success("Blood Cell Detection Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading Blood Cell Detection Model: {e}")
    st.info(f"Please ensure '{BCD_MODEL_PATH}' exists in the correct location.")
    st.stop() # Stop the app if model fails to load


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image, image_tensor = preprocess_image(uploaded_file)

    # Perform inference
    predictions = predict_image(bcd_model, image_tensor, device)

    # Plot the predictions
    fig, stats = plot_predictions(image, predictions, wbc_model, confidence_threshold=0.5)

    st.pyplot(fig)
    st.write("Detection Statistics:")
    st.json(stats)
