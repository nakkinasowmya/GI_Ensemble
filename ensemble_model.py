# import numpy as np
# import json
# import os
# import cv2
# from tensorflow.keras.models import load_model
# from PIL import Image

# from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
# from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
# from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# import tensorflow as tf
# import matplotlib.cm as cm


# # Load ensemble weights and class labels
# with open('ensemble_weights.json', 'r') as f:
#     weights = json.load(f)

# total_weight = sum(weights.values())

# with open('labels.txt', 'r') as f:
#     class_labels = [line.strip() for line in f]

# # Model file paths
# model_paths = {
#     "vgg16": r"C:\Users\pooja\Downloads\GI_Ensemble_Flask_App\GI_Ensemble_Flask_App\models\vgg16_finetuned_best.h5",
#     "inceptionv3": r"C:\Users\pooja\Downloads\GI_Ensemble_Flask_App\GI_Ensemble_Flask_App\models\fine_tuned_inceptionv3_model.h5",
#     "resnet50": r"C:\Users\pooja\Downloads\GI_Ensemble_Flask_App\GI_Ensemble_Flask_App\models\fine_tuned_resnet_model (1).h5",
#     "mobilenet": r"C:\Users\pooja\Downloads\GI_Ensemble_Flask_App\GI_Ensemble_Flask_App\models\best_mobilenetv2_finetuned.h5"
# }

# # Load models
# models = {}
# for name, path in model_paths.items():
#     if os.path.exists(path):
#         models[name] = load_model(path)
#     else:
#         raise FileNotFoundError(f"Model file not found: {path}")

# def find_last_conv_layer(model):
#     # Iterate layers in reverse to find last conv2d layer
#     for layer in reversed(model.layers):
#         if 'conv' in layer.name.lower() and 'conv2d' in layer.__class__.__name__.lower():
#             return layer.name
#     # Fallback for other conv layers, if needed
#     for layer in reversed(model.layers):
#         if 'conv' in layer.name.lower():
#             return layer.name
#     raise ValueError("No convolutional layer found in the model.")

# def preprocess_image(img_path, target_size):
#     img = Image.open(img_path).convert('RGB').resize(target_size)
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# def generate_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )
    
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(predictions[0])
#         class_channel = predictions[:, pred_index]
    
#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
    
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
#     heatmap = heatmap.numpy()
    
#     return heatmap

# def predict_image(img_path):
#     ensemble_probs = np.zeros(len(class_labels))
#     heatmaps = []
    
#     # Map model names to their preprocessing, input size, and last conv layer name
#     model_info = {
#         "vgg16": {
#             "preprocess": vgg_preprocess,
#             "input_size": (224, 224),
#             "last_conv_layer": "block5_conv3"
#         },
#         "inceptionv3": {
#             "preprocess": inception_preprocess,
#             "input_size": (299, 299),
#             "last_conv_layer": "conv2d_94"  # corrected based on your existing layers
#         },
#         "resnet50": {
#             "preprocess": resnet_preprocess,
#             "input_size": (224, 224),
#             "last_conv_layer": "conv5_block3_out"
#         },
#         "mobilenet": {
#             "preprocess": mobilenet_preprocess,
#             "input_size": (224, 224),
#             "last_conv_layer": "Conv_1"
#         }
#     }
    
#     for name, model in models.items():
#         info = model_info[name]
#         img_array = preprocess_image(img_path, target_size=info["input_size"])
#         img_array = info["preprocess"](img_array.astype(np.float32))
        
#         prob = model.predict(img_array)[0]
#         ensemble_probs += weights[name] * prob
        
#         # Generate Grad-CAM heatmap for predicted class index
#         pred_index = np.argmax(prob)
#         hm = generate_gradcam(model, img_array, info["last_conv_layer"], pred_index)
#         heatmaps.append(hm)
    
#     # Normalize ensemble probs
#     ensemble_probs /= total_weight
#     predicted_index = np.argmax(ensemble_probs)
#     predicted_label = class_labels[predicted_index]
    
#     # Resize all heatmaps to the max shape for combining
#     max_h = max(h.shape[0] for h in heatmaps)
#     max_w = max(h.shape[1] for h in heatmaps)
    
#     heatmaps_resized = [cv2.resize(hm, (max_w, max_h), interpolation=cv2.INTER_LINEAR) for hm in heatmaps]
    
#     combined_heatmap = np.zeros((max_h, max_w))
#     for hm in heatmaps_resized:
#         combined_heatmap += hm
    
#     # Normalize combined heatmap between 0 and 1
#     combined_heatmap = combined_heatmap / np.max(combined_heatmap)
    
#     # Convert heatmap to RGB image
#     import matplotlib.cm as cm
#     heatmap_img = cm.jet(combined_heatmap)[:, :, :3]  # Remove alpha channel
#     heatmap_img = np.uint8(heatmap_img * 255)
    
#     # Load original image to overlay heatmap on it
#     original_img = Image.open(img_path).resize((max_w, max_h)).convert('RGB')
#     original_img_np = np.array(original_img)
    
#     # Overlay heatmap with some transparency
#     overlayed_img = cv2.addWeighted(original_img_np, 0.6, heatmap_img, 0.4, 0)
    
#     # Resize overlayed image to double the size
#     new_width = max_w * 2
#     new_height = max_h * 2
#     overlayed_img_resized = cv2.resize(overlayed_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
#     # Save overlayed heatmap image (resized)
#     heatmap_path = os.path.splitext(img_path)[0] + '_combined_gradcam_large.jpg'
#     cv2.imwrite(heatmap_path, cv2.cvtColor(overlayed_img_resized, cv2.COLOR_RGB2BGR))
    
#     return predicted_label, ensemble_probs, heatmap_path



import numpy as np
import json
import os
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

import tensorflow as tf
import matplotlib.cm as cm


# -------------------------------
# Load ensemble weights
# -------------------------------
with open("ensemble_weights.json", "r") as f:
    weights = json.load(f)

total_weight = sum(weights.values())


# -------------------------------
# Load class labels
# -------------------------------
with open("labels.txt", "r") as f:
    class_labels = [line.strip() for line in f]


# -------------------------------
# Model paths
# -------------------------------
model_paths = {
    "vgg16": r"C:\prj review 2k26\GI_Ensemble_Flask_App\GI_Ensemble_Flask_App\models\vgg16_finetuned_best.h5",
    "inceptionv3": r"C:\prj review 2k26\GI_Ensemble_Flask_App\GI_Ensemble_Flask_App\models\fine_tuned_inceptionv3_model.h5",
    "resnet50": r"C:\prj review 2k26\GI_Ensemble_Flask_App\GI_Ensemble_Flask_App\models\fine_tuned_resnet_model (1).h5",
    "mobilenet": r"C:\prj review 2k26\GI_Ensemble_Flask_App\GI_Ensemble_Flask_App\models\best_mobilenetv2_finetuned.h5",
}


# -------------------------------
# Load models
# -------------------------------
models = {}

for name, path in model_paths.items():

    if os.path.exists(path):

        print(f"Loading {name} model...")

        models[name] = load_model(path)

    else:

        raise FileNotFoundError(f"Model file not found: {path}")


# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(img_path, target_size):

    img = Image.open(img_path).convert("RGB").resize(target_size)

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# -------------------------------
# Safe GradCAM generator
# -------------------------------
def generate_gradcam(model, img_array, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        # Fix: if predictions is list convert it
        if isinstance(predictions, list):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        pred_index = int(pred_index)

        num_classes = predictions.shape[-1]

        if pred_index >= num_classes:
            pred_index = num_classes - 1

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


# -------------------------------
# Main prediction function
# -------------------------------
def predict_image(img_path):

    ensemble_probs = np.zeros(len(class_labels))

    heatmaps = []

    model_info = {

        "vgg16": {
            "preprocess": vgg_preprocess,
            "input_size": (224, 224),
            "last_conv_layer": "block5_conv3",
        },

        "inceptionv3": {
            "preprocess": inception_preprocess,
            "input_size": (299, 299),
            "last_conv_layer": "conv2d_94",
        },

        "resnet50": {
            "preprocess": resnet_preprocess,
            "input_size": (224, 224),
            "last_conv_layer": "conv5_block3_out",
        },

        "mobilenet": {
            "preprocess": mobilenet_preprocess,
            "input_size": (224, 224),
            "last_conv_layer": "Conv_1",
        },
    }

    for name, model in models.items():

        info = model_info[name]

        img_array = preprocess_image(img_path, info["input_size"])

        img_array = info["preprocess"](img_array.astype(np.float32))

        prob = model.predict(img_array)[0]

        ensemble_probs += weights[name] * prob

        pred_index = int(np.argmax(prob))

        heatmap = generate_gradcam(
            model,
            img_array,
            info["last_conv_layer"],
            pred_index,
        )

        heatmaps.append(heatmap)

    # Normalize ensemble probabilities
    ensemble_probs /= total_weight

    predicted_index = int(np.argmax(ensemble_probs))

    predicted_label = class_labels[predicted_index]

    # -------------------------------
    # Combine heatmaps
    # -------------------------------
    max_h = max(h.shape[0] for h in heatmaps)
    max_w = max(h.shape[1] for h in heatmaps)

    heatmaps_resized = [
        cv2.resize(hm, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
        for hm in heatmaps
    ]

    combined_heatmap = np.zeros((max_h, max_w))

    for hm in heatmaps_resized:
        combined_heatmap += hm

    combined_heatmap = combined_heatmap / np.max(combined_heatmap)

    # Convert heatmap to RGB
    heatmap_img = cm.jet(combined_heatmap)[:, :, :3]

    heatmap_img = np.uint8(heatmap_img * 255)

    # Load original image
    original_img = Image.open(img_path).resize((max_w, max_h)).convert("RGB")

    original_img_np = np.array(original_img)

    overlayed_img = cv2.addWeighted(
        original_img_np,
        0.6,
        heatmap_img,
        0.4,
        0,
    )

    # Enlarge output
    new_width = max_w * 2
    new_height = max_h * 2

    overlayed_img_resized = cv2.resize(
        overlayed_img,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_path = os.path.splitext(img_path)[0] + "_combined_gradcam.jpg"

    cv2.imwrite(
        heatmap_path,
        cv2.cvtColor(overlayed_img_resized, cv2.COLOR_RGB2BGR),
    )

    return predicted_label, ensemble_probs, heatmap_path