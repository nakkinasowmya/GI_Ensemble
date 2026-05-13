# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import cv2
# from PIL import Image

# def get_last_conv_layer(model):
#     for layer in reversed(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             return layer.name
#     raise ValueError("No convolutional layer found.")

# def generate_gradcam_inception(model, img_path, preprocess_fn, target_size, output_path):
#     # Load and preprocess image
#     img = Image.open(img_path).convert('RGB').resize(target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_fn(img_array)

#     preds = model.predict(img_array)
#     class_idx = np.argmax(preds[0])

#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(get_last_conv_layer(model)).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, class_idx]

#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     heatmap = heatmap.numpy()

#     # Read original image for overlay
#     original_img = cv2.imread(img_path)
#     heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
#     heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

#     superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
#     cv2.imwrite(output_path, superimposed_img)
#     return output_path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image


# -------------------------------------
# Find last convolution layer
# -------------------------------------
def get_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):

            return layer.name

    raise ValueError("No convolutional layer found in the model.")


# -------------------------------------
# Generate GradCAM heatmap
# -------------------------------------
def generate_gradcam_inception(model, img_path, preprocess_fn, target_size, output_path):

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB").resize(target_size)

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_fn(img_array)


    # Predict
    preds = model.predict(img_array)

    # Fix: convert list output to array
    if isinstance(preds, list):
        preds = preds[0]

    preds = np.array(preds)

    class_idx = int(np.argmax(preds[0]))


    # Create GradCAM model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(get_last_conv_layer(model)).output,
            model.output
        ]
    )


    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        # Fix if predictions returned as list
        if isinstance(predictions, list):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)

        num_classes = predictions.shape[-1]

        # Prevent out-of-bounds index
        if class_idx >= num_classes:
            class_idx = num_classes - 1

        loss = predictions[:, class_idx]


    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]


    # Create heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

    heatmap = heatmap.numpy()


    # -------------------------------------
    # Overlay heatmap on original image
    # -------------------------------------
    original_img = cv2.imread(img_path)

    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        original_img,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    cv2.imwrite(output_path, superimposed_img)

    return output_path