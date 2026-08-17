import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Industrial Surface Defect Detection",
    page_icon="🏭",
    layout="wide"
)


# ============================================================
# Paths
# ============================================================

MODEL_PATH = "resnet50.keras"


# ============================================================
# Class Names
# ============================================================

class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# ============================================================
# Grad-CAM Function
# ============================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None
):

    # ResNet50 backbone is the second layer
    base_model = model.layers[1]

    # Get the required convolutional layer
    last_conv_layer = base_model.get_layer(
        last_conv_layer_name
    )

    # Model used for Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    # Calculate gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array,
            training=False
        )

        if pred_index is None:

            pred_index = tf.argmax(
                predictions[0]
            )

        class_channel = predictions[:, pred_index]

    # Gradient of target class
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # Average gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weighted combination of feature maps
    heatmap = (
        conv_outputs
        @ pooled_grads[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap
    )

    # Keep only positive activations
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize between 0 and 1
    heatmap /= (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


# ============================================================
# Create Grad-CAM Overlay
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap
):

    # Resize heatmap to image size
    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (
            original_image.shape[0],
            original_image.shape[1]
        )
    ).numpy()

    heatmap = np.squeeze(
        heatmap
    )

    # Plot overlay
    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.imshow(
        original_image
    )

    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    ax.axis("off")

    plt.tight_layout()

    return fig


# ============================================================
# Title
# ============================================================

st.title(
    "🏭 Industrial Surface Defect Detection"
)

st.write(
    "Upload a steel surface image to classify the defect "
    "using a ResNet50 model with Grad-CAM explainability."
)


# ============================================================
# Upload Image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    # Load original image
    original_image = image.load_img(
        uploaded_file
    )

    original_image = image.img_to_array(
        original_image
    )

    # Keep original for display
    display_image = original_image.astype(
        np.uint8
    )

    # Resize for model
    model_image = image.array_to_img(
        original_image
    )

    model_image = model_image.resize(
        (224, 224)
    )

    model_array = image.img_to_array(
        model_image
    )

    # Add batch dimension
    model_array = np.expand_dims(
        model_array,
        axis=0
    )

    # Same preprocessing used during training
    model_array = preprocess_input(
        model_array
    )

    # Prediction
    predictions = model.predict(
        model_array,
        verbose=0
    )

    predicted_class = np.argmax(
        predictions[0]
    )

    confidence = predictions[0][
        predicted_class
    ]

    predicted_label = class_names[
        predicted_class
    ]


    # ========================================================
    # Display Prediction
    # ========================================================

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            display_image,
            use_container_width=True
        )

    with col2:

        st.subheader("Prediction")

        st.metric(
            "Defect",
            predicted_label.replace(
                "_",
                " "
            ).title()
        )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        # Probability for every class
        st.write("Class Probabilities")

        for i, class_name in enumerate(
            class_names
        ):

            st.progress(
                float(
                    predictions[0][i]
                ),
                text=(
                    f"{class_name.replace('_', ' ').title()} "
                    f"— {predictions[0][i]:.2%}"
                )
            )


    # ========================================================
    # Grad-CAM
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔎 Grad-CAM Explainability"
    )

    last_conv_layer_name = (
        "conv5_block3_out"
    )

    heatmap = make_gradcam_heatmap(
        model_array,
        model,
        last_conv_layer_name,
        predicted_class
    )

    gradcam_fig = create_gradcam_overlay(
        display_image,
        heatmap
    )

    st.pyplot(
        gradcam_fig
    )

    st.caption(
        "Highlighted regions show the areas that contributed "
        "most strongly to the model's prediction."
    )


    # ========================================================
    # Prediction Message
    # ========================================================

    st.markdown("---")

    if confidence >= 0.90:

        st.success(
            f"High-confidence prediction: "
            f"{predicted_label.replace('_', ' ').title()}"
        )

    elif confidence >= 0.60:

        st.warning(
            f"Moderate-confidence prediction: "
            f"{predicted_label.replace('_', ' ').title()}"
        )

    else:

        st.error(
            f"Low-confidence prediction: "
            f"{predicted_label.replace('_', ' ').title()}"
        )