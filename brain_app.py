# brain_app.py

import streamlit as st
import torch
from PIL import Image
from explainer import (
    GradCAM, plot_grad_cam, shap_explainer, generate_saliency, plot_integrated_gradients,
    plot_occlusion_sensitivity, lime_explainer
)
from model import CNN_BT
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the parameters for the CNN_BT model
params = {
    "shape_in": (3, 256, 256),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2
}

# Initialize the model and move it to the appropriate device (GPU/CPU)
model = CNN_BT(params)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to prepare image for model input
def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)

# Prediction function for LIME and other explainability methods
def predict_fn(image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

# Label mapping for the classification task
label_mapping = {0: "Brain Tumor", 1: "Healthy"}

# Streamlit app layout
st.title("Brain Tumor Classification with Explainability Techniques")

# Create session states to store the output images and explanations
if 'output_images' not in st.session_state:
    st.session_state['output_images'] = []

# Image uploader widget
uploaded_file = st.file_uploader("Upload a brain scan image...", type=["jpg", "png", "jpeg", "tiff"])

# Button to clear the outputs
if st.button('Clear'):
    st.session_state['output_images'] = []

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Brain Scan', use_column_width=True)
    image_tensor = prepare_image(image)

    # Display prediction button
    if st.button('Predict'):
        label_index = predict_fn(image_tensor).argmax()
        label_name = label_mapping[label_index]
        st.write(f"Predicted Label: {label_name}")

    # Subheader for explainability techniques
    st.subheader("Explainability Techniques")

    # Create columns for organizing buttons
    col1, col2, col3 = st.columns(3)

    # Grad-CAM Button
    with col1:
        if st.button("Grad-CAM"):
            grad_cam = GradCAM(model, model.conv4, device)  # Update for Grad-CAM
            grad_cam_output = grad_cam(image_tensor)
            fig, ax = plt.subplots()
            plot_grad_cam(image_tensor.squeeze(0), grad_cam_output)
            st.session_state['output_images'].append(fig)

    # SHAP Button
    with col2:
        if st.button("SHAP"):
            background = image_tensor
            fig, ax = plt.subplots()
            shap_explainer(model, background, image_tensor)
            st.session_state['output_images'].append(fig)

    # LIME Button
    with col3:
        if st.button("LIME"):
            fig, ax = plt.subplots()
            lime_explainer(model, predict_fn, image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
            st.session_state['output_images'].append(fig)

    # Create more columns for additional techniques
    col4, col5, col6 = st.columns(3)

    # Salience Map Button
    with col4:
        if st.button("Salience Map"):
            saliency_map = generate_saliency(model, image_tensor.squeeze(0))
            fig, ax = plt.subplots()
            ax.imshow(saliency_map, cmap='hot')
            ax.axis('off')
            st.pyplot(fig)
            st.session_state['output_images'].append(fig)

    # Occlusion Sensitivity Button
    with col5:
        if st.button("Occlusion"):
            fig, ax = plt.subplots()
            plot_occlusion_sensitivity(model, image_tensor.squeeze(0))
            st.session_state['output_images'].append(fig)

    # Integrated Gradients Button
    with col6:
        if st.button("Integrated Gradients"):
            fig, ax = plt.subplots()
            plot_integrated_gradients(model, image_tensor.squeeze(0))
            st.session_state['output_images'].append(fig)

    # Display all saved outputs (from session state)
    st.subheader("Generated Explanations")
    for fig in st.session_state['output_images']:
        st.pyplot(fig)

else:
    st.write("Please upload an image first.")
