# explainer.py
import torch
import numpy as np
from captum.attr import IntegratedGradients, Saliency, Occlusion
from lime import lime_image
import shap
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries
import matplotlib.patches as mpatches

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activation = None

        # Hook for gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_image, class_idx=None):
        input_image = input_image.to(self.device)
        output = self.model(input_image)
        if class_idx is None:
            class_idx = output.argmax().item()

        # Zero the gradients and backward pass
        self.model.zero_grad()
        output[:, class_idx].backward()

        # Get weights of the gradients and compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        grad_cam = torch.sum(weights * self.activation, dim=1).squeeze(0).cpu().detach().numpy()

        # Apply ReLU and resize
        grad_cam = np.maximum(grad_cam, 0)
        grad_cam = cv2.resize(grad_cam, (input_image.shape[-1], input_image.shape[-2]))

        return grad_cam

# Plot Grad-CAM heatmap
def plot_grad_cam(image, grad_cam, title="Grad-CAM"):
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())  # Display the original image
    plt.imshow(grad_cam, cmap='jet', alpha=0.5)  # Overlay the Grad-CAM heatmap with transparency
    plt.colorbar(label="Importance")  # Add color bar to show values of the heatmap
    plt.title(title)
    plt.axis('off')
    plt.show()

# SHAP Implementation
def shap_explainer(model, background, input_image):
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(input_image)

    # Transpose and plot SHAP values
    shap_numpy = np.transpose(shap_values[0], (0, 2, 3, 1))
    test_numpy = np.transpose(input_image.cpu().numpy(), (0, 2, 3, 1))
    shap.image_plot(shap_numpy, test_numpy)

# Saliency Implementation
def generate_saliency(model, input_image, target_class=None):
    input_image = input_image.unsqueeze(0).to(model.device)
    input_image.requires_grad_()

    output = model(input_image)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[:, target_class].backward()

    saliency, _ = torch.max(input_image.grad.data.abs(), dim=1)
    return saliency[0].cpu().detach().numpy()

# Integrated Gradients Implementation
def plot_integrated_gradients(model, input_image, target_class=None):
    model.eval()
    ig = IntegratedGradients(model)

    input_image = input_image.unsqueeze(0).to(model.device)
    baseline = torch.zeros_like(input_image).to(model.device)

    if target_class is None:
        with torch.no_grad():
            output = model(input_image)
            target_class = output.argmax(dim=1).item()

    attributions = ig.attribute(input_image, baseline, target=target_class)
    attributions = attributions.squeeze(0).cpu().detach().numpy().sum(axis=0)

    plt.figure(figsize=(6, 6))
    img = plt.imshow(attributions, cmap='hot')
    cbar = plt.colorbar(img)
    cbar.set_label('Attribution Score', rotation=270, labelpad=20)
    plt.title(f"Integrated Gradients - Target Class {target_class}")
    plt.axis('off')
    plt.show()

# Occlusion Sensitivity Implementation
def plot_occlusion_sensitivity(model, input_image, target_class=0):
    model.eval()
    occlusion = Occlusion(model)

    input_image = input_image.unsqueeze(0).to(model.device)

    attributions = occlusion.attribute(input_image, sliding_window_shapes=(3, 30, 30), target=target_class)
    attributions = attributions.squeeze(0).cpu().detach().numpy().sum(axis=0)

    plt.figure(figsize=(6, 6))
    img = plt.imshow(attributions, cmap='hot')
    cbar = plt.colorbar(img)
    cbar.set_label('Attribution Score', rotation=270, labelpad=15)
    plt.title(f"Occlusion Sensitivity for Class {target_class}")
    plt.axis('off')
    plt.show()

# LIME Implementation
def lime_explainer(model, predict_fn, input_image):
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        input_image,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(temp, mask))

    green_patch = mpatches.Patch(color='green', label='Positive contribution')
    red_patch = mpatches.Patch(color='red', label='Negative contribution')
    plt.legend(handles=[green_patch, red_patch], loc='upper right')
    plt.title(f"LIME Explanation for Class {explanation.top_labels[0]}")
    plt.axis('off')
    plt.show()
