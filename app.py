import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNN_BT  # Assuming CNN_BT is in model.py

# Define the parameters needed for the CNN_BT model
params = {
    "shape_in": (3, 256, 256),  # Input shape: (channels, height, width)
    "initial_filters": 8,        # Initial number of filters
    "num_fc1": 100,              # Number of neurons in the first fully connected layer
    "dropout_rate": 0.25,        # Dropout rate for regularization
    "num_classes": 2             # Number of output classes
}

# Initialize the model
model = CNN_BT(params)

# Move the model to the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define image transformations (resize the image to the model's expected input size and convert to tensor)
transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
   ]
)

# Function to make predictions
def predict(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert grayscale images to RGB
    
    image = transform(image).unsqueeze(0)  # Transform and add batch dimension
    image = image.to(device)  # Move the image to the same device as the model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation (not needed for inference)
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)  # Get the index of the highest score
    return predicted.item()

# Define a mapping from numeric label to human-readable label
label_mapping = {0: "Brain Tumor", 1: "Healthy"}

# Streamlit app
st.title("Brain Tumor Classification")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image of a brain scan...", type=["jpg", "png", "jpeg","tiff"])

# Add a submit button for classification
if st.button('Submit for Classification'):
    if uploaded_file is not None:
        # Open the uploaded image file
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Brain Scan', use_column_width=True)

        # Classify the image using the model
        st.write("Classifying...")
        label_index = predict(image)  # Get the predicted index (0 or 1)

        # Get the label name based on the predicted index
        label_name = label_mapping[label_index]

        # Display the predicted label name
        st.write(f"Predicted Label: {label_name}")
    else:
        st.write("Please upload an image first.")
