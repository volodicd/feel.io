import torch
import cv2
import numpy as np
from tkinter import Tk, filedialog, Button, Label, Canvas, PhotoImage
from torchvision import transforms
from src.models.model import ImprovedEmotionModel
from PIL import Image, ImageTk

def preprocess_image(image_path):
    """Preprocesses the input image to fit the model's input format."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_emotion(model, image_tensor):
    """Runs inference on the input image and returns the predicted emotion."""
    with torch.no_grad():
        output = model(image=image_tensor)
        if 'image_pred' in output:
            # Add debugging prints
            logits = output['image_pred']
            probs = torch.softmax(logits, dim=1)
            print("Raw logits:", logits)
            print("Probabilities:", probs)
            emotion = torch.argmax(output['image_pred']).item()
            return emotion
    return 'Unknown'


def load_image():
    """Opens a file dialog to select an image and processes it for emotion detection."""
    global img_label, canvas, model
    try:
        # Explicitly specify file types
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if file_path:  # Ensure a valid file path is selected
            # Display the selected image
            image = Image.open(file_path)
            image_resized = image.resize((300, 300))
            img = ImageTk.PhotoImage(image_resized)
            canvas.create_image(0, 0, anchor='nw', image=img)
            canvas.image = img

            # Preprocess and predict
            image_tensor = preprocess_image(file_path)
            emotion = predict_emotion(model, image_tensor)
            emotion_label.config(text=f"Predicted Emotion: {emotion}")
    except Exception as e:
        print(f"Error loading image: {e}")
        emotion_label.config(text="Error: Failed to load image")


def create_gui():
    """Creates a simple GUI for uploading an image and displaying the result."""
    global img_label, canvas, emotion_label

    root = Tk()
    root.title("Emotion Detection from Image")

    # Buttons and labels
    upload_btn = Button(root, text="Upload Image", command=load_image, padx=10, pady=5)
    upload_btn.pack(pady=10)

    # Canvas to display the image
    canvas = Canvas(root, width=300, height=300, bg="white")
    canvas.pack(pady=10)

    # Label to display predicted emotion
    emotion_label = Label(root, text="Predicted Emotion: None", font=("Arial", 14))
    emotion_label.pack(pady=10)

    root.mainloop()

def main():
    global model
# Load the trained model
    model = ImprovedEmotionModel(num_emotions=7)  # Adjust num_emotions as needed
    checkpoint = torch.load('models/checkpoints/checkpoint_epoch_0_acc_1.000.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model successfully loaded.")

    # Start the GUI
    create_gui()

if __name__ == "__main__":
    main()
