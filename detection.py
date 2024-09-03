import torch
import warnings
import gradio as gr
import cv2
import torchvision
from torch import nn
from torchvision.models import mobilenet_v3_small
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")


def flip_text(x):
    return x[::-1]


def method1_prep(image):
    transforms = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    return transforms(image)


def method2_prep(image):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor()  # Transform to tensor at the end
    ])

    t_lower = 50
    t_upper = 150
    
    height, width = image.shape[:2]

    x = (width - 1920) // 2
    y = (height - 1080) // 2
    
    image = image[y:y+1080, x:x+1920]

    img = cv2.Canny(image, t_lower, t_upper)
    img = np.stack([img] * 3, axis=0)  # Convert to 3 channels
    img = img.transpose(1, 2, 0)  # HWC format for PIL conversion

    # Convert numpy array to PIL Image for transforms
    img_pil = Image.fromarray(img.astype(np.uint8))
    
    return transforms(img_pil)


def model1_inf(x):
    print("Method 1")

    image = method1_prep(x).unsqueeze(dim=0)
    model = mobilenet_v3_small(weights='DEFAULT')
    model.classifier[3] = nn.Linear(in_features=1024, out_features=2, bias=True)

    # Load weights with map_location
    model.load_state_dict(torch.load('./weights/method1(0.668).pt', map_location=device))

    model.eval()

    with torch.no_grad():
        model = model.to(device)
        image = image.to(device)
        output = torch.softmax(model(image), dim=1).detach().cpu()
        prediction = torch.argmax(output, dim=1).item()
        del model
        torch.cuda.empty_cache()  # Clear cache if CUDA is used
        if prediction == 0:
            return "The image is not pixelated"
        else:
            return "The image is pixelated"


def model2_inf(x):
    print("Method 2")

    image = method2_prep(x).unsqueeze(dim=0)
    model = mobilenet_v3_small(weights='DEFAULT')
    model.classifier[3] = nn.Linear(in_features=1024, out_features=2, bias=True)

    # Load weights with map_location
    model.load_state_dict(torch.load('./weights/method2(0.960).pt', map_location=device))
    print("\nModel weights loaded successfully")

    model.eval()

    with torch.no_grad():
        model = model.to(device)
        image = image.to(device)
        output = torch.softmax(model(image), dim=1).detach().cpu()
        prediction = torch.argmax(output, dim=1).item()
        del model
        torch.cuda.empty_cache()  # Clear cache if CUDA is used
        if prediction == 0:
            return "The image is not pixelated"
        else:
            return "The image is pixelated"


with gr.Blocks() as app:
    gr.Markdown("### Pixelation Detection App")
    method = gr.Radio(["Method 1", "Method 2 (Proposed Method)"], label="Select Method")

    with gr.Tab("Classification by image"):
        image_input = gr.Image(type="numpy", label="Upload an image")
        output_text = gr.Textbox(label="Output")

        def classify_image(img, method):
            if method == "Method 1":
                return model1_inf(img)
            else:
                return model2_inf(img)

        method.change(fn=classify_image, inputs=[image_input, method], outputs=output_text)
        image_input.change(fn=classify_image, inputs=[image_input, method], outputs=output_text)

app.launch(share=False)
