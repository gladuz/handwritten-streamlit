import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import torch
import torchvision
import torch.nn.functional as F
import PIL.ImageOps


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = Net()
model.eval()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))

st.title("Handwritten digit recognition")

canvas_result = st_canvas(width=400, height=400, stroke_width=30)

if canvas_result.image_data is not None:
    
    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
    background = Image.new('RGBA', img.size, (255, 255, 255))
    alpha_composite = PIL.ImageOps.invert(Image.alpha_composite(background, img).resize((28, 28)).convert("L"))
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_norm = transforms(alpha_composite)
    input = (image_norm).unsqueeze(0)
    
    
    with torch.no_grad():
        
        pred = torch.topk(model(input), 3,  dim=1)

        preds = pred[1].squeeze(0)
        confs = (torch.exp(pred[0]) * 100).squeeze(0).long()
        
        st.header(f'Top 3 predictions are: {preds[0]}, {preds[1]}, {preds[2]}')
        st.header(f'Confidence scores: {confs[0]}%, {confs[1]}%, {confs[2]}%')