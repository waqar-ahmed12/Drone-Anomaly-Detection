import os
import torch
import torch.nn as nn
from torchvision import models


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x): return self.relu(self.conv(x) + x)

class ComplexMultiModalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_input = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.enc1  = nn.Sequential(nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
        self.res1  = ResidualBlock(64)
        self.enc2  = nn.Sequential(nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.res2  = ResidualBlock(128)
        self.enc3  = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.res3  = ResidualBlock(256)
        self.dec1     = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.res_dec1 = ResidualBlock(128)
        self.dec2     = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
        self.res_dec2 = ResidualBlock(64)
        self.dec3     = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, inplace=True))
        self.res_dec3 = ResidualBlock(32)
        self.final_conv = nn.Conv2d(32, 4, 3, padding=1)
    def forward(self, x):
        x = self.res3(self.enc3(self.res2(self.enc2(self.res1(self.enc1(self.encoder_input(x)))))))
        x = self.res_dec3(self.dec3(self.res_dec2(self.dec2(self.res_dec1(self.dec1(x))))))
        x = self.final_conv(x)
        return torch.cat([torch.sigmoid(x[:, 0:3]), torch.tanh(x[:, 3:4])], dim=1)

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level
BASE_DIR = os.path.dirname(current_dir)
# BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
# print(BASE_DIR)
CLASSIFIER_PATH = os.path.join(BASE_DIR, "Field patch extraction", "Classification")
CLASSIFIER_WEIGHTS = os.path.join(CLASSIFIER_PATH, "field_model_pytorch.pth")

AE_PATH = os.path.join(BASE_DIR, "Complete pipeline", "weights")
AE_WEIGHTS = os.path.join(AE_PATH, "multimodal_ae_epoch_15.pth")

CLASSIFIER_ONNX = os.path.join(current_dir, "ONNX", "field_model.onnx")
AE_ONNX = os.path.join(current_dir, "ONNX", "multimodal_ae.onnx")

def export_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}")


    print("\n1. For MobileNet")
    cls_model = models.mobilenet_v2()
    cls_model.classifier[1] = nn.Sequential(nn.Linear(cls_model.last_channel, 1), nn.Sigmoid())
    cls_model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=device))
    cls_model.to(device).eval()

    # MobileNet expects: Batch Size, 3 Channels (RGB), 224x224
    cls_dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    print("   to ONNX")
    torch.onnx.export(
        cls_model, cls_dummy_input, CLASSIFIER_ONNX,
        export_params=True, opset_version=11, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"    Saved to: {CLASSIFIER_ONNX}")

    print("\n2. for Custom Autoencoder")
    ae_model = ComplexMultiModalAutoencoder().to(device)
    state = torch.load(AE_WEIGHTS, map_location=device)
    if "model_state_dict" in state: state = state["model_state_dict"]
    ae_model.load_state_dict(state)
    ae_model.eval()

    ae_dummy_input = torch.randn(1, 4, 128, 128).to(device)
    
    print("   Autoencoder to ONNX...")
    torch.onnx.export(
        ae_model, ae_dummy_input, AE_ONNX,
        export_params=True, opset_version=11, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"   Saved to: {AE_ONNX}")

if __name__ == "__main__":
    # print(BASE_DIR)
    export_models()