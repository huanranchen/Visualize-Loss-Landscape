from torchvision.models import resnet18
from skimage.io import imread
import torch
from D2Landscape import D2Landscape
from torchvision import transforms

model = resnet18(pretrained=True)
image = imread('image.jpg') / 255
image = torch.tensor(image, dtype=torch.float32)
image = image.permute(2, 0, 1)
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
image = norm(image)
image = image.unsqueeze(0)
pre = model(image)
_, category = torch.max(pre, dim=1)
criterion = torch.nn.CrossEntropyLoss

a = D2Landscape(lambda x: criterion(x, torch.tensor([category])), image)
a.synthesize_coordinates()
a.draw()
