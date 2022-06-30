from torchvision.models import resnet18
from skimage.io import imread
import torch
from D2Landscape import D2Landscape
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(pretrained=True).to(device)
image = imread('image.jpg') / 255
image = torch.tensor(image, dtype=torch.float32)
image = image.permute(2, 0, 1)
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
image = norm(image)
image = image.unsqueeze(0)
pre = model(image.to(device))
_, category = torch.max(pre, dim=1)
criterion = torch.nn.CrossEntropyLoss().to(device)

x = image.clone().to(device)
x.requires_grad = True
label = torch.tensor([category], device=device)
optimizer = torch.optim.Adam([x], lr = 1e-3, maximize = True)
for i in range(100):
    pre = model(x)
    loss = criterion(pre, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, loss.item())

a = D2Landscape(lambda x: criterion(model(x), label), x.detach())
a.synthesize_coordinates()
a.draw()
