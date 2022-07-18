import torch
import torch.nn as nn
from Landscape4Model import Landscape4Model
from data.data import get_loader, get_test_loader
from pyramidnet import pyramidnet272


class GetLoss():
    def __init__(self, batch_size=96):
        train_image_path = './public_dg_0416/train/'
        valid_image_path = './public_dg_0416/train/'
        label2id_path = './dg_label_id_mapping.json'
        test_image_path = './public_dg_0416/public_test_flat/'
        self.train_loader = get_loader(batch_size=batch_size,
                                       valid_category=None,
                                       train_image_path=train_image_path,
                                       valid_image_path=valid_image_path,
                                       label2id_path=label2id_path)
        self.test_loader_predict, _ = get_test_loader(batch_size=batch_size,
                                                      transforms=None,
                                                      label2id_path=label2id_path,
                                                      test_image_path=test_image_path)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda')

    @torch.no_grad()
    def __call__(self, model, estimate_times=5):
        result = 0
        for step, (x, y) in enumerate(self.train_loader):
            x = x.to(device)
            y = y.to(device)
            pre = model(x)
            result += self.criterion(pre, y).item()

            if step + 1 >= estimate_times:
                return result / (step + 1)


device = torch.device('cuda')
model = pyramidnet272(num_classes=60).to(device)
model.load_state_dict(torch.load('model.pth'))
w = Landscape4Model(model, GetLoss())
w.synthesize_coordinates()
w.draw()
