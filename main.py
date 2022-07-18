import torch
import torch.nn as nn

from data import get_test_loader, get_train_loader
from models.pyramidnet import pyramidnet272
from LossLandscape import LossLandscape


class PlotTrainer:
    def __init__(
        self,
        train_image_path,
        label2id_path,
        batch_size,
        num_workers,
        track_mode,
        img_size,
        device,
        x_interval,
        y_interval,
        pretrain_path,
    ):
        dataset = get_train_loader(
            train_image_path, label2id_path, batch_size, num_workers, track_mode, False, img_size
        ).dataset
        criterion = nn.CrossEntropyLoss()
        model = pyramidnet272(num_classes=60).to(device)
        state_dict = torch.load(pretrain_path)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
        self.losslandscape = LossLandscape(model, dataset, criterion, 1e-4, 0.01, "3D")
        self.device = torch.device("cuda")
        self.x_interval = x_interval
        self.y_interval = y_interval

    def __call__(self):
        self.losslandscape(x_interval=self.x_interval, y_interval=self.y_interval)


if __name__ == "__main__":
    plottrainer = PlotTrainer(
        train_image_path="/home/Bigdata/NICO/nico/train/",
        label2id_path="/home/Bigdata/NICO/dg_label_id_mapping.json",
        batch_size=48,
        num_workers=4,
        track_mode="track1",
        img_size=224,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        x_interval=11,
        y_interval=11,
        pretrain_path="./lib/model.pth",
    )
    plottrainer()
