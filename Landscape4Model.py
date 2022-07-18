import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


modes = ['3D', 'contour', '2D']


class Landscape4Model():
    def __init__(self, model: nn.Module, loss: callable, mode='3D'):
        '''

        :param model:
        :param loss: given a model, return loss
        '''
        assert mode in modes
        self.mode = mode
        self.model = model
        self.loss = loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def synthesize_coordinates(self, x_min=-0.02, x_max=0.02, x_interval=1e-3,
                               y_min=-0.02, y_max=0.02, y_interval=1e-3):
        x = np.arange(x_min, x_max, x_interval)
        y = np.arange(y_min, y_max, y_interval)
        self.x, self.y = np.meshgrid(x, y)
        return self.x, self.y

    @torch.no_grad()
    def draw(self):
        self._find_direction()
        z = self._compute_for_draw()
        self.draw_figure(self.x, self.y, z)

    @torch.no_grad()
    def _find_direction(self):
        self.x0 = {}
        self.y0 = {}
        for name, param in self.model.named_parameters():
            self.x0[name] = torch.randn_like(param.data)
            self.y0[name] = torch.randn_like(param.data)

    @torch.no_grad()
    def _compute_for_draw(self):
        result = []
        if self.mode == '2D':
            self.x = self.x[0]
            for i in tqdm(range(self.x.shape[0])):
                now_x = self.x[i]
                loss = self._compute_loss_for_one_coordinate(now_x, 0)
                result.append(loss)
        else:
            for i in tqdm(range(self.x.shape[0])):
                for j in range(self.x.shape[1]):
                    now_x = self.x[i, j]
                    now_y = self.y[i, j]
                    loss = self._compute_loss_for_one_coordinate(now_x, now_y)
                    result.append(loss)

        result = np.array(result)
        result = result.reshape(self.x.shape)
        return result

    @torch.no_grad()
    def _compute_loss_for_one_coordinate(self, now_x: float, now_y: float):
        temp_model = copy.deepcopy(self.model)
        for name, param in temp_model.named_parameters():
            param.data = param.data + now_x * self.x0[name] + now_y * self.y0[name]

        return self.loss(temp_model)


    def draw_figure(self, mesh_x, mesh_y, mesh_z):
        if self.mode == '3D':
            figure = plt.figure()
            axes = Axes3D(figure)

            axes.plot_surface(mesh_x, mesh_y, mesh_z, cmap='rainbow')
            plt.show()

            plt.savefig(Landscape4Model.get_datetime_str() + ".png")

        if self.mode == '2D':
            plt.plot(mesh_x, mesh_z)
            plt.show()

            plt.savefig(Landscape4Model.get_datetime_str() + ".png")


    @staticmethod
    def get_datetime_str(style='dt'):
        import datetime
        cur_time = datetime.datetime.now()

        date_str = cur_time.strftime('%y_%m_%d_')
        time_str = cur_time.strftime('%H_%M_%S')

        if style == 'data':
            return date_str
        elif style == 'time':
            return time_str
        else:
            return date_str + time_str

if __name__ == '__main__':
    pass
