import torch
import numpy as np
from torch import nn
from tqdm import tqdm

class D2Landscape():
    def __init__(self, model, input: torch.tensor):
        '''

        :param model: taken input as input, output loss
        :param input:
        '''
        self.model = model
        self.input = input

    def synthesize_coordinates(self, x_min=-5, x_max=5, x_interval=0.25,
                               y_min=-5, y_max=5, y_interval=0.25):
        x = np.arange(x_min, x_max, x_interval)
        y = np.arange(y_min, y_max, y_interval)
        self.x, self.y = np.meshgrid(x, y)

    def draw(self):
        self._find_direction()
        z = self._compute_for_draw()
        self.draw3D(self.x, self.y, z)

    def _find_direction(self):
        self.x0 = torch.randn(self.input.shape)
        self.y0 = torch.randn(self.input.shape)
        self.x0 /= torch.norm(self.x0, p=2)
        self.y0 /= torch.norm(self.y0, p=2)

        # keep perpendicular
        if torch.abs(self.x0.reshape(-1) @ self.y0.reshape(-1)) >= 0.1:
            self._find_direction()

    def _compute_for_draw(self):
        result = []
        for i in tqdm(range(self.x.shape[0])):
            for j in range(self.x.shape[1]):
                now_x = self.x[i, j]
                now_y = self.y[i, j]
                x = self.input + self.x0 * now_x + self.y0 * now_y
                loss = self.model(x)
                result.append(loss.item())
        result = np.array(result)
        result = result.reshape(self.x.shape)
        return result

    @staticmethod
    def draw3D(mesh_x, mesh_y, mesh_z):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        figure = plt.figure()
        axes = Axes3D(figure)

        axes.plot_surface(mesh_x, mesh_y, mesh_z, cmap='rainbow')
        plt.show()

        plt.savefig(D2Landscape.get_datetime_str() + ".png")

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
