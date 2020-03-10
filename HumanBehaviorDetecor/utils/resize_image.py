from core.config import cfg

class ResizeWithPad:
    def __init__(self, h, w):
        super(ResizeWithPad, self).__init__()
        self.H = cfg.TRAIN.INPUT_SIZE[0]
        self.W = cfg.TRAIN.INPUT_SIZE[0]
        self.h = h
        self.w = w

    def get_transform_coefficient(self):
        if self.h <= self.w:
            longer_edge = "w"
            scale = self.W / self.w
            padding_length = (self.H - self.h * scale) / 2
        else:
            longer_edge = "h"
            scale = self.H / self.h
            padding_length = (self.W - self.w * scale) / 2
        return longer_edge, scale, padding_length

    def raw_to_resized(self, x_min, y_min, x_max, y_max):
        longer_edge, scale, padding_length = self.get_transform_coefficient()
        x_min = x_min * scale
        x_max = x_max * scale
        y_min = y_min * scale
        y_max = y_max * scale
        if longer_edge == "h":
            x_min += padding_length
            x_max += padding_length
        else:
            y_min += padding_length
            y_max += padding_length
        return x_min, y_min, x_max, y_max

    def resized_to_raw(self, center_x, center_y, width, height):
        longer_edge, scale, padding_length = self.get_transform_coefficient()
        center_x *= self.W
        width *= self.W
        center_y *= self.H
        height *= self.H
        if longer_edge == "h":
            center_x -= padding_length
        else:
            center_y -= padding_length
        center_x = center_x / scale
        center_y = center_y / scale
        width = width / scale
        height = height / scale
        return center_x, center_y, width, height

if __name__ == '__main__':
    a = ResizeWithPad(w=90,h=40)
    from PIL import Image
    import numpy as np
    b = Image.open(r'F:\Gfanqi\YOLOv3_TensorFlow2-master\test_results_during_training\epoch-2683-picture-10.jpg')
    b =np.array(b)
    print(type(b))
    print(a.raw_to_resized(1,2,3,4,))