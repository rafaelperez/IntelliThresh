import numpy as np
import cv2
import matplotlib.pyplot as plt

bg_path = '/media/mscv1/14fecf86-bdfa-4ebd-8b47-eea4ddee198e/ChengleiSocial/undistorted/000000/400036.png'
object_path = '/media/mscv1/14fecf86-bdfa-4ebd-8b47-eea4ddee198e/alphamatting_datasets/alphamatting_dot_com/input_training_highres/GT14.png'
alpha_path = '/media/mscv1/14fecf86-bdfa-4ebd-8b47-eea4ddee198e/alphamatting_datasets/alphamatting_dot_com/gt_training_highres/GT14.png'

alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
obj = cv2.imread(object_path, cv2.IMREAD_COLOR)
bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)

canvas = np.zeros(bg.shape, dtype=np.uint8)


def show_img(img):
    plt.imshow(img[:, :, ::-1])
    plt.show()


class SynObj(object):
    def __init__(self, img, alpha, size, coord, angle, color_mix_factor, shadow_size_factor, shadow_strength):
        self.img = img
        self.alpha = alpha
        self.size = size
        self.coord = coord
        # degree, anti-clockwise
        self.angle = angle
        self.color_mix_factor = color_mix_factor # < 0.3
        self.shadow_size_factor = shadow_size_factor # < 0.2
        self.shadow_strength = shadow_strength # < 0.5

    def get_transformed_layer(self, canvas, img):
        canvas_h, canvas_w = canvas.shape[0:2]
        # scale
        th, tw = self.size
        img = cv2.resize(img, (tw, th))

        cx, cy = 0.5 * tw, 0.5 * th

        # rotate
        rotate_M_23 = cv2.getRotationMatrix2D((cx, cy), self.angle, 1.0)
        rotate_M = np.zeros((3, 3), dtype=np.float32)
        rotate_M[0:2, :] = rotate_M_23
        rotate_M[2, 2] = 1

        # translate
        tx, ty = self.coord
        dx, dy = tx - cx, ty - cy
        translate_M = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

        # do transform
        transform_M = translate_M @ rotate_M
        transform_M = transform_M[:2, :]
        transformed_layer = cv2.warpAffine(img, transform_M, (canvas_w, canvas_h))

        return transformed_layer

    def render(self, canvas):
        img_layer = self.get_transformed_layer(canvas, self.img)
        alpha_layer = self.get_transformed_layer(canvas, self.alpha).astype(np.float32) / 255
        alpha_layer = alpha_layer[:, :, np.newaxis]
        canvas[:, :, :] = np.uint8(canvas * (1 - alpha_layer) + img_layer * alpha_layer)

    def light_img(self, light, factor):
        new_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)
        light_lab = cv2.cvtColor(light, cv2.COLOR_Lab2BGR)
        new_img[:, :, 1:] = (light_lab[:, :, 1:] * factor + new_img[:, :, 1:] * (1 - factor)).astype(np.uint8)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_Lab2BGR)
        return new_img

    def render_with_lighting(self, canvas, lighting):
        img = self.light_img(lighting, self.color_mix_factor)
        img_layer = self.get_transformed_layer(canvas, img)
        alpha_layer = self.get_transformed_layer(canvas, self.alpha).astype(np.float32) / 255
        alpha_layer = alpha_layer[:, :, np.newaxis]
        canvas[:, :, :] = np.uint8(canvas * (1 - alpha_layer) + img_layer * alpha_layer)

    def render_with_shadow(self, canvas, lighting):
        img = self.light_img(lighting, self.color_mix_factor)
        img_layer = self.get_transformed_layer(canvas, img)
        alpha_layer = self.get_transformed_layer(canvas, self.alpha)
        
        radius = int(self.shadow_size_factor*max(self.size)) // 2 * 2 + 1
        strength = self.shadow_strength
        shadow = cv2.GaussianBlur(self.alpha, (radius, radius), radius)
        shadow_layer = self.get_transformed_layer(canvas, shadow).astype(np.float32) / 255 * strength
        
        alpha_layer = alpha_layer.astype(np.float32) / 255

        shadow_layer = shadow_layer[:, :, np.newaxis]
        alpha_layer = alpha_layer[:, :, np.newaxis]

        tmp_canvas = canvas * (1 - shadow_layer)
        canvas[:, :, :] = np.uint8(tmp_canvas * (1 - alpha_layer) + img_layer * alpha_layer)


if __name__ == '__main__':
    syn_obj = SynObj(obj, alpha, (2000, 1600), (1000, 2000), 45, 0.30, 0.2, 0.2)
    canvas = bg.copy()

    """
    lighting = cv2.resize(canvas, (200, 400))
    lighting = cv2.GaussianBlur(lighting, (41, 41), 20)
    lighting = cv2.resize(lighting, canvas.shape[0:2][::-1], interpolation=cv2.INTER_CUBIC)
    """

    lighting = np.mean(canvas, axis=(0, 1), keepdims=True).astype(np.uint8)

    syn_obj.render_with_shadow(canvas, lighting)

    show_img(canvas)