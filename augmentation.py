import PIL
import torchvision.transforms as transforms
import numpy as np


def cutout_test(image, p):

    image = np.asarray(image).copy()

    draw = np.random.rand()
    if draw > p:
        return image

    h, w = image.shape[:2]

    draw = np.random.uniform(0, 0.5, 1)
    if draw == 0:
        return image
    else:
        patch_size = int(draw * h)

    lu_x = np.random.randint(0, w - patch_size)
    lu_y = np.random.randint(0, h - patch_size)

    mask_color = np.asarray([0.5, 0.5, 0.5])

    image[lu_y:lu_y + patch_size, lu_x:lu_x + patch_size] = mask_color

    return image


def auto_contrast(image):
    return PIL.ImageOps.autocontrast(image)


def brightness(image):
    enhancer = PIL.ImageEnhance.Brightness(image)
    factor = np.random.random()
    enhancer.enhance(factor)
    return image


def contrast(image):
    enhancer = PIL.ImageEnhance.Contrast(image)
    factor = np.random.random()
    enhancer.enhance(factor)
    return image


def equalize(image, alpha=0.3):
    eq_image = PIL.ImageOps.equalize(image)
    blend_image = PIL.Image.blend(image, eq_image, alpha)
    return blend_image


def invert(image, alpha=0.3):
    inv_image = PIL.ImageOps.invert(image)
    blend_image = PIL.Image.blend(image, inv_image, alpha)
    return blend_image


def nop(image):
    return image


def posterize(image):
    num_bits = np.random.choice(np.arange(1, 8))
    return PIL.ImageOps.posterize(image, num_bits)


def solarize(image):
    max_val = image.getextrema()[0][0]
    values = np.linspace(0, max_val, 256)
    threshold = np.random.choice(values)
    return PIL.ImageOps.solarize(image, threshold)


def sharpness(image):
    radius = 2
    percent = 150
    threshold = 3
    H = PIL.ImageFilter.UnsharpMask(radius, percent, threshold)
    image = image.filter(H)
    return image


def smooth(image):
    radius = 0.5
    H = PIL.ImageFilter.GaussianBlur(radius)
    image = image.filter(H)
    return image


def cifar_strong_transforms():
    transform_list = [
        transforms.Lambda(lambda x: auto_contrast(x)),
        transforms.Lambda(lambda x: brightness(x)),
        transforms.RandomGrayscale(p=0.2),
        transforms.Lambda(lambda x: contrast(x)),
        transforms.Lambda(lambda x: equalize(x, alpha=0.3)),
        transforms.Lambda(lambda x: invert(x, alpha=0.3)),
        transforms.Lambda(lambda x: nop(x)),
        transforms.RandomResizedCrop(32),
        transforms.Lambda(lambda x: posterize(x)),
        transforms.Lambda(lambda x: solarize(x)),
        transforms.RandomRotation(degrees=35),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Lambda(lambda x: sharpness(x)),
        transforms.Lambda(lambda x: smooth(x)),
        transforms.Lambda(lambda x: cutout_test(x, p=1.0))
    ]

    transform_compose = transforms.Compose([
        transforms.RandomChoice(transform_list),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    return transform_compose


def cifar_weak_transforms():
    transform_compose = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    return transform_compose


def cifar_test_transforms():
    transform_compose = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    return transform_compose
