from self_supervised.augmentations.helper import norm_mean_std, get_color_distortion, GaussianBlur
from torchvision import transforms


class SimCLRTransform:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, size):
        normalize = norm_mean_std(size)
        if size == 28: # MNIST
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(size, size)),
                    transforms.RandomHorizontalFlip(),
                    # get_color_distortion(s=1),
                    # transforms.RandomGrayscale(p=0.3),
                    # transforms.ToPILImage(),
                    # transforms.RandomApply([GaussianBlur([.5, 1.])], p=0.5),
                    # transforms.ToTensor(),
                    # normalize
                ]
            )
        elif size == 224:  # ImageNet
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=size),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=1.0),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    # transforms.ToTensor(),
                    normalize
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(size, size)),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=1),
                    # transforms.RandomGrayscale(p=0.3),
                    # transforms.ToPILImage(),
                    # transforms.RandomApply([GaussianBlur([.5, 1.])], p=0.5),
                    # transforms.ToTensor(),
                    normalize
                ]
            )


    def __call__(self, x, single_view=True):
        if single_view:
            return self.transform(x)
        else:
            return self.transform(x), self.transform(x)

