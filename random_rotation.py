import imgaug as ia
from imgaug import augmenters as iaa


def get():
    #def sometimes(aug): return iaa.Sometimes(0.5, aug)

    return iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            
            iaa.OneOf([
                # blur images with a sigma between 0 and 3.0
                iaa.GaussianBlur((0, 3.0)),
                # blur image using local means with kernel sizes between 2 and 7
                iaa.AverageBlur(k=(2, 7)),
                # blur image using local medians with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)),
                
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.05*255), per_channel=0.5),                
                ]),            
        ],
        random_order=True
    )
