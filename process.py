"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./export.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
import torch
import os
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms
import math
import gc

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
def origin_crop(label, size=[64,256,256]):
    
    img_d, img_h, img_w = label.shape

    d0 = math.ceil((size[2]- img_d))
    h0 = math.ceil((size[1]- img_h))
    w0 = math.ceil((size[0]- img_w))
    d1 = -d0 + size[2]
    h1 = -h0 + size[1]
    w1 = -w0 + size[0]

    d0 = np.max([d0, 0])
    d1 = np.min([d1, size[2]])
    h0 = np.max([h0, 0])
    h1 = np.min([h1, size[1]])
    w0 = np.max([w0, 0])
    w1 = np.min([w1, size[0]])
    return label[d0: d1, h0: h1, w0: w1]
def transform():
    options = []
    options.append(transforms.Lambda(lambda x: norm_img(x))) # min max nomalization
    options.append(transforms.Lambda(lambda x: pad_img(x))) # resize
    options.append(transforms.Lambda(lambda x: crop_img(x))) # centor_crop
    options.append(transforms.Lambda(lambda x: tensor_img(x))) # Totensor
    transform = transforms.Compose(options)
    return transform
def centor_crop(label, size=[64,256,256]):

    img_d, img_h, img_w = label.shape
    d0 = math.ceil((img_d - size[0])/2)
    h0 = math.ceil((img_h - size[1])/2)
    w0 = math.ceil((img_w - size[2])/2)
    d1 = d0 + size[0]
    h1 = h0 + size[1]
    w1 = w0 + size[2]

    d0 = np.max([d0, 0])
    d1 = np.min([d1, img_d])
    h0 = np.max([h0, 0])
    h1 = np.min([h1, img_h])
    w0 = np.max([w0, 0])
    w1 = np.min([w1, img_w])

    return label[d0: d1, h0: h1, w0: w1]

def pad_image(img, size=[64,256,256]):
    rows_missing = math.ceil(size[0] - img.shape[0])
    cols_missing = math.ceil(size[1] - img.shape[1])
    dept_missing = math.ceil(size[2] - img.shape[2])
    if rows_missing < 0:
        rows_missing = 0
    if cols_missing < 0:
        cols_missing = 0
    if dept_missing < 0:
        dept_missing = 0

    padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
    return padded_img    
def tensor_img(dic_image):
    for key in dic_image:
        x = dic_image[key]
        dic_image[key] = torch.from_numpy(x.astype(np.float32))
    return dic_image   
def norm_img(dic_image):
    for key in dic_image:
        if key in ['q', 'v']:
            x = dic_image[key]
            dic_image[key] = (x - np.min(x))/(np.max(x)-np.min(x))
    return dic_image   
def pad_img(dic_image):
    for key in dic_image:
        x = dic_image[key]
        dic_image[key] = pad_image(x)
    return dic_image  
def crop_img(dic_image):
    for key in dic_image:
        x = dic_image[key]
        dic_image[key] = centor_crop(x)
    return dic_image  
def get_default_device():
    ######## set device#########
    if torch.cuda.is_available():
        print ("Using gpu device")
        return torch.device('cuda')
    else:
        print ("Using cpu device")
        return torch.device('cpu')
def stack_data(image):

    return torch.stack([image['q'], image['k'], image['v']])
def get_metainfo(itk_image):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    original_origin = itk_image.GetOrigin()
    original_direction = itk_image.GetDirection()
    original_pixelid = itk_image.GetPixelIDValue()
    return {'spacing': original_spacing, 'size':original_size, 
            'origin': original_origin, 'direction':original_direction, 'pixelid':original_pixelid}

def pooling3x3x3(kernel_size=2, stride=2) -> nn.MaxPool3d:
    """3x3x3 pooling"""
    return nn.MaxPool3d(kernel_size=kernel_size)
def convT3x3x3(in_channels=3, out_channels=1, kernel_size=2, stride=2) -> nn.ConvTranspose2d:
    """3x3x3 convolution transpose"""
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
def conv3x3x3(in_channels, out_channels, kernel_size=1) -> nn.Conv3d:
    """3x3x3 convolution """
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size)

class UNet3D(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = pooling3x3x3(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = pooling3x3x3(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = pooling3x3x3(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = pooling3x3x3(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = convT3x3x3(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = convT3x3x3(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = convT3x3x3(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = convT3x3x3(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = conv3x3x3(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
    def odd_resolution_wrapper(self, x, new_size):
        # return nn.functional.interpolate(x, size=new_size, scale_factor=(2, 1, 1), mode="trilinear", align_corners=True)
        return nn.functional.interpolate(x, size=new_size, mode="trilinear")
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        if dec4.shape[-3] != enc4.shape[-4]:
            dec4 = self.odd_resolution_wrapper(dec4, tuple(enc4.shape[-3:]))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        if dec3.shape[-3] != enc3.shape[-4]:
            dec3 = self.odd_resolution_wrapper(dec3, tuple(enc3.shape[-3:]))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        if dec2.shape[-3] != enc2.shape[-4]:
            dec2 = self.odd_resolution_wrapper(dec2, tuple(enc2.shape[-3:]))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        if dec1.shape[-3] != enc1.shape[-4]:
            dec1 = self.odd_resolution_wrapper(dec1, tuple(enc1.shape[-3:]))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
def get_images():
    image_paths = []
    input_skull_stripped_adc = glob(str(INPUT_PATH / "images/skull-stripped-adc-brain-mri" / "*.mha"))
    z_adc = glob(str(INPUT_PATH / "images/z-score-adc" / "*.mha"))
    for adc, z in zip (input_skull_stripped_adc, z_adc):
        tmp = z.split(os.sep)[-1]
        name = tmp.split('.')[0]
        image_paths.append([adc, z, name])
    return image_paths
def run():
    # load pre trained parameters
    state_dict = torch.load(f'unet_11-03-19-51.pt', map_location=get_default_device())
    # get transform
    train_transform = transform()
    # Read the input

    image_paths = get_images()
    for adc, z, name in image_paths:
        input_skull_stripped_adc, _ = load_image_file_as_array(adc)
        z_adc, info = load_image_file_as_array(z)
        metainfo = get_metainfo(info)
        zadcb = np.where(z_adc<-2, 1, 0)
        images = train_transform({"q": z_adc, "k":zadcb, "v": input_skull_stripped_adc})
        img = stack_data(images) 
        # Process the inputs: any way you'd like
        _show_torch_cuda_info()

        with torch.no_grad():
            znadcnth_input=img[None,...].to(get_default_device())

            net=UNet3D(in_channels=3, out_channels=1)
            net.load_state_dict(state_dict)
            net.eval()
            net=net.to(get_default_device())   

            out=net(znadcnth_input)

            out=out.detach().cpu().numpy().squeeze(0).squeeze(0)
            out = np.where(out>0.90, 1, 0).astype(np.uint8)
            out = origin_crop(out, metainfo['size'])

            print ("check out",np.sum(out))
            

        hie_segmentation=SimpleITK.GetImageFromArray(out)    


        # For now, let us save predictions
        save_image(hie_segmentation, name)
        del out, znadcnth_input, img, images, input_skull_stripped_adc, z_adc, zadcb
        gc.collect()
    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))



def save_image(pred_lesion, name):
    relative_path="images/hie-lesion-segmentation"
    output_directory = OUTPUT_PATH / relative_path

    output_directory.mkdir(exist_ok=True, parents=True)

    file_save_name=output_directory / f"{name}.mha"

    SimpleITK.WriteImage(pred_lesion, file_save_name)
    check_file = os.path.isfile(file_save_name)
    print ("check file", check_file)



def load_image_file_as_array(location):
    # Use SimpleITK to read a file
    result = SimpleITK.ReadImage(location)

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), result


def _show_torch_cuda_info():


    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
