from numpy import imag
import torch
import torchvision

class EstimatorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(pretrained=True)

        self.backbone = mobilenet.features
        self.aspp = torchvision.models.segmentation.deeplabv3.ASPP(mobilenet.last_channel, [1, 3, 6, 9], mobilenet.last_channel)
        self.aspp.modules

        dconv1_nch = 2 * mobilenet.last_channel
        self.dws_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(dconv1_nch, dconv1_nch, kernel_size=3, stride=1, padding=1, groups=dconv1_nch),
            torch.nn.Conv2d(dconv1_nch, 96, kernel_size=1, stride=1)
        )

        dconv2_nch = 96 + 24
        self.dws_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(dconv2_nch, dconv2_nch, kernel_size=3, stride=1, padding=1, groups=dconv2_nch),
            torch.nn.Conv2d(dconv2_nch, 96, kernel_size=1, stride=1)
        )
        
        self.dws_conv_depth = torch.nn.Sequential(
            torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96),
            torch.nn.Conv2d(96, 96, kernel_size=1, stride=1)
        )
        
        self.dws_conv_seg = torch.nn.Sequential(
            torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96),
            torch.nn.Conv2d(96, 96, kernel_size=1, stride=1)
        )

        self.conv_depth = torch.nn.Conv2d(96 * 2, 1, kernel_size=3, stride=1, padding=1)
        self.conv_seg = torch.nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1)

        self.upsaple8 = torch.nn.UpsamplingBilinear2d(scale_factor=8)
        self.upsaple4 = torch.nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsaple2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, images):
        x = images

        # encoder
        low_lvl_features = x
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 3:
                low_lvl_features = x
        high_lvl_features = x

        #import pdb; pdb.set_trace()
        x = self.aspp(high_lvl_features)
        x = torch.cat([high_lvl_features, x], dim=1)

        # decoder
        x = self.dws_conv1(x)
        x = self.upsaple8(x)
        x = torch.cat([low_lvl_features, x], dim=1)
        x = self.dws_conv2(x)

        x_depth = self.dws_conv_depth(x)
        x_seg = self.dws_conv_seg(x)

        x_depth = torch.cat([x_depth, x_seg], dim=1)
        x_depth = self.conv_depth(x_depth)
        x_seg = self.conv_seg(x_seg)

        x_depth = self.upsaple4(x_depth)
        x_seg = self.upsaple4(x_seg)

        assert x_depth.shape == x_seg.shape
        N, _, H, W = x_seg.shape

        x_depth = x_depth.view(N, H, W)
        x_seg = x_seg.view(N, H, W)

        return self.sigmoid(x_seg), self.sigmoid(x_depth)

if __name__ == '__main__':
    from data import DepthEstimatorDataset, collate_fn
    import cv2

    data_path = '/home/oodapow/data/RHD_published_v2'
    dataset = DepthEstimatorDataset(data_path, 'evaluation')

    image, mask, depth = dataset[0]

    model = EstimatorModel()

    image = cv2.resize(image, (224, 224))

    image_tensor = torch.tensor(image).permute(2, 0, 1).div(255.)[None,:,:,:]

    e_mask, e_depth = model(image_tensor)

    print(e_mask.shape)
    print(e_depth.shape)


        
