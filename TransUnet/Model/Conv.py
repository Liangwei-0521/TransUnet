import torch
import torch.nn as nn
import torch.nn.functional as F


class Embeddings(nn.Module):
    def __init__(self, args) -> None:
        super(Embeddings, self).__init__()
        self.content = []
        self.layers = nn.ModuleList()
        for i in range(args.num_layers):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=args.in_channels[i], out_channels=args.out_channels[i],
                          stride=args.stride[i], kernel_size=args.kernel_size[i], padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=args.pool_kernel, stride=args.pool_stride, padding=1)
            )
            self.layers.append(layer)

    def forward(self, x):
        for item_conv2d in self.layers:
            x = item_conv2d(x)
            self.content.append(x.detach())
        return x, self.content


class Cascaded_Upsampler(nn.Module):
    def __init__(self, args) -> None:
        super(Cascaded_Upsampler, self).__init__()
        self.args = args
        self.up_conv = nn.ModuleList()
        for i in range(args.num_layers):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=args.up_in_channels[i], out_channels=args.up_out_channels[i],
                          stride=args.up_stride[i], kernel_size=args.up_kernel[i], padding='same'),
                nn.ReLU()
            )
            self.up_conv.append(layer)
        # Last Conv
        self.last_conv2d = nn.Conv2d(in_channels=args.up_out_channels[-1], out_channels=args.num_classes,
                                     stride=(1, 1), kernel_size=(3, 3), padding='same')

    def forward(self, x, down):
        down = list(reversed(down))
        for i, item_module in enumerate(self.up_conv):
            x = item_module(torch.cat(
                    (F.interpolate(x, size=(down[i].shape[-2], down[i].shape[-1]),
                                   mode='bilinear', align_corners=True)[:, :down[i].shape[-3], :, :], down[i]), dim=-1))
        # Last up sample layer
        x = F.relu(self.last_conv2d(F.interpolate(x, size=(self.args.new_height, self.args.new_width), mode='bilinear',
                                                  align_corners=True)))
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type=list, default=[3, 12, 24], help='in channel')
    parser.add_argument('--out_channels', type=list, default=[12, 24, 48], help='out channels')
    parser.add_argument('--stride', type=list, default=[(1, 1), (1, 3), (1, 3)], help='convolution stride')
    parser.add_argument('--kernel_size', type=list, default=[(2, 3), (2, 3), (2, 3)], help='convolution kernel size')
    args = parser.parse_args()

    params = {
        'batch_size': 32,
        'channel': 3,
        'height': 128,
        'width': 128,
    }

    x = torch.rand(size=(params['batch_size'], params['channel'], params['height'], params['width']))
    conv_net = Embeddings(num_layers=3, args=args)
    result = conv_net(x)
    print(result[0].shape)
    print(result[_].shape for _ in range(len(result[1])))