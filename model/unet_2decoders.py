"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""
from torch import nn, cat
from torch.utils.tensorboard import SummaryWriter
from torch import rand

class Encoder(nn.Module):

    def __init__(self, filters=64, in_channels=3, n_block=3, kernel_size=(3, 3), batch_norm=True, padding='same'):
        super().__init__()
        self.filter = filters
        for i in range(n_block):
            out_ch = filters * 2 ** i
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = filters * 2 ** (i - 1)

            if padding == 'same':
                pad = kernel_size[0] // 2
            else:
                pad = 0
            encoder = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                     nn.ReLU(inplace=True)]
            if batch_norm:
                encoder += [nn.BatchNorm2d(num_features=out_ch)]
            encoder += [nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                      nn.ReLU(inplace=True)]
            if batch_norm:
                encoder += [nn.BatchNorm2d(num_features=out_ch)]
            self.add_module('encoder%d' % (i + 1), nn.Sequential(*encoder))
            conv = [nn.Conv2d(in_channels=in_ch * 3, out_channels=out_ch, kernel_size=1), nn.ReLU(inplace=True)]
            self.add_module('conv1_%d' % (i + 1), nn.Sequential(*conv))

    def forward(self, x):
        skip = []
        output = x
        res = None
        i = 0
        for name, layer in self._modules.items():
            if i % 2 == 0:
                output = layer(output)
                skip.append(output)
            else:
                if i > 1:
                    output = cat([output, res], 1)
                    output = layer(output)
                output = nn.MaxPool2d(kernel_size=(2,2))(output)
                res = output
            i += 1
        return output, skip


class Bottleneck(nn.Module):
    def __init__(self, filters=64, n_block=3, depth=4, kernel_size=(3,3)):
        super().__init__()
        out_ch = filters * 2 ** n_block
        in_ch = filters * 2 ** (n_block - 1)
        for i in range(depth):
            dilate = 2 ** i
            blocks = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=dilate,
                          dilation=dilate),nn.ReLU(inplace=True)]
            self.add_module('bottleneck%d' % (i + 1), nn.Sequential(*blocks))
            if i == 0:
                in_ch = out_ch

    def forward(self, x):
        bottleneck_output = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            bottleneck_output += output
        return bottleneck_output


class Decoder(nn.Module):
    def __init__(self, filters=64, n_block=3, kernel_size=(3, 3), batch_norm=True, padding='same'):
        super().__init__()
        self.n_block = n_block
        if padding == 'same':
            pad = kernel_size[0] // 2
        else:
            pad = 0
        for i in reversed(range(n_block)):
            out_ch = filters * 2 ** i
            in_ch = 2 * out_ch
            decoder = [nn.UpsamplingNearest2d(scale_factor=(2, 2)),
                     nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                               padding=pad)]
            self.add_module('decoder1_%d' % (i + 1), nn.Sequential(*decoder))

            decoder = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                     nn.ReLU(inplace=True)]
            if batch_norm:
                decoder += [nn.BatchNorm2d(num_features=out_ch)]
            decoder += [nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                      nn.ReLU(inplace=True)]
            if batch_norm:
                decoder += [nn.BatchNorm2d(num_features=out_ch)]
            self.add_module('decoder2_%d' % (i + 1), nn.Sequential(*decoder))

    def forward(self, x, skip):
        i = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            if i % 2 == 0:
                output = cat([skip.pop(), output], 1)
            i += 1
        return output


class Segmentation_model(nn.Module):
    def __init__(self, filters=32, in_channels=3, n_block=4, bottleneck_depth=4, n_class=3):
        super().__init__()
        self.encoder = Encoder(filters=filters, in_channels=in_channels, n_block=n_block)
        self.bottleneck = Bottleneck(filters=filters, n_block=n_block, depth=bottleneck_depth)
        self.decoder = Decoder(filters=filters, n_block=n_block)
        self.classifier = nn.Conv2d(in_channels=filters, out_channels=n_class, kernel_size=(1, 1))

    def forward(self, x, features_out=False):
        output, skip = self.encoder(x)
        output_bottleneck = self.bottleneck(output)
        output = self.decoder(output_bottleneck, skip)
        output = self.classifier(output)
        if features_out:
            return output, output_bottleneck
        else:
            return output
    @staticmethod
    def plot_model(model, x):
        with SummaryWriter('graph') as writer:
            writer.add_graph(model, x)
            print('model plotted')


class Ensemble_model(nn.Module):
    def __init__(self, filters=32, in_channels=3, n_block=4, bottleneck_depth=4, n_class=6):
        super().__init__()
        self.net_C0 = Segmentation_model(filters, in_channels, n_block, bottleneck_depth, n_class=n_class)
        self.net_DE = Segmentation_model(filters, in_channels, n_block, bottleneck_depth, n_class=n_class)
        self.net_T2 = Segmentation_model(filters, in_channels, n_block, bottleneck_depth, n_class=n_class)


    def forward(self, x):
        x_C0 = x[:, 0, None].repeat(1, 3, 1, 1).clone()
        x_DE = x[:, 1, None].repeat(1, 3, 1, 1).clone()
        x_T2 = x[:, 2, None].repeat(1, 3, 1, 1).clone()
        out1 =  self.net_C0(x_C0)
        out2 =  self.net_DE(x_DE)
        out3 = self.net_DE(x_T2)
        return out1 + out2 + out3

if __name__ == '__main__':
    model = Ensemble_model(filters=64, n_block=4)
    x = rand(2, 3, 256 , 256)
    output = model(x)

    print("finish")
    input()

