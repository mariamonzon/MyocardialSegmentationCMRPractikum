"""
Multi-Modality Pathology Segmentation Framework:
Application to Cardiac Magnetic Resonance Images
"""
from torch import nn, cat
from torch.utils.tensorboard import SummaryWriter
from torch import rand
import torch.nn.functional as F


class denoising_model(nn.Module):
    def __init__(self, imsize=64):
        super(denoising_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(imsize * imsize, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)

        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, imsize * imsize),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = F.relu(self.enc4(x))
        x = self.pool(x)  # the latent space representation

        # decode
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.sigmoid(self.out(x))
        return x

class Encoder(nn.Module):

    def __init__(self, filters=64, in_channels=3, n_block=3, kernel_size=(3, 3), batch_norm=True, padding='same',
                 activation = nn.LeakyReLU(inplace=True)):
        super().__init__()
        self.filter = filters
        pad = kernel_size[0] // 2 if padding == 'same' else 0

        for i in range(n_block):
            out_ch = filters * 2 ** i
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = filters * 2 ** (i - 1)

            encoder = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                     activation]
            if batch_norm:
                encoder += [nn.BatchNorm2d(num_features=out_ch)]
            encoder += [nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                      activation]
            if batch_norm:
                encoder += [nn.BatchNorm2d(num_features=out_ch)]
            self.add_module('encoder%d' % (i + 1), nn.Sequential(*encoder))
            conv = [nn.Conv2d(in_channels=in_ch * 3, out_channels=out_ch, kernel_size=1), activation]
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
            decoder = [nn.UpsamplingBilinear2d(scale_factor=(2, 2)),
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

    def forward(self, x):
        i = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            i += 1
        return output


class EncoderBlock(nn.Module):
    def __init__(self, in_ch = 0, out_ch=32, kernel_size=(3, 3), pad = 'same'):
        super().__init__()
        in_ch  = 1 if in_ch ==0 else in_ch
        pad = kernel_size[0]//2 if pad == 'same' else pad
        conv_block = [ nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                    nn.BatchNorm2d(num_features=out_ch),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                    nn.BatchNorm2d(num_features=out_ch),
                    nn.LeakyReLU(inplace=True)]
        self.conv_block = nn.Sequential(*conv_block)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch=0, kernel_size=(3, 3)):
        super().__init__()
        out_ch = in_ch//2 if out_ch == 0 else out_ch
        pad = kernel_size[0]//2
        conv_block = [  nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                    nn.BatchNorm2d(num_features=out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad),
                    nn.BatchNorm2d(num_features=out_ch),
                    nn.ReLU(inplace=True)]
        self.conv_block = nn.Sequential(*conv_block)
        self.unpool = nn.UpsamplingBilinear2d(scale_factor=(2, 2))

    def forward(self, x):
        x = self.conv_block(x)
        x = self.unpool(x)
        return x


class FusionConvBlock(nn.Module):
    def __init__(self, filters=32,  branches = 3, stride = (96,96), kernel_size=(3, 3)):
        super().__init__()

        in_ch  = (branches+1)*filters
        mid_ch = in_ch//stride[0]

        pad = kernel_size[0]//2
        blocks = [nn.AvgPool2d(kernel_size=stride),
                 nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=(1,1), padding=0),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(in_channels=mid_ch, out_channels=in_ch, kernel_size=(1,1), padding=0),
                 nn.Sigmoid() ]
        self.add_module('att_conv', nn.Sequential(*blocks))

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, padding=pad)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, x_concat):
        for x_cat in x_concat:
            x = cat([x, x_cat], 1)
        output = x
        output = self.att_conv(output)
        output = x*output
        output = self.conv(output)
        output= self.act(output)
        return output



class EncoderDecoderBranch(nn.Module):

    def __init__(self, filters=64, in_channels=1, n_block=3, n_class=2, kernel_size=(3, 3),  padding='same'):
        super().__init__()
        self.filter = filters
        pad =  kernel_size[0] // 2  if padding == 'same' else 0
        num_filters = [ in_channels]
        num_filters += [filters * 2 ** i for i in range(n_block)]
        for i, f  in enumerate(num_filters[:-1]) :
            encode_block = EncoderBlock(f, num_filters[i+1])
            self.add_module('encoder%d' % (i), encode_block)
        for d in range(len(num_filters)-1, 0, -1):
            decode_block = DecoderBlock(num_filters[d], num_filters[d - 1])
            self.add_module('decoder%d' % (d), decode_block)
        self.classifier =  nn.Conv2d(in_channels=in_channels, out_channels= n_class , kernel_size=(1, 1))
    def forward(self, x):
        features = []
        output = x
        for name, layer in self._modules.items():
            output = layer(output)
            features.append(output)
        return output, features[:-2]


class FusionBranch(nn.Module):

    def __init__(self, filters=32, n_block=3, kernel_size=(3, 3), im_size=96, branches=3):
        super().__init__()
        self.filter = filters
        self.im_size = im_size
        self.filters = [filters*2**i for i in range(n_block)]

        for i in range(n_block-1):
            fusion_convblock = FusionConvBlock( filters=self.filters[i], branches = branches,
                                                stride = (self.im_size//(2**(i+1)), self.im_size//(2**(i+1))),
                                                kernel_size=kernel_size)
            self.add_module('fusion_convblock%d' % (i), fusion_convblock)
            ch = (branches+1)*self.filters[i]
            conv_block = EncoderBlock(ch, self.filters[i]*2 )
            self.add_module('encoder_convblock%d' % (i), conv_block)

        for i in range(n_block-1, -1, -1):
            fusion_convblock = FusionConvBlock( filters=self.filters[i], branches = branches,
                                                stride = (self.im_size//(2**(i+1)), self.im_size//(2**(i+1))),
                                                kernel_size=kernel_size)
            self.add_module('fusion_block%d' % (n_block-i), fusion_convblock)
            conv_block = DecoderBlock((branches+1)*self.filters[i],  self.filters[i]//2 )
            self.add_module('decoder_convblock%d' % (n_block-i), conv_block)

    def forward(self, x_fusion):
        f1,f2,f3 =  x_fusion
        branches  = len(x_fusion)
        output = cat([f1[0], f2[0], f3[0]], 1)
        output = nn.Conv2d(branches * self.filters[0], self.filters[0], kernel_size=(3,3), padding = 1)(output)
        i = 0
        for name, layer in self._modules.items():#[:len(self._modules)//2]:
            if i % 2 == 0:
                # Cahnnelwise fusion attention block
                output = layer(output, [f1[i // 2], f2[i // 2], f3[i // 2]])
            else:
                # Encoder / Decoder block
                output = layer(output)
            i += 1

        return output


class PSNR(nn.Module):
    def __init__(self, filters=32, in_channels=3, n_block=3, n_class=4, im_size = 96):
        super().__init__()
        self.branch_C0 = EncoderDecoderBranch(filters=filters, in_channels=in_channels, n_block=n_block, n_class = 2)
        self.branch_DE = EncoderDecoderBranch(filters=filters, in_channels=in_channels, n_block=n_block, n_class = 3)
        self.branch_T2 = EncoderDecoderBranch(filters=filters, in_channels=in_channels, n_block=n_block, n_class =3)

        self.fusion_branch =FusionBranch(filters=filters, n_block=n_block, kernel_size=(3, 3), im_size=im_size, branches=3)

        self.classifier = nn.Conv2d(in_channels=filters//2, out_channels=n_class, kernel_size=(1, 1))

    def forward(self, x):
        x_C0 = x[:, 0, None].repeat(1, 3, 1, 1).clone()
        x_DE = x[:, 1, None].repeat(1, 3, 1, 1).clone()
        x_T2 = x[:, 2, None].repeat(1, 3, 1, 1).clone()

        output_C0, F_C0 = self.branch_C0(x_C0)
        output_DE, F_DE= self.branch_DE(x_DE)
        output_T2, F_T2 = self.branch_T2(x_T2)
        F_fusion = self.fusion_branch( (F_C0, F_DE, F_T2) )
        output_fusion = self.classifier(F_fusion)
        return [output_fusion, output_C0, output_DE, output_T2]

    @staticmethod
    def plot_model(model, x):
        with SummaryWriter('graph') as writer:
            writer.add_graph(model, x)
            print('model plotted')


class Branched_model(nn.Module):
    def __init__(self, filters=32, in_channels=3, n_block=4, bottleneck_depth=4, n_class=4):
        super().__init__()
        self.encoder_C0 = Encoder(filters=filters, in_channels=in_channels, n_block=n_block)
        self.encoder_DE = Encoder(filters=filters, in_channels=in_channels, n_block=n_block)
        self.encoder_T2 = Encoder(filters=filters, in_channels=in_channels, n_block=n_block)
        self.fusion_unet = FusionBranch(filters=filters, n_block = n_block, im_size=96)
        self.bottleneck = Bottleneck(filters=filters, n_block=n_block, depth=bottleneck_depth)

        self.decoder = Decoder(filters=filters, n_block=n_block)
        self.decoder2  = Decoder(filters=filters, n_block=n_block)
        self.classifier = nn.Conv2d(in_channels=filters, out_channels=n_class, kernel_size=(1, 1))
        self.classifier2 = nn.Conv2d(in_channels=filters, out_channels=2, kernel_size=(1, 1))

    def forward(self, x, features_out=False):
        output, skip = self.encoder(x)
        output_bottleneck = self.bottleneck(output)
        output_2 = output_bottleneck.clone()
        # Branch 1
        output2 = self.decoder(output_2, skip.copy())
        output2 = self.classifier2(output2)
        # Branch 2
        output4 = self.decoder2(output_bottleneck, skip)
        output4 = self.classifier(output4)
        return output2, output4

    @staticmethod
    def plot_model(model, x):
        with SummaryWriter('graph') as writer:
            writer.add_graph(model, x)
            print('model plotted')


if __name__ == '__main__':
    model = PSNR(filters=32, in_channels= 3, n_block= 3)
    x = rand(4, 3, 96 , 96)
    out = model(x)
    print(out[1].shape)
    print("finish")


