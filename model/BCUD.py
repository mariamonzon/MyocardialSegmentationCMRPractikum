import torch.nn as nn
import torch
from ConvLSTM import ConvLSTM2D
""" Pytorch implementation of https://github.com/rezazad68/BCDU-Net/"""

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)



class Bottleneck(nn.Module):
    def __init__(self, filters=256, depth=3, kernel_size=(3,3), dropout = 0.5):
        super().__init__()
        in_ch = filters
        out_ch = filters*2
        pad = tuple(k//2 for k in kernel_size)
        # for i in range(depth):
        #     dilate = 2 ** i
        #     blocks = [  nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=dilate,
        #                   dilation=dilate),
        #                 nn.ReLU(inplace=True),
        #                 nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=dilate,
        #                 dilation=dilate),
        #                 nn.ReLU(inplace=True)]
        #     if  dropout :
        #         blocks +=[nn.Dropout2d(p= dropout )]
        #     self.add_module('bottleneck%d' % (i + 1), nn.Sequential(*blocks))
        #     if i == 0:
        #         in_ch = out_ch

        self.conv_1 =  nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad)
        self.conv_2 =  nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad)
        self.conv_3 = nn.Conv2d(in_channels= 2*out_ch, out_channels=out_ch, kernel_size=kernel_size, padding=pad)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        # D1
        x = self.conv_1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv_2(x)
        x = nn.ReLU(inplace=True)(x)
        x1 = nn.Dropout(p=0.5)(x)

        # D2
        x = self.conv_2(x1)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv_2(x)
        x = nn.ReLU(inplace=True)(x)
        x2 = nn.Dropout(p=0.5)(x)
        x_cat = torch.cat( [x1, x2], dim = 1)

        # D3
        x = self.conv_3(x_cat)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv_2(x)
        x = nn.ReLU(inplace=True)(x)
        output = nn.Dropout(p=0.5)(x)


        # output_tot = 0
        # output = x
        # concat = []
        # # ResDecoder
        # for _, layer in self._modules.items():
        #     output = layer(output)
        #     output_tot += output

        return output



class BCDU_net_D3(nn.Module):

    def __init__(self, filters =64, in_channels = 1, n_class=4, n_layers = 3, kernel = 3, pad = 1, activation = nn.ReLU(inplace=True)):
        super().__init__()
        self.filters = [filters * 2 ** l for l in range(n_layers)]
        self.n_layers = n_layers

        # self.act = nn.ReLU(inplace=True) # activation
        self.encoder_convblock_0 = nn.Sequential(*[nn.Conv2d(in_channels, self.filters[0], kernel_size= kernel, padding=1),
                                                   activation,
                                                   nn.Conv2d( self.filters[0],  self.filters[0], kernel_size= kernel, padding=1),
                                                   activation])

        for l  in range(1, n_layers):
            in_ch =  self.filters[l - 1]
            out_ch = self.filters[l]
            conv_block =  [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, padding=pad),
                           activation,
                           nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel, padding=pad),
                           activation]
            self.add_module('encoder_convblock_%d' % (l), nn.Sequential(*conv_block))


        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.dropout = nn.Dropout2d(p=0.5)
        self.bottleneck = Bottleneck(filters=self.filters[-1], depth=3, kernel_size=(3,3))

        for i in reversed(range(self.n_layers)):
            out_ch =  self.filters[i] # filters[0] * 2 ** i
            in_ch = 2 * out_ch
            decoder = [nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch , kernel_size=2, stride=2),
                       nn.BatchNorm2d(num_features=out_ch),
                       activation]
            self.add_module('decoder_upconv_%d' % (i ), nn.Sequential(*decoder))

            # x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
            # x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)

            # decoder = [ nn.Conv2d(in_channels=out_ch, out_channels=out_ch , kernel_size=2, stride=2),
            #             nn.BatchNorm2d(num_features=out_ch),
            #             # ConvLSTM2D(input_bands, input_dim, kernels, num_layers, bidirectional, dropout)
            #             nn.ReLU(inplace=True)]
            # decoder += [nn.BatchNorm2d(num_features=out_ch)]

            # decoder = [nn.ConvTranspose2d(self.filters[-1], self.filters[-2], kernel_size=2, padding=1),
            #            nn.BatchNorm2d(num_features=self.filters[-1]),
            #            activation]
            #
            # self.add_module('decoder_upconv_%d' % (i), nn.Sequential(*decoder))
            #

        decoder = [ConvLSTM2D(self.filters[i], self.filters[i] , kernels =(3, 3), num_layers=2, bidirectional=True, dropout=0),
                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel, padding=pad),
                   activation,
                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel, padding=pad),
                   activation ]
        self.add_module('decoder_conv_%d' % (i + 1), nn.Sequential(*decoder))


        self.classifier = nn.Conv2d(in_channels=filters, out_channels=n_class, kernel_size=(1, 1))

        self.up_6 = nn.ConvTranspose2d(self.filters[-1] ,  self.filters[-2], kernel_size=2, padding=1)
        self.bn = nn.BatchNorm2d(num_features=self.filters[-1] )



        # up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
        # up6 = BatchNormalization(axis=3)(up6)
        # up6 = Activation('relu')(up6)
        #
        # x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
        # x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
        # merge6  = concatenate([x1,x2], axis = 1)
        # merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
        #
        # conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        # conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        #
        # up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
        # up7 = BatchNormalization(axis=3)(up7)
        # up7 = Activation('relu')(up7)
        #
        # x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)
        # x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
        # merge7  = concatenate([x1,x2], axis = 1)
        # merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        #
        # conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        # conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        #
        # up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
        # up8 = BatchNormalization(axis=3)(up8)
        # up8 = Activation('relu')(up8)

        # x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
        # x2 = Reshape(target_shape=(1, N, N, 64))(up8)
        # merge8  = concatenate([x1,x2], axis = 1)
        # merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)
        #
        # conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        # conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        # conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        # conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)



    def forward(self, input):

        concat = []
        x = input
        # Encoder
        for l, (name, layer) in zip(range(self.n_layers), self._modules.items()):
            x = layer(x)
            concat.append(x)
            if l >=2:
                x = self.dropout(x)
            x = self.pool(x)

        x = self.bottleneck(x)



if __name__ == '__main__':
    from torch import rand
    model = BCDU_net_D3(filters=64)
    x = rand(2, 1, 256 , 256)
    output = model(x)

    bottle = Bottleneck(filters=256, depth=3, kernel_size=(3, 3), dropout=0.5)

    print(output.shape)
    print("finish")


