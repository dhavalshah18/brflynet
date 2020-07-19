import torch
import torch.nn as nn


def crop(input1, input2):
    assert input1.shape[0] == input2.shape[0]
    assert input1.shape[2] - input2.shape[2] in (0, 1)
    assert input1.shape[3] - input2.shape[3] in (0, 1)

    return (input1[:, :, :input2.shape[2], :input2.shape[3]], input2)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, drop_out):
        super(Conv, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5 if drop_out else 0),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x):
        output = self.blk(x)
        return output


class Deconv(nn.Module):
    def __init__(self, in_channels):
        super(Deconv, self).__init__()
        self.blk = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.blk(x)
        return output


class Green(nn.Module):
    def __init__(self, cfg, pos):
        super(Green, self).__init__()
        self.conv = Conv(in_channels=cfg.MODEL.CHANNELS[pos],
                         out_channels=cfg.MODEL.CHANNELS[pos+1],
                         kernel_size=1 if pos == 12 else 3,
                         stride=1,
                         padding=0 if pos == 12 else 1,
                         drop_out=True if pos in (4, 5) else False,
                         )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        output_side = self.conv(x)
        output_side_pad = nn.functional.pad(output_side,
                                            (0, (output_side.shape[3] % 2), 0, (output_side.shape[2] % 2), 0, 0, 0, 0))
        output_main = self.pool(output_side_pad)
        return output_main, output_side


class Purple(nn.Module):
    def __init__(self, cfg, pos):
        super(Purple, self).__init__()
        self.deconv = Deconv(in_channels=cfg.MODEL.CHANNELS[pos])
        self.conv = Conv(in_channels=cfg.MODEL.CHANNELS[pos] + (cfg.MODEL.CHANNELS[13 - pos] if pos < 9
                                                                else cfg.MODEL.CHANNELS[12 - pos]),
                         out_channels=cfg.MODEL.CHANNELS[pos+1],
                         kernel_size=3, stride=1, padding=1, drop_out=False,
                         )

    def forward(self, x_main, x_side):
        output = self.deconv(x_main)

        output = torch.cat(crop(output, x_side), dim=1)
        output = self.conv(output)
        return output


class Red(nn.Module):
    def __init__(self, cfg, pos):
        super(Red, self).__init__()
        self.blk = nn.Conv2d(
            in_channels=cfg.MODEL.CHANNELS[pos],
            out_channels=cfg.MODEL.CHANNELS[pos+1],
            kernel_size=1, stride=1, padding=0,
            )

    def forward(self, x):
        output = self.blk(x)
        return output


class InputArm(nn.Module):
    def __init__(self, cfg):
        super(InputArm, self).__init__()
        self.green0 = Green(cfg, pos=0)
        self.green1 = Green(cfg, pos=1)
        self.green2 = Green(cfg, pos=2)

    def forward(self, x):
        out_main, out_side0 = self.green0(x)
        out_main, out_side1 = self.green1(out_main)
        out_main, out_side2 = self.green2(out_main)
        return out_main, out_side0, out_side1, out_side2


class OutputArm(nn.Module):
    def __init__(self, cfg):
        super(OutputArm, self).__init__()
        self.purple0 = Purple(cfg, pos=9)
        self.purple1 = Purple(cfg, pos=10)
        self.purple2 = Purple(cfg, pos=11)
        self.red = Red(cfg, pos=12)

    def forward(self, x_main, x_side2, x_side1, x_side0):
        out = self.purple0(x_main, x_side2)
        out = self.purple1(out, x_side1)
        out = self.purple2(out, x_side0)
        out = self.red(out)
        return out


class Middle(nn.Module):
    def __init__(self, cfg):
        super(Middle, self).__init__()
        self.green0 = Green(cfg, pos=4)
        self.green1 = Green(cfg, pos=5)
        self.conv = Conv(in_channels=cfg.MODEL.CHANNELS[6],
                         out_channels=cfg.MODEL.CHANNELS[7],
                         kernel_size=3, stride=1, padding=1, drop_out=True,
                         )
        self.purple0 = Purple(cfg, pos=7)
        self.purple1 = Purple(cfg, pos=8)

    def forward(self, x_top, x_bottom):
        out = torch.cat(crop(x_top, x_bottom), dim=1)
        out, side0 = self.green0(out)
        out, side1 = self.green1(out)
        out = self.conv(out)
        out = self.purple0(out, side1)
        out = self.purple1(out, side0)
        return out


class BtrflyNet(nn.Module):
    def __init__(self, cfg):
        super(BtrflyNet, self).__init__()
        self.input_arm_top = InputArm(cfg)
        self.input_arm_bottom = InputArm(cfg)
        self.body = Middle(cfg)
        self.output_arm_top = OutputArm(cfg)
        self.output_arm_bottom = OutputArm(cfg)

    def forward(self, top, bottom):
        body_top, top_side0, top_side1, top_side2 = self.input_arm_top(top)
        body_bottom, bottom_side0, bottom_side1, bottom_side2 = self.input_arm_bottom(bottom)
        body_out = self.body(body_top, body_bottom)
        out_top = self.output_arm_top(body_out, top_side2, top_side1, top_side0)
        out_bottom = self.output_arm_bottom(body_out, bottom_side2, bottom_side1, bottom_side0)
        return out_top, out_bottom
