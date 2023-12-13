import torch.nn as nn
import torch
import torch.nn.functional as F

#selfattention
class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)
        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*input.shape)
        return self.gamma * out + input

#MGCE
class MGCE(nn.Module):
    def __init__(self,c_in,c_out,feature_num):
        super(MGCE, self).__init__()
        self.dcnn2d1_MGCE = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(1, 2), padding=(0, 0), dilation=(1, 2)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )

        self.fcD11_MGCE = nn.Linear(feature_num, feature_num, bias=True)
        self.fcD12_MGCE = nn.Linear(feature_num, feature_num, bias=True)
        self.fcD13_MGCE = nn.Linear(feature_num, feature_num, bias=True)

        self.dcnn2d2_MGCE = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=(1, 2), padding=(0, 1), dilation=(1, 2)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )

        self.cnn2dRes_MGCE = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(1, 2), padding=(0, 0), dilation=(1, 2)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )

        self.cnn2ddown_MGCE = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=(1, 2), padding=(0, 1), dilation=(1, 2)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )

    def forward(self, X0,X,AC0,AD0,AW0):
        B = X.shape[0]
        X = self.dcnn2d1_MGCE1(X)
        X = X.view(B, -1, X.shape[2], X.shape[3])
        c = X.shape[1]
        AC0_MGCE1 = AC0.view(B, 1, AC0.shape[1], AC0.shape[2])
        AC0_MGCE1 = torch.broadcast_to(AC0_MGCE1, (
        B, X.shape[1], AC0_MGCE1.shape[2], AC0_MGCE1.shape[3]))

        AD0_MGCE1 = AD0.view(B, 1, AD0.shape[1], AD0.shape[2])
        AD0_MGCE1 = torch.broadcast_to(AD0_MGCE1,
                                       (B, X.shape[1], AD0_MGCE1.shape[2],
                                        AD0_MGCE1.shape[3]))
        AW0_MGCE1 = AW0.view(B, 1, AW0.shape[1], AW0.shape[2])
        AW0_MGCE1 = torch.broadcast_to(AW0_MGCE1,
                                       (B, X.shape[1], AW0_MGCE1.shape[2],
                                        AW0_MGCE1.shape[3]))
        AC0_MGCE1 = AC0_MGCE1.reshape(-1, AC0_MGCE1.shape[2], AC0_MGCE1.shape[3])
        AD0_MGCE1 = AD0_MGCE1.reshape(-1, AD0_MGCE1.shape[2], AD0_MGCE1.shape[3])
        AW0_MGCE1 = AW0_MGCE1.reshape(-1, AW0_MGCE1.shape[2], AW0_MGCE1.shape[3])
        X = X.view(-1, X.shape[2], X.shape[3])
        X = self.fcD11_MGCE1(torch.bmm(AC0_MGCE1, X)) + self.fcD12_MGCE1(torch.bmm(AD0_MGCE1, X)) + self.fcD13_MGCE1(
            torch.bmm(AW0_MGCE1, X))
        X = X.view(B, c, X.shape[1], X.shape[2])
        X = self.dcnn2d2_MGCE1(X)
        X = X + self.cnn2dRes_MGCE1(X0)
        X = F.relu(X)
        X = self.cnn2ddown_MGCE1(X)
        MGCE = X
        return MGCE
    # MGCE

#MGCD
class MGCD(nn.Module):
    def __init__(self, c_in, c_out, feature_num):
        super(MGCD, self).__init__()
        self.selfattentiont1_MGCD = selfattention(c_in)
        self.selfattentiont2_MGCD = selfattention(c_in)

        self.tcnn2ddown_MGCD = nn.Sequential(
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )

        self.fcD11_MGCD = nn.Linear(feature_num, feature_num, bias=True)
        self.fcD12_MGCD = nn.Linear(feature_num, feature_num, bias=True)
        self.fcD13_MGCD = nn.Linear(feature_num, feature_num, bias=True)

        self.cnn2dup_MGCD = nn.Sequential(
            nn.ConvTranspose2d(c_out, c_out, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )

        self.cnn2dRes_MGCD = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=(1, 2), padding=(0, 0), dilation=(1, 2)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )

        self.cnn2dup1_MGCD = nn.Sequential(
            nn.ConvTranspose2d(c_out, c_out, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True)
        )
    def forward(self, MGCE_out,X,AC0,AD0,AW0):
        B = X.shape[0]
        X=self.selfattentiont1_MGCD(X)*torch.sigmoid(self.selfattentiont1_MGCD(MGCE_out))
        X0 = X
        X = self.tcnn2ddown_MGCD1(X)
        AC0_MGCD1 = AC0.view(B, 1, AC0.shape[1], AC0.shape[2])
        AC0_MGCD1 = torch.broadcast_to(AC0_MGCD1,
                                       (B, X.shape[1], AC0_MGCD1.shape[2],
                                        AC0_MGCD1.shape[3]))
        AD0_MGCD1 = AD0.view(B, 1, AD0.shape[1], AD0.shape[2])
        AD0_MGCD1 = torch.broadcast_to(AD0_MGCD1,
                                       (B, X.shape[1], AD0_MGCD1.shape[2],
                                        AD0_MGCD1.shape[3]))
        AW0_MGCD1 = AW0.view(B, 1, AW0.shape[1], AW0.shape[2])
        AW0_MGCD1 = torch.broadcast_to(AW0_MGCD1,
                                       (B, X.shape[1], AW0_MGCD1.shape[2],
                                        AW0_MGCD1.shape[3]))
        AC0_MGCD1 = AC0_MGCD1.reshape(-1, AC0_MGCD1.shape[2], AC0_MGCD1.shape[3])
        AD0_MGCD1 = AD0_MGCD1.reshape(-1, AD0_MGCD1.shape[2], AD0_MGCD1.shape[3])
        AW0_MGCD1 = AW0_MGCD1.reshape(-1, AW0_MGCD1.shape[2], AW0_MGCD1.shape[3])
        X = X.view(-1, X.shape[2], X.shape[3])
        X = self.fcD11_MGCD1(torch.bmm(AC0_MGCD1, X)) + self.fcD12_MGCD1(torch.bmm(AD0_MGCD1, X)) + self.fcD13_MGCD1(
            torch.bmm(AW0_MGCD1, X))
        X = X.view(B, -1, X.shape[1], X.shape[2])
        X = self.cnn2dup_MGCD1(X)
        X = X + self.cnn2dRes_MGCD1(X0)
        X = F.relu(X)
        X = self.cnn2dup1_MGCD1(X)
        MGCD = X
        return MGCD




