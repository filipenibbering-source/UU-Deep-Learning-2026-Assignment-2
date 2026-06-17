import torch
from torch import nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torch.nn import functional as F

class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=False):
        super(BaselineLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # width of the single context vector produced for the whole sequence
        self.context_size = hidden_size * self.num_directions

        self.lstm_enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        self.fc = nn.Linear(hidden_size * self.num_directions, 4)

    def forward(self, x):
        b, sensors, n_mels, frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, frames, sensors * n_mels)

        out, (last_h_state, last_c_state) = self.lstm_enc(x)
        
        # print(out.shape)
        
        # only keep the last time step
        out = out[:, -1, :]

        out = self.fc(out)
        # print(out.shape)
        # exit()
        return out
    
# written by AI:
def _disable_bn_running_stats(model: nn.Module) -> None:
    """Make every BatchNorm layer use the current batch's statistics in both
    train and eval mode.

    With only ~32 training samples and batch_size=4 the running mean/var that
    BatchNorm accumulates are far too noisy, so eval-mode validation activations
    blow up and the validation loss explodes while the training loss looks fine.
    Disabling running stats removes that train/eval mismatch.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.num_batches_tracked = None


class BaselineResNet18(nn.Module):
    def __init__(self):
        super(BaselineResNet18, self).__init__()

        self.model = resnet18(
            ResNet18_Weights.IMAGENET1K_V1
        )
        # Change the first convolution to accept 1 input channel instead of 3
        self.model.conv1 = nn.Conv2d(
            in_channels=248,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=self.model.conv1.bias is not None
        )
 
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)

        _disable_bn_running_stats(self.model)
        
    def forward(self, x):
        return self.model(x)
    

class BaselineResNet50(nn.Module):
    def __init__(self):
        super(BaselineResNet50, self).__init__()
        self.model = resnet50(
            ResNet50_Weights.IMAGENET1K_V1
        )
        self.model.conv1 = nn.Conv2d(
            in_channels=248,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=self.model.conv1.bias is not None
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)

        _disable_bn_running_stats(self.model)
        
    def forward(self, x):
        return self.model(x)
    

class SeparableConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple | int, padding: tuple | int = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv2d_1x1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv2d_1x1(y)
        return y

class SeparableConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple | int, padding: tuple | int = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv1d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv1d_1x1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv1d_1x1(y)
        return y    

class BaselineEEGNet(nn.Module):
    """
    Taken from https://github.com/s4rduk4r/eegnet_pytorch/blob/main/model/eegnet_pt.py
    """
    def __init__(self, nb_classes: int, Chans: int = 248, Samples: int = 128,
                 dropoutRate: float = 0.5, kernLength: int = 63,
                 F1:int = 8, D:int = 2):
        super().__init__()

        F2 = F1 * D

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        # In: (B, Chans, Samples, 1)
        # Out: (B, F1, Samples, 1)
        self.conv1 = nn.Conv1d(Chans, F1, kernLength, padding=(kernLength // 2))
        self.bn1 = nn.BatchNorm1d(F1) # (B, F1, Samples, 1)
        # In: (B, F1, Samples, 1)
        # Out: (B, F2, Samples - Chans + 1, 1)
        self.conv2 = nn.Conv1d(F1, F2, Chans, groups=F1)
        self.bn2 = nn.BatchNorm1d(F2) # (B, F2, Samples - Chans + 1, 1)
        # In: (B, F2, Samples - Chans + 1, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.avg_pool = nn.AvgPool1d(4)
        self.dropout = nn.Dropout(dropoutRate)

        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.conv3 = SeparableConv1d(F2, F2, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(F2)
        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 32, 1)
        self.avg_pool2 = nn.AvgPool1d(8)
        # In: (B, F2 *  (Samples - Chans + 1) / 32)
        
        fc_input_size = self._get_fc_input_size(torch.ones(1, 248, 64*279))
        
        
        self.fc = nn.Linear(fc_input_size[-1], nb_classes)
        
    def _get_fc_input_size(self, x: torch.Tensor):
        with torch.no_grad():
            # Block 1
            y1 = self.conv1(x)
            #print("conv1: ", y1.shape)
            y1 = self.bn1(y1)
            #print("bn1: ", y1.shape)
            y1 = self.conv2(y1)
            #print("conv2", y1.shape)
            y1 = F.relu(self.bn2(y1))
            #print("bn2", y1.shape)
            y1 = self.avg_pool(y1)
            #print("avg_pool", y1.shape)
            # y1 = self.dropout(y1)
            #print("dropout", y1.shape)

            # Block 2
            y2 = self.conv3(y1)
            #print("conv3", y2.shape)
            y2 = F.relu(self.bn3(y2))
            #print("bn3", y2.shape)
            y2 = self.avg_pool2(y2)
            #print("avg_pool2", y2.shape)
            # y2 = self.dropout(y2)
            #print("dropout", y2.shape)
            y2 = torch.flatten(y2, 1)
            #print("flatten", y2.shape)
            return y2.shape

    def forward(self, x: torch.Tensor):
        
        x = x.reshape(x.size(0), 248, 64*279)
        
        # Block 1
        y1 = self.conv1(x)
        #print("conv1: ", y1.shape)
        y1 = self.bn1(y1)
        #print("bn1: ", y1.shape)
        y1 = self.conv2(y1)
        #print("conv2", y1.shape)
        y1 = F.relu(self.bn2(y1))
        #print("bn2", y1.shape)
        y1 = self.avg_pool(y1)
        #print("avg_pool", y1.shape)
        y1 = self.dropout(y1)
        #print("dropout", y1.shape)

        # Block 2
        y2 = self.conv3(y1)
        #print("conv3", y2.shape)
        y2 = F.relu(self.bn3(y2))
        #print("bn3", y2.shape)
        y2 = self.avg_pool2(y2)
        #print("avg_pool2", y2.shape)
        y2 = self.dropout(y2)
        #print("dropout", y2.shape)
        y2 = torch.flatten(y2, 1)
        #print("flatten", y2.shape)
        y2 = self.fc(y2)
        #print("fc", y2.shape)

        return y2

