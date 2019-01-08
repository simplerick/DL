from torchvision.models.inception import*
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo



__all__ = ['Inception_ft', 'inception_ft']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def inception_ft(pretrained=False,**kwargs):
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception_ft(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']),strict=False)
        return model
    return Inception_ft(**kwargs)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, s = 1):
        super(ConvBlock, self).__init__()
        self.Conv = nn.Conv2d(in_c,out_c,kernel_size=kernel,padding=int(kernel[0]/2), stride=s)
        self.BatchNorm = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.Conv(x)
        x =  self.BatchNorm(x)
        x = F.relu(x)
        return x


class Inception_ft(Inception3):
    def __init__(self, num_classes = 2, aux_logits=False, transform_input=False):
        super(Inception_ft, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        del self.fc
        del self.Mixed_7b
        del self.Mixed_7c
        self.Conv_block_8a = ConvBlock(1280,1280,(3,3),2)
        self.Conv_block_8b = ConvBlock(1280,2048,(3,3),2)
        self.Linear = nn.Linear(2048, num_classes)


    def forward(self,x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Conv_block_8a(x)
        # 4 x 4 x 1280
        x = F.dropout(x, training=self.training)
        # 4 x 4 x 1280
        x = self.Conv_block_8b(x)
        # 2 x 2 x 2048
        x = F.avg_pool2d(x, kernel_size=2)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.Linear(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x
