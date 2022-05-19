import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
import torchaudio.models as models
from conformer import ConformerBlock, ConformerConvModule
class myTransformer(nn.Module):
    def __init__(self, out_dim=600, embed_dim=256, depth=12,num_heads=8,mlp_ratio=4.,norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed = nn.Linear(40, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.heads = nn.Linear(embed_dim, out_dim)
        # self.softmax = nn.Softmax(dim=-1)


    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def forward(self,x):
        x = self.embed(x)
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = x[:,0, :]
        x = self.heads(x)
        # x = self.softmax(x)
        return x

class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2
        )
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        args:
          mels: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out


class self_Attentive_pooling(nn.Module):
    def __init__(self, dim):
        super(self_Attentive_pooling, self).__init__()
        self.sap_linaer = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.FloatTensor(dim,1))
        torch.nn.init.normal_(self.attention, std=.02)
        print(1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = torch.tanh(self.sap_linaer(x))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        x = torch.sum(x * w, dim=1)
        return x

# class AMSoftmax(nn.Module):
#
#     '''
#     Additve Margin Softmax as proposed in:
#     https://arxiv.org/pdf/1801.05599.pdf
#     Implementation Extracted From
#     https://github.com/clovaai/voxceleb_trainer/blob/master/loss/cosface.py
#     '''
#
#     def __init__(self, in_feats, n_classes, m=0.3, s=15, annealing=False):
#         super(AMSoftmax, self).__init__()
#         self.in_feats = in_feats
#         self.m = m
#         self.s = s
#         self.annealing = annealing
#         self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
#         nn.init.xavier_normal_(self.W, gain=1)
#         self.annealing=annealing
#
#     def getAnnealedFactor(self,step):
#         alpha = self.__getAlpha(step) if self.annealing else 0.
#         return 1/(1+alpha)
#
#     def __getAlpha(self,step):
#         return max(0, 1000./(pow(1.+0.0001*float(step),2.)))
#
#     def __getCombinedCosth(self, costh, costh_m, step):
#
#         alpha = self.__getAlpha(step) if self.annealing else 0.
#         costh_combined = costh_m + alpha*costh
#         return costh_combined/(1+alpha)
#
#     def forward(self, x, label=None, step=0):
#         assert x.size()[0] == label.size()[0]
#         assert x.size()[1] == self.in_feats
#         x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         x_norm = torch.div(x, x_norm)
#         w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
#         w_norm = torch.div(self.W, w_norm)
#         costh = torch.mm(x_norm, w_norm)
#         label_view = label.view(-1, 1)
#         if label_view.is_cuda: label_view = label_view.cpu()
#         delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
#         if x.is_cuda: delt_costh = delt_costh.cuda()
#         costh_m = costh - delt_costh
#         costh_combined = self.__getCombinedCosth(costh, costh_m, step)
#         costh_m_s = self.s * costh_combined
#         return costh, costh_m_s


class AMSoftmax(nn.Module):
    def __init__(self, in_feats, n_classes, m=0.3, s=15, annealing=False):
        super(AMSoftmax, self).__init__()
        self.linaer = nn.Linear(in_feats, n_classes, bias=False)
        self.m = m
        self.s = s

    def _am_logsumexp(self, logits):
        max_x = torch.max(logits, dim=-1)[0].unsqueeze(-1)
        term1 = (self.s*(logits - (max_x + self.m))).exp()
        term2 = (self.s * (logits - max_x)).exp().sum(-1).unsqueeze(-1) - (self.s*(logits-max_x)).exp()
        return self.s * max_x + (term2 + term1).log()

    def forward(self, *inputs):

        x_vector = F.normalize(inputs[0], p=2, dim=-1)
        self.linaer.weight.data = F.normalize(self.linaer.weight.data, p=2,dim=-1)
        logits = self.linaer(x_vector)
        scaled_logits = (logits-self.m)*self.s
        return scaled_logits - self._am_logsumexp(logits)


class FocalSoftmax(nn.Module):
    '''
    Focal softmax as proposed in:
    "Focal Loss for Dense Object Detection"
    by T-Y. Lin et al.
    https://github.com/foamliu/InsightFace-v2/blob/master/focal_loss.py
    '''
    def __init__(self, gamma=2):
        super(FocalSoftmax, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class Classifier2(nn.Module):
    def __init__(self, d_model=512, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        # self.encoder_layer = ConformerBlock(
        #     dim=512,
        #     dim_head=64,
        #     heads=8,
        #     ff_mult=4,
        #     conv_expansion_factor=2,
        #     conv_kernel_size=31,
        #     attn_dropout=0.,
        #     ff_dropout=0.,
        #     conv_dropout=0.
        # )
        self.blocks = nn.ModuleList([
            ConformerBlock(
                dim=d_model,
                dim_head=64,
                heads=8,
                ff_mult=4,
                conv_expansion_factor=2,
                conv_kernel_size=31,
                attn_dropout=dropout,
                ff_dropout=dropout,
                conv_dropout=dropout
            )
            for i in range(1)])

        # self.encoder_layer = models.Conformer(
        #     input_dim=d_model, ffn_dim=800, num_heads=8, dropout = dropout, depthwise_conv_kernel_size=15, num_layers=8
        # )

        # Project the the dimension of features from d_model into speaker nums.
        # self.pred_layer = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, n_spks),
        # )
        self.pooling = self_Attentive_pooling(d_model)
        self.pred_layer = AMSoftmax(d_model, n_spks)
    def forward(self, mels):
        """
        args:
          mels: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        out = out.permute(1,0,2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        for blk in self.blocks:
            out = blk(out)
        # out = self.encoder_layer(out,torch.tensor([out.shape[1]]*out.shape[0]).to('cuda'))
        # out,_=out
        # mean pooling
        out = out.permute(1,2,0)
        stats = self.pooling(out)
        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out

def model_Datapara(model, device,  pre_path=None):
    model = torch.nn.DataParallel(model).to(device)
    if pre_path != None:
        model_dict = torch.load(pre_path).module.state_dict()
        model.module.load_state_dict(model_dict)
    return model

def load_model(model, device,  pre_path=None, para=False):
    # model = torch.nn.DataParallel(model).to(device)
    if pre_path != None:
        if para:
            model_dict = torch.load(pre_path, map_location='cpu').module.state_dict()
            model_dict['fc.0.weight'] = model_dict['fc.weight']
            model_dict['fc.0.bias'] = model_dict['fc.bias']
            del model_dict['fc.weight']
            del model_dict['fc.bias']
        else:
            model_dict = torch.load(pre_path, map_location='cpu')
        model.load_state_dict(model_dict, strict=True)
    model = model.to(device)
    return model


def prilearn_para(model_ft,feature_extract):
    # 将模型发送到GPU
    device = torch.device("cuda:0")
    model_ft = model_ft.to(device)

    # 在此运行中收集要优化/更新的参数。
    # 如果我们正在进行微调，我们将更新所有参数。
    # 但如果我们正在进行特征提取方法，我们只会更新刚刚初始化的参数，即`requires_grad`的参数为True。
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    #
    # # 观察所有参数都在优化
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)




def init_para(model):
    def weights_init(model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
    model.apply(weights_init)
    return model

def weight_decay(model,optimizer, lr, wd, eps=1e-8):
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    optimizer = optimizer(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=lr,
        betas=(1-wd, 0.999),
        eps=eps,
    )
    return optimizer


import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

class Classifier3(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()


        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = models.Conformer(
            input_dim=d_model, ffn_dim=800, num_heads=8, dropout = dropout, depthwise_conv_kernel_size=15, num_layers=8
        )
        # self.encoder_layer =   nn.ModuleList([
        #     ConformerBlock(
        #         dim=d_model,
        #         dim_head=64,
        #         heads=8,
        #         ff_mult=4,
        #         conv_expansion_factor=2,
        #         conv_kernel_size=31,
        #         attn_dropout=dropout,
        #         ff_dropout=dropout,
        #         conv_dropout=dropout
        #     )
        #     for i in range(8)])


        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm([d_model]),
            weight_norm(nn.Linear(d_model, n_spks)),
        )

        self.attention_scores = nn.Sequential(
            nn.Linear(d_model,1),
            nn.Softmax(dim=1),
        )

    def forward(self, mels):
        """
        args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # for blk in self.encoder_layer:
        #     out = blk(out)

        out = self.encoder_layer(out, torch.tensor([out.shape[1]]*out.shape[0]).to('cuda'))
        # out: (batch size, length, d_model)
        out,_=out


        #out = out.transpose(0, 1)
        # mean pooling
        #stats = out.mean(dim=1)

        # batch, 1, lenth
        scores = self.attention_scores(out).permute(0,2,1)
        out = scores @ out
        #batch, d_model
        out = out.squeeze(1)


        # out: (batch, n_spks)
        out = self.pred_layer(out)
        #out = (out-m)/t

        return out








