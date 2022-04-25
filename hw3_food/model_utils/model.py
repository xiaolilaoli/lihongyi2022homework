import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
import torchvision.models as models
from models.backbones import resnet18_cifar_variant1, resnet18_cifar_variant2

class myTransformer(nn.Module):
    def __init__(self, seq_len=11,seq_embed=39 ,out_dim=39, embed_dim=256,depth=12,num_heads=8,mlp_ratio=4.,norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed = nn.Linear(seq_embed, embed_dim)
        self.norm = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.sinusoid_table = self.get_sinusoid_table(seq_len+1, embed_dim) # (seq_len+1, d_model)

        # layers
        self.embedding = nn.Embedding(seq_len, embed_dim)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.heads = nn.Linear(embed_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)


    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)

        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)

    def forward(self,x):
        positions = torch.arange(x.size(1), device=x.device, dtype=torch.int64).repeat(x.size(0), 1) + 1
        cls_positions = torch.arange(1, device=x.device, dtype=torch.int64).repeat(1, 1)
        x = self.embed(x)
        x = x + self.pos_embedding(positions)
        cls_token = self.cls_token + self.pos_embedding(cls_positions)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = x[:,0, :]
        x = self.heads(x)
        x = self.softmax(x)
        return x


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class MyModel(nn.Module):
    def __init__(self,numclass = 2):
        super(MyModel, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #64*64
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #32*32
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #16*16
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #8*8
        self.pool1 = nn.MaxPool2d(2)#4*4
        self.fc = nn.Linear(8192, 512)
        # self.drop = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, numclass)
    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool1(x)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        # x = self.drop(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

def model_Datapara(model, device,  pre_path=None):
    model = torch.nn.DataParallel(model).to(device)

    model_dict = torch.load(pre_path).module.state_dict()
    model.module.load_state_dict(model_dict)
    return model


def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

def load_backBone(model, prepath, is_dict=False):
    save_dict = torch.load(prepath, map_location='cpu')
    if is_dict:
        new_state_dict = save_dict
    else:
        new_state_dict = {'state_dict':save_dict.module.state_dict()}
    msg = model.load_state_dict({k[9:]:v for k, v in new_state_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
    return model

class SimModel(nn.Module):
    def __init__(self,pre_path, train_backB=True, num_classes=11):
        super(SimModel, self).__init__()
        save_dict = torch.load(pre_path, map_location='cpu')
        resModel = get_backbone('resnet18_cifar_variant1')

        # new_state_dict = {'state_dict':save_dict.module.state_dict()}
        new_state_dict = save_dict
        msg = resModel.load_state_dict({k[9:]:v for k, v in new_state_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
        self.backBone = resModel

        classifier = nn.Sequential(
            nn.Linear(in_features=resModel.output_dim, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(768, num_classes, True)
        )
        self.classfier = classifier
        if not train_backB:
            for param in self.backBone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backBone(x)
        x = self.classfier(x)
        return x

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 初始化将在此if语句中设置的这些变量。
    # 每个变量都是模型特定的。
    model_ft = None
    input_size = 0
    if model_name =="MyModel":
        if use_pretrained == True:
            model_ft = torch.load('model_save/MyModel')
        else:
            model_ft = MyModel(num_classes)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name =="myTrans":
        if use_pretrained == True:
            model_ft = torch.load('model_save/myTrans')
        else:
            model_ft = myTransformer(num_classes)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "googlenet":
        """ googlenet
        """
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "alexnet":
        """ Alexnet
 """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
 """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
 """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
 """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
 Be careful, expects (299,299) sized images and has auxiliary output
 """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # 处理辅助网络
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # 处理主要网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model_utils name, exiting...")
        exit()

    return model_ft, input_size

# # 在这步中初始化模型
#
#
# # 打印我们刚刚实例化的模型
# print(model_ft)
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











