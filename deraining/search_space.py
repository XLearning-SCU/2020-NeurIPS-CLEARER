import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# parallel module
class HorizontalModule(nn.Module):
    def __init__(self, num_inchannels, num_blocks=None):
        self.num_branches = len(num_inchannels)
        assert self.num_branches > 1
        if num_blocks is None:
            num_blocks = [1 for _ in range(self.num_branches)]
        else:
            assert self.num_branches==len(num_blocks)
        
        super(HorizontalModule, self).__init__()
        
        self.branches = nn.ModuleList()
        for i in range(self.num_branches):
            layers=[]
            for _ in range(num_blocks[i]):
                layers.append(ResidualBlock(num_inchannels[i], num_inchannels[i]))
            self.branches.append(nn.Sequential(*layers))
    
    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        return x

class FusionModule(nn.Module):
    def __init__(self, num_inchannels, multi_scale_output=True):
        super(FusionModule, self).__init__()
        self.num_branches = len(num_inchannels)
        self.multi_scale_output = multi_scale_output
        assert self.num_branches > 1

        self.relu = nn.ReLU(True)
        self.fuse_layers = nn.ModuleList()
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    # Unsample
                    fuse_layer.append(nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        ))
                elif j == i:
                    # identity
                    fuse_layer.append(None)
                else:
                    # downsample
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                ))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            self.fuse_layers.append(nn.ModuleList(fuse_layer))
    
    def forward(self, x):
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class TransitionModule(nn.Module):
    def __init__(self, num_channels_pre_layer, num_channels_cur_layer):
        super(TransitionModule, self).__init__()

        self.num_branches_cur = len(num_channels_cur_layer)
        self.num_branches_pre = len(num_channels_pre_layer)

        self.transition_layers = nn.ModuleList()
        for i in range(self.num_branches_cur):
            if i < self.num_branches_pre:
                # horizontal transition
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    # alter channels
                    self.transition_layers.append(nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        ))
                else:
                    # no operation
                    self.transition_layers.append(None)
            else:
                # downsample transition (new scales)
                conv3x3s = []
                for j in range(i+1-self.num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i-self.num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        ))
                self.transition_layers.append(nn.Sequential(*conv3x3s))

    def forward(self, x):
        x_list = []
        for i in range(self.num_branches_cur):
            if self.transition_layers[i] is not None:
                if i < self.num_branches_pre:
                    x_list.append(self.transition_layers[i](x[i]))
                else:
                    x_list.append(self.transition_layers[i](x[-1]))
            else:
                x_list.append(x[i])
        return x_list

class MixModule(nn.Module):
    def __init__(self, num_inchannels):
        super(MixModule, self).__init__()
        self.horizontal = HorizontalModule(num_inchannels)
        self.fuse = FusionModule(num_inchannels)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, weights):
        h_x = self.horizontal(x)
        f_x = self.fuse(x)

        weights = self.softmax(weights)

        x_list = []
        for i in range(len(x)):
            x_list.append(weights[0]*h_x[i]+weights[1]*f_x[i])
        
        return x_list


class SearchSpace(nn.Module):
    # list_channels of branches; list_modules of stages
    def __init__(self, in_channel=3, list_channels=[32, 64, 128, 256], list_modules=[2,4,4,4]):
        super(SearchSpace,self).__init__()
        self.list_channels = list_channels
        self.list_modules = list_modules

        self.stage1 = self.__make_layer(in_channel, list_channels[0], list_modules[0])

        self.transition1 = TransitionModule([list_channels[0]*Bottleneck.expansion],list_channels[:2])
        self.stage2 = self.__make_stage(list_channels[:2],list_modules[1])

        self.transition2 = TransitionModule(list_channels[:2],list_channels[:3])
        self.stage3 = self.__make_stage(list_channels[:3],list_modules[2])

        self.transition3 = TransitionModule(list_channels[:3],list_channels[:4])
        self.stage4 = self.__make_stage(list_channels[:4],list_modules[3])

        self.final_fuse = FusionModule(list_channels, multi_scale_output=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(list_channels[0], in_channel, 1, 1, bias=False)
        )

        self.__init_architecture()
        self.init_weights()

        self.MSELoss = torch.nn.MSELoss().cuda()


    def __make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers.append(Bottleneck(inplanes, planes, stride, downsample))
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))
        return nn.Sequential(*layers)
    
    def __make_stage(self, num_inchannels, num_module):
        modules = nn.ModuleList()
        for _ in range(num_module):
            modules.append(MixModule(num_inchannels))
        return modules
    
    def new(self):
        model_new = SearchSpace().cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def __init_architecture(self):
        self.arch_param = torch.randn(sum(self.list_modules[1:]), 2, dtype=torch.float).cuda()*1e-3
        self.arch_param.requires_grad = True
    
    def arch_parameters(self):
        return [self.arch_param]

    def init_weights(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)

        x_list = self.transition1([x])
        idx = 0
        for i in range(self.list_modules[1]):
            x_list = self.stage2[i](x_list, self.arch_param[idx+i])
            #print([idx+i, self.arch_param[idx+i]])
        
        x_list = self.transition2(x_list)
        idx = idx + self.list_modules[1]
        for i in range(self.list_modules[2]):
            x_list = self.stage3[i](x_list,self.arch_param[idx+i])
            #print([idx+i, self.arch_param[idx+i]])

        x_list = self.transition3(x_list)
        idx = idx + self.list_modules[2]
        for i in range(self.list_modules[3]):
            x_list = self.stage4[i](x_list,self.arch_param[idx+i])
            #print([idx+i, self.arch_param[idx+i]])

        x = self.final_fuse(x_list)[0]
        x = self.final_conv(x)
        return x
    
    def loss(self, input, target):
        logits = self(input)
        mse_loss = self.MSELoss(logits, target)
        arch_weights = torch.nn.functional.softmax(self.arch_param, dim=1)

        # regular_loss
        regular_loss = -arch_weights*torch.log10(arch_weights)-(1-arch_weights)*torch.log10(1-arch_weights)
        # latency loss: computing the average complexity (params num) of modules by running `python search_space.py`
        # latency_loss = torch.mean(arch_weights[:,0]*0.7+arch_weights[:,1]*0.3)

        return  mse_loss + regular_loss.mean()*0.01 #+ latency_loss*0.1

# multi-resolution network
class HighResolutionNet(nn.Module):
    def __init__(self, ckt_path, in_channel=3, list_channels=[32, 64, 128, 256], list_modules=[2,4,4,4]):
        super(HighResolutionNet, self).__init__()
        self.list_channels = list_channels
        self.list_modules = list_modules

        arch = torch.load(ckt_path)['arch_param']
        softmax = nn.Softmax(dim=1)
        arch_soft = softmax(arch)
        arch_hard = torch.argmax(arch_soft, dim=1)

        self.stage1 = self.__make_layer(in_channel, list_channels[0], list_modules[0])

        self.transition1 = TransitionModule([list_channels[0]*Bottleneck.expansion],list_channels[:2])
        idx = 0
        self.stage2 = self.__make_stage(arch_hard[idx:list_modules[1]], list_channels[:2], list_modules[1])

        self.transition2 = TransitionModule(list_channels[:2],list_channels[:3])
        idx += list_modules[1]
        self.stage3 = self.__make_stage(arch_hard[idx:idx+list_modules[2]], list_channels[:3],list_modules[2])

        self.transition3 = TransitionModule(list_channels[:3],list_channels[:4])
        idx += list_modules[2]
        self.stage4 = self.__make_stage(arch_hard[idx:idx+list_modules[3]], list_channels[:4],list_modules[3])

        self.final_fuse = FusionModule(list_channels, multi_scale_output=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(list_channels[0], in_channel, 1, 1, bias=False)
        )

        self.init_weights()

    def __make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers.append(Bottleneck(inplanes, planes, stride, downsample))
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))
        return nn.Sequential(*layers)
    
    def __make_stage(self, arch, num_inchannels, num_module):
        modules = []
        for i in range(num_module):
            if arch[i] == 0:
                modules.append(HorizontalModule(num_inchannels))
            elif arch[i] == 1:
                modules.append(FusionModule(num_inchannels))
        return nn.Sequential(*modules)

    def init_weights(self):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stage1(x)
        x_list = self.transition1([x])
        x_list = self.stage2(x_list)
        x_list = self.transition2(x_list)
        x_list = self.stage3(x_list)
        x_list = self.transition3(x_list)
        x_list = self.stage4(x_list)
        x = self.final_fuse(x_list)[0]
        x = self.final_conv(x)
        return x
    
if __name__=='__main__':
    '''
    Computing the average complexity (params) of modules.
    '''
    from utils import count_parameters_in_MB
    import numpy as np
    list_channels=[32, 64, 128, 256]
    run_time_ratio = []
    params = np.zeros((2,3))
    for i in range(1, len(list_channels)):
        H = HorizontalModule(list_channels[:i+1]).cuda()
        F = FusionModule(list_channels[:i+1]).cuda()
        params[0,i-1] = count_parameters_in_MB(H) # parallel module
        params[1,i-1] = count_parameters_in_MB(F) # fusion module
    comp = np.sum(params, axis=1)/np.sum(params)
    print('params ratio: %.1f %.1f' % (comp[0], comp[1]))
