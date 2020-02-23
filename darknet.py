from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 



def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  #img是【h,w,channel】，这里的img[:,:,::-1]是将第三个维度channel从opencv的BGR转化为pytorch的RGB，然后transpose((2,0,1))的意思是将[height,width,channel]->[channel,height,width]
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    我们定义一个函数 parse_cfg，该函数使用配置文件的路径作为输入。
    Takes a configuration file
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    输入: 配置文件路径
    返回值: 列表对象,其中每一个元素为一个字典类型对应于一个要建立的神经网络模块（层）
    
    这里的思路是解析 cfg，将每个块存储为词典。这些块的属性和值都以键值对的形式存储在词典中。
    解析过程中，我们将这些词典（由代码中的变量 block 表示）添加到列表 blocks 中。我们的函数将返回该 block。
    """
    
    # 将配置文件内容保存在字符串列表中
    # 加载文件并过滤掉文本中多余内容
    file = open('cfg\yolov3.cfg', 'r',encoding='utf-8')
    lines = file.read().split('\n')             # store the lines in a list等价于readlines
    lines = [x for x in lines if len(x) > 0]    # 去掉空行
    lines = [x for x in lines if x[0] != '#']    # 去掉以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]    # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)
    # 遍历预处理后的列表，得到块
    # cfg文件中的每个块用[]括起来最后组成一个列表，一个block存储一个块的内容，即每个层用一个字典block存储。

    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # 这是cfg文件中一个层(块)的开始           
            if len(block) != 0:          # 如果块内已经存了信息, 说明是上一个块的信息还没有保存
                blocks.append(block)     # 那么这个块（字典）加入到blocks列表中去
                block = {}               # 覆盖掉已存储的block,新建一个空白块存储描述下一个块的信息(block是字典)
            block["type"] = line[1:-1].rstrip()  # 把cfg的[]中的块名作为键type的值   
        else:
            key,value = line.split("=") #按等号分割
            block[key.rstrip()] = value.lstrip()#左边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对
    blocks.append(block) # 退出循环，将最后一个未加入的block加进去
    print('\n\n'.join([repr(x) for x in blocks]))
    return blocks
 
# 配置文件定义了6种不同type
# 'net': 相当于超参数,网络全局配置的相关参数
# {'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}
 
# cfg = parse_cfg("cfg/yolov3.cfg")
# print(cfg)


class EmptyLayer(nn.Module):
    """
    对于在 Route 模块中设计一个层，我们必须建立一个 nn.Module 对象，其作为 layers 的成员被初始化。
    然后，我们可以写下代码，将 forward 函数中的特征图拼接起来并向前馈送。
    最后，我们执行网络的某个 forward 函数的这个层。
    
    但拼接操作的代码相当地短和简单（在特征图上调用 torch.cat），像上述过程那样设计一个层将导致不必要的抽象，增加样板代码。
    取而代之，我们可以将一个假的层置于之前提出的路由层的位置上，
    然后直接在代表 darknet 的 nn.Module 对象的 forward 函数中执行拼接运算。
    
    为shortcut layer / route layer 准备, 具体功能不在此实现，在Darknet类的forward函数中有体现
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    """
    yolo 检测层的具体实现, 在特征图上使用锚点预测目标区域和类别, 功能函数在predict_transform中
    新的层 DetectionLayer 保存用于检测边界框的锚点
    """
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


"""
现在我们将使用上面 parse_cfg 返回的列表来构建 PyTorch 模块，作为配置文件中的构建块。
列表中有 5 种类型的层。PyTorch 为 convolutional 和 upsample 提供预置层。
我们将通过扩展 nn.Module 类为其余层写自己的模块。
create_modules 函数使用 parse_cfg 函数返回的 blocks 列表：
"""

def create_modules(blocks):
    # 迭代该列表之前，我们先定义变量 net_info，来存储该网络的信息
    net_info = blocks[0]     # blocks[0]存储了cfg中[net]的信息，它是一个字典，获取网络输入和预处理相关信息    
    module_list = nn.ModuleList() # module_list用于存储每个block,每个block对应cfg文件中一个块，类似[convolutional]里面就对应一个卷积块
    prev_filters = 3   #初始值对应于输入数据3通道，用来存储我们需要持续追踪被应用卷积层的卷积核数量（上一层的卷积核数量（或特征图深度））
    output_filters = []   #我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。
    
    # 迭代模块的列表，并为每个模块创建一个 PyTorch 模块
    for index, x in enumerate(blocks[1:]): #这里，我们迭代block[1:] 而不是blocks，因为blocks的第一个元素是一个net块，它不属于前向传播。
        # nn.Sequential 类被用于按顺序地执行 nn.Module 对象的一个数字。
        # 如果你查看 cfg 文件，你会发现，一个模块可能包含多于一个层。
        # 例如，一个 convolutional 类型的模块有一个批量归一化层、一个 leaky ReLU 激活层以及一个卷积层。
        # 我们使用 nn.Sequential 将这些层串联起来，得到 add_module 函数。
        module = nn.Sequential()# 这里每个块用nn.sequential()创建为了一个module,一个module有多个层
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # 1. 卷积层
            # Get the info about the layer
            # 获取激活函数/批归一化/卷积层参数（通过字典的键获取值）
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False # 卷积层后接BN就不需要bias
            except:
                batch_normalize = 0
                bias = True # 卷积层后无BN层就需要bias
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            # 开始创建并添加相应层
            # Add the convolutional layer
            # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            # Check the activation. 
            # It is either Linear or a Leaky ReLU for YOLO
            # 给定参数负轴系数0.1
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
            
        elif (x["type"] == "upsample"):
            """
            2. upsampling layer
            没有使用 Bilinear2dUpsampling
            实际使用的为最近邻插值
            """
            stride = int(x["stride"]) # 这个stride在cfg中就是2，所以下面的scale_factor写2或者stride是等价的
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        # If it is a route layer
        # route layer -> Empty layer
        # route层的作用：当layer取值为正时，输出这个正数对应的层的特征，如果layer取值为负数，输出route层向后退layer层对应层的特征
        elif (x["type"] == "route"):
            # 首先，我们提取关于层属性的值，将其表示为一个整数，并保存在一个列表中。
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation : 正值
            if start > 0: 
                start = start - index
            if end > 0: # 若end>0，由于end= end - index，再执行index + end输出的还是第end层的特征
                end = end - index
                
            # 然后我们得到一个新的称为 EmptyLayer 的层，顾名思义，就是空的层。
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            
            # 在路由层之后的卷积层会把它的卷积核应用到之前层的特征图（可能是拼接的）上。
            # 以下的代码更新了 filters 变量以保存路由层输出的卷积核数量。
            if end < 0: #若end<0，则end还是end，输出index+end(而end<0)故index向后退end层的特征。
                filters = output_filters[index + start] + output_filters[index + end]
            else: 
                #如果没有第二个参数，end=0，则对应下面的公式，
                # 此时若start>0，由于start = start - index，
                # 再执行index + start输出的还是第start层的特征;
                # 若start<0，则start还是start，
                # 输出index+start(而start<0)故index向后退start层的特征。
                filters= output_filters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            # 使用空的层，因为它还要执行一个非常简单的操作（加）。
            # 没必要更新 filters 变量,因为它只是将前一层的特征图添加到后面的层上而已。
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            # 锚点,检测,位置回归,分类，这个类见predict_transform中
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
                              
        # 在这个回路结束时，我们做了一些统计（bookkeeping.）。
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        # 如前所述，我们使用 nn.Module 在 PyTorch 中构建自定义架构。
        # 这里，我们可以为检测器定义一个网络。在 darknet.py 文件中，我们添加了以下类别：
        # 这里，我们对 nn.Module 类别进行子分类，并将我们的类别命名为 Darknet。
        # 我们用 members、blocks、net_info 和 module_list 对网络进行初始化。
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile) # 调用parse_cfg函数
        self.net_info, self.module_list = create_modules(self.blocks) # 调用create_modules函数
        
    def forward(self, x, CUDA):
        # 该网络的前向传播通过覆写 nn.Module 类别的 forward 方法而实现。
        # forward 主要有两个目的。
        # 一，计算输出；
        # 二，尽早处理的方式转换输出检测特征图（例如转换之后，这些不同尺度的检测图就能够串联，不然会因为不同维度不可能实现串联）。
        
        # 这里，我们迭代 self.block[1:] 而不是 self.blocks，因为 self.blocks 的第一个元素是一个 net 块，它不属于前向传播。
        # 由于路由层和捷径层需要之前层的输出特征图，我们在字典 outputs 中缓存每个层的输出特征图。关键在于层的索引，且值对应特征图。
        # 正如 create_module 函数中的案例，我们现在迭代 module_list，它包含了网络的模块。
        # 需要注意的是这些模块是以在配置文件中相同的顺序添加的。这意味着，我们可以简单地让输入通过每个模块来得到输出。
        modules = self.blocks[1:] # 除了net块之外的所有，forward这里用的是blocks列表中的各个block块字典
        outputs = {}   #We cache the outputs for the route layer
        
        # write表示我们是否遇到第一个检测。
        # write=0，则收集器尚未初始化，
        # write=1，则收集器已经初始化，
        # 我们只需要将检测图与收集器级联起来即可。
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    print(layers)
                    layers[0] = layers[0] - i
                    
                # 如果只有一层时。从前面的if (layers[0]) > 0:语句中可知，
                # 如果layer[0]>0，则输出的就是当前layer[0]这一层的特征,
                # 如果layer[0]<0，输出就是从route层(第i层)向后退layer[0]层那一层得到的特征 
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                # 第二个元素同理
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    #第二个参数设为 1,这是因为我们希望将特征图沿anchor数量的维度级联起来。
                    x = torch.cat((map1, map2), 1)
                
    
            elif module_type == "shortcut":
                from_ = int(module["from"])
                # 求和运算，它只是将前一层的特征图添加到后面的层上而已
                x = outputs[i-1] + outputs[i+from_]
                """
                YOLO 的输出是一个卷积特征图，包含沿特征图深度的边界框属性。
                边界框属性由彼此堆叠的单元格预测得出。
                因此，如果你需要在 (5,6) 处访问单元格的第二个边框，那么你需要通过 map[5,6, (5+C): 2*(5+C)] 将其编入索引。
                这种格式对于输出处理过程（例如通过目标置信度进行阈值处理、添加对中心的网格偏移、应用锚点等）很不方便。
                
                另一个问题是由于检测是在三个尺度上进行的，预测图的维度将是不同的。
                虽然三个特征图的维度不同，但对它们执行的输出处理过程是相似的。
                如果能在单个张量而不是三个单独张量上执行这些运算，就太好了。
                为了解决这些问题，我们引入了函数 predict_transform。
                """
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #从net_info(实际就是blocks[0]，即[net])中get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data # 这里得到的是预测的yolo层feature map
                # 在util.py中的predict_transform()函数利用x(是传入yolo层的feature map)，得到每个格子所对应的anchor最终得到的目标
                # 坐标与宽高，以及出现目标的得分与每种类别的得分。经过predict_transform变换后的x的维度是(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)
                # 函数 predict_transform 在文件 util.py 中，我们在 Darknet 类别的 forward 中使用该函数时，将导入该函数。
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                 
                if not write:              #if no collector has been intialised. 因为一个空的tensor无法与一个有数据的tensor进行concatenate操作，
                    detections = x #所以detections的初始化在有预测值出来时才进行，
                    write = 1   #用write = 1标记，当后面的分数出来后，直接concatenate操作即可。
        
                else:  
                    """
                    变换后x的维度是(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)，
                    这里是在维度1上进行concatenate，即按照anchor数量的维度进行连接，
                    对应教程part3中的Bounding Box attributes图的行进行连接。
                    yolov3中有3个yolo层，所以对于每个yolo层的输出,
                    先用predict_transform()变成每行为一个anchor对应的预测值的形式(不看batch_size这个维度，x剩下的
                    维度可以看成一个二维tensor)，这样3个yolo层的预测值按照每个方框对应的行的维度进行连接。
                    得到了这张图处所有anchor的预测值，后面的NMS等操作可以一次完成
                    """
                    detections = torch.cat((detections, x), 1)# 将在3个不同level的feature map上检测结果存储在 detections 里
        
            outputs[i] = x
        
        return detections
# blocks = parse_cfg('cfg/yolov3.cfg')
# x,y = create_modules(blocks)
# print(y)


    def load_weights(self, weightfile):
        # 我们写一个函数来加载权重，它是 Darknet 类的成员函数。它使用 self 以外的一个参数作为权重文件的路径。
        #Open the weights file
        fp = open(weightfile, "r",encoding = 'utf-8')
        # 第一个 160 比特的权重文件保存了 5 个 int32 值，它们构成了文件的标头。
        # The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        
        # 这里读取first 5 values权重
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        # 加载 np.ndarray 中的剩余权重，之后的比特代表权重，按上述顺序排列。权重是以float32类型存储的
        weights = np.fromfile(fp, dtype = np.float32)
        
        # 现在，我们迭代地加载权重文件到网络的模块上
        ptr = 0
        for i in range(len(self.module_list)):
            # blocks中的第一个元素是网络参数和图像的描述，所以从blocks[1]开始读入
            module_type = self.blocks[i + 1]["type"]
    
            # If module_type is convolutional load weights
            # Otherwise ignore.
            # 在循环过程中，我们首先检查 convolutional 模块是否有 batch_normalize（True）。
            # 基于此，我们加载权重。
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    # 当有bn层时，"batch_normalize"对应值为1
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                # 我们保持一个称为 ptr 的变量来追踪我们在权重数组中的位置。
                # 现在，如果 batch_normalize 检查结果是 True，则我们按以下方式加载权重：
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else: #如果 batch_normalize 的检查结果不是 True，只需要加载卷积层的偏置项
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

"""
通过模型构建和权重加载，我们终于可以开始进行目标检测了。
未来，我们还将介绍如何利用 objectness 置信度阈值和非极大值抑制生成最终的检测结果。

总的来说，darknet.py程序包含
函数parse_cfg输入: 配置文件路径返回一个列表,其中每一个元素为一个字典类型对应于一个要建立的神经网络模块（层）;
函数create_modules: 创建网络层级，
Darknet类的forward函数:实现网络前向传播函数了
load_weights: 导入预训练的网络权重参数
forward函数:产生需要的预测输出形式，因此需要变换输出即函数 predict_transform 在文件 util.py 中，
我们在 Darknet 类别的 forward 中使用该函数时，将导入该函数。
"""

