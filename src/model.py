import torch.nn as nn
import torch.nn.functional as F 
class CNN(nn.Module):
    def __init__(self , num_classes ) :
        def __init__(self , num_classes): 
            super(CNN , self).__init() 
            #B1 : feature
            self.conv1 = nn.Conv2d(1 , 32 , kernel_size = 3 , padding = 1 )
            