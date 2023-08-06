import torch
import torch.nn as nn
from layers import *


# Check in 2022-1-4
class ACMVH(nn.Module):
    def __init__(self, args):
        super(GMMH, self).__init__()
        self.image_dim = args.image_dim
        self.text_dim = args.text_dim

        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.common_dim = args.img_hidden_dim[-1]
        self.nbit = int(args.nbit)
        self.classes = args.classes
        
        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]

        self.dropout = args.dropout

        self.imageMLP = MLP(hidden_dim=self.img_hidden_dim, act=nn.Tanh())

        self.textMLP = MLP(hidden_dim=self.txt_hidden_dim, act=nn.Tanh())
       
        self.ifeat_gate = nn.Sequential(
            nn.Linear(self.common_dim,self.common_dim),
            nn.Sigmoid())

        self.tfeat_gate = nn.Sequential(
            nn.Linear(self.common_dim,self.common_dim),
            nn.Sigmoid())

        params = torch.ones(2, requires_grad=True)

        self.params = torch.nn.Parameter(params)

       
        
        self.neck = nn.Sequential(
            nn.Linear(self.common_dim,self.common_dim*4),
            nn.ReLU(),
            nn.Dropout(0.1)
            nn.Linear(self.common_dim*4,self.common_dim)
        )

        self.hash_output = nn.Sequential(
            nn.Linear(self.common_dim, self.nbit),
            nn.Tanh()
        )
        self.classify = nn.Linear(self.nbit, self.classes)

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, image, text, tgt=None):
        self.batch_size = len(image)
        #将不同模态的特征映射成相同维度的表征空间中
        imageH = self.imageMLP(image)

        textH = self.textMLP(text)
        #通过映射成粗的概念
        #通过解耦层提取位的粗语义概念
        ifeat_info = self.ifeat_gate(imageH)
        
        tfeat_info = self.tfeat_gate(textH)

        image_feat = ifeat_info*imageH

        text_feat = tfeat_info*textH

      
        cfeat_concat = torch.mul(image_feat, self.params[0]) + torch.mul(text_feat, self.params[1])

       

        nec_vec = self.neck(cfeat_concat)

       
        code = self.hash_output(nec_vec)

        
        return code, self.classify(code)

