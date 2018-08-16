# helloworld
hehe
#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
# Train_Few_Shot
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 64)
parser.add_argument("-w","--class_num",type = int, default = 7)                        # 5 
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)             # could use:2
parser.add_argument("-b","--batch_num_per_class",type = int, default = 1)              #could use:3
parser.add_argument("-e","--episode",type = int, default= 5000)
parser.add_argument("-t","--test_episode", type = int, default = 10)                   # 1000
parser.add_argument("-l","--learning_rate", type = float, default = 0.0000001)            #0.001
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),                   #1
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))              
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True))                                    #add by me
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True))                                    #add by me

    def forward(self,x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer2_1(out)
        #print(out.shape)
        out = self.layer2_2(out)
        #print(out.shape)
        #out = self.layer2_3(out)
        #print(out.shape)
        #out = self.layer2_4(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        #self.drop = nn.Dropout(0.2)

    def forward(self,x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer2_1(out)
        #print(out.shape)
        out = self.layer2_2(out)
        #print(out.shape)
        #out = self.layer2_3(out)
        #print(out.shape)
        #out = self.layer2_4(out)
        #print(out.shape)
        out = out.view(out.size(0),-1)
        #print(out.shape)
        out = F.relu(self.fc1(out))
        #print(out.shape)
        out = F.sigmoid(self.fc2(out))
        #print(out.shape)
        #out = self.drop(out)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)                               #GPU
    relation_network.cuda(GPU)                               #GPU

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0
    
    loss_sum = 0
    
    loss_all = 0
    for episode in range(EPISODE):

        
        
        CLASS_NUM_TEST = 1

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        degrees = random.choice([0])
        task = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)


        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches_ori,batch_labels_ori = batch_dataloader.__iter__().next()
        
        #print(batch_labels_ori)
        
        index = random.randint(0,6)
        
        batches = batches_ori[index:index+1][:][:][:]
        batch_labels = batch_labels_ori[index]
        
        #print(samples)
        #print(sample_labels)
        
        #print(batches)
        #print(batch_labels)

        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 5x64*5*5               print(sample_features)                #GPU
        #print(sample_features)
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,16,23)                     #5,5                        
        sample_features = torch.sum(sample_features,1).squeeze(1)
        batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5                               #GPU

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM_TEST,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)

        #print(sample_features_ext.shape)
        #print(batch_features_ext.shape)
        
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,16,23)          #5,5
        
        #print(relation_pairs)
        
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
        
        
        '''#####################
        CLASS_NUM_TEST = 1
        SAMPLE_NUM_PER_CLASS_TEST = 1
        task1 = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,0)
        task2 = tg.OmniglotTask(metatest_character_folders,CLASS_NUM_TEST,0,SAMPLE_NUM_PER_CLASS_TEST)                      #
        
        #print("img_name : ",task2)
        
        sample_dataloader = tg.get_data_loader(task1,num_per_class=SAMPLE_NUM_PER_CLASS,split="train")
        test_dataloader = tg.get_data_loader(task2,num_per_class=SAMPLE_NUM_PER_CLASS_TEST,split="test")       #num per class

        sample_images,sample_labels = sample_dataloader.__iter__().next()
        test_images,test_labels = test_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64                           
        
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,16,23)                      #5,5  
        
        sample_features = torch.sum(sample_features,1).squeeze(1)
        
        test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS_TEST*CLASS_NUM_TEST,1,1,1,1)                        # 
        test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        test_features_ext = torch.transpose(test_features_ext,0,1)

        #print("sample_features_ext = ",sample_features_ext)
        #print("test_features_ext = ",test_features_ext)
        
        relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,16,23)            #strange            #5,5
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
        
        print(relations)

        ###################################'''


        mse = nn.MSELoss().cuda(GPU)                                                                                  #GPU
        criterion = nn.CrossEntropyLoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM_TEST, CLASS_NUM).scatter_(1, torch.LongTensor([[batch_labels]]), 1)).cuda(GPU)                               #GPU
              
        #one_hot_labels[0][batch_labels - 1] = 1
        
        
        
        #print(one_hot_labels)
        #print(relations)
        
        
        loss = mse(relations,one_hot_labels)
        
        loss_sum = loss_sum + loss.data
        
        '''if float(loss) > 0.1 :
          print(one_hot_labels)
          print(relations)'''


        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode+1)%10 == 0:
          print("episode:",episode+1,"loss",loss_sum/10)
          loss_all = loss_all + loss_sum
          loss_sum = 0
          #print("one_hot_labels:",one_hot_labels)
          #print("relations",relations)

        if (episode+1)%100 == 0:                      #original 5000
          torch.save(feature_encoder.state_dict(),str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
          torch.save(relation_network.state_dict(),str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

          print("save networks for episode:",episode)
          print("\n")
          print("loss_all:",loss_all/100)
          print("\n")
          loss_all = 0


if __name__ == '__main__':
    main()

#######################################################

#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
# Test_Few_Shot
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_test as tg
import os
import math
import argparse
import random
import scipy as sp
import scipy.stats
import sys

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 64)
parser.add_argument("-w","--class_num",type = int, default = 7)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 1)
parser.add_argument("-e","--episode",type = int, default= 1000)
parser.add_argument("-t","--test_episode", type = int, default = 1)
parser.add_argument("-l","--learning_rate", type = float, default = 0.000001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a),scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),                   #1
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))              
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True))                                    #add by me
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True))                                    #add by me

    def forward(self,x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer2_1(out)
        #print(out.shape)
        out = self.layer2_2(out)
        #print(out.shape)
        #out = self.layer2_3(out)
        #print(out.shape)
        #out = self.layer2_4(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        #self.drop = nn.Dropout(0.5)

    def forward(self,x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer2_1(out)
        #print(out.shape)
        out = self.layer2_2(out)
        #print(out.shape)
        #out = self.layer2_3(out)
        #print(out.shape)
        #out = self.layer2_4(out)
        #print(out.shape)
        out = out.view(out.size(0),-1)
        #print(out.shape)
        out = F.relu(self.fc1(out))
        #print(out.shape)
        out = F.sigmoid(self.fc2(out))
        #print(out.shape)
        #out = self.drop(out)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)


    feature_encoder.cuda(GPU)              #GPU
    relation_network.cuda(GPU)             #GPU

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")


    total_accuracy = 0.0
    
    right_num = 0
    for episode in range(EPISODE):
            
        # test
        print("Testing...")
        total_rewards = 0
        accuracies = []
        
        CLASS_NUM_TEST = 1

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        degrees = random.choice([0])
        task = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)


        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches_ori,batch_labels_ori = batch_dataloader.__iter__().next()
        
        #print(batch_labels_ori)
        
        index = random.randint(0,6)
        
        batches = batches_ori[index:index+1][:][:][:]
        batch_labels = batch_labels_ori[index]
        
        #print(samples)
        #print(sample_labels)
        
        #print(batches)
        #print(batch_labels)

        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 5x64*5*5               print(sample_features)                #GPU
        #print(sample_features)
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,16,23)                     #5,5                        
        sample_features = torch.sum(sample_features,1).squeeze(1)
        batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5                               #GPU

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM_TEST,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)

        #print(sample_features_ext.shape)
        #print(batch_features_ext.shape)
        
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,16,23)          #5,5
        
        #print(relation_pairs)
        
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
        
        
        
        '''degrees = random.choice([0])
        task = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS,)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        test_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=False,rotation=degrees)

        sample_images,sample_labels = sample_dataloader.__iter__().next()
        test_images,test_labels = test_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,16,23)          #7_add_2conv_3x4_randomcrop
        sample_features = torch.sum(sample_features,1).squeeze(1)
        test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        test_features_ext = torch.transpose(test_features_ext,0,1)

        relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,16,23)
        relations = relation_network(relation_pairs.cuda(GPU)).view(-1,CLASS_NUM)
        
        #print(relations)
        
        #for index, div_relations in enumerate(relations.data) :
        #  if max(div_relations) <0.85:
        #    predict_labels[index] = 100
        
        
        ##############
        
        del task, sample_dataloader, sample_images, sample_labels, sample_features, sample_features_ext, relation_pairs
        
        
        task = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS,)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        
        sample_images,sample_labels = sample_dataloader.__iter__().next()
        # calculate features
        sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,16,23)          #7_add_2conv_3x4_randomcrop
        sample_features = torch.sum(sample_features,1).squeeze(1)

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)

        relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,16,23)
        relations_2 = relation_network(relation_pairs.cuda(GPU)).view(-1,CLASS_NUM)
        
        #print(relations_2)
        
        for index_1, div_relations in enumerate(relations.data) :
          for index_2, div in enumerate(div_relations) :
            relations.data[index_1][index_2] = max(relations.data[index_1][index_2],relations_2.data[index_1][index_2])
        
        del task, sample_dataloader, sample_images, sample_labels, sample_features, sample_features_ext, relation_pairs, relations_2
        
        #print(relations)
        
        #################
        
        task = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS,)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        
        sample_images,sample_labels = sample_dataloader.__iter__().next()
        # calculate features
        sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,16,23)          #7_add_2conv_3x4_randomcrop
        sample_features = torch.sum(sample_features,1).squeeze(1)

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)

        relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,16,23)
        relations_2 = relation_network(relation_pairs.cuda(GPU)).view(-1,CLASS_NUM)
        
        #print(relations_2)
        
        for index_1, div_relations in enumerate(relations.data) :
          for index_2, div in enumerate(div_relations) :
            relations.data[index_1][index_2] = max(relations.data[index_1][index_2],relations_2.data[index_1][index_2])
        
        del relations_2
        
        #################
        
        print(relations)'''
        


        _,predict_labels = torch.max(relations.data,1)
        
        #for index, div_relations in enumerate(relations.data) :
        #  if max(div_relations) <0.85:
        #    predict_labels[index] = 100
        
        print(predict_labels)
        
        print(batch_labels)
        
        if int(predict_labels) == int(batch_labels) :
          right_num = right_num + 1
          print("right" + " " + str(episode))
        else:
          print("wrong" + " " + str(episode))
        
        

        #rewards = [1 if predict_labels[j]==batch_labels[j] else 0 for j in range(CLASS_NUM*BATCH_NUM_PER_CLASS)]

        #total_rewards += np.sum(rewards)
        #accuracy = np.sum(rewards)/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS
        #accuracies.append(accuracy)

        #test_accuracy,h = mean_confidence_interval(accuracies)

        #print("test accuracy:",test_accuracy,"h:",episode)
        #total_accuracy += test_accuracy
        
        #del task, sample_dataloader, batch_dataloader, samples, sample_labels, batches, batch_labels, batches_ori, batch_labels_ori, sample_features, batch_features, sample_features_ext, batch_features_ext, relation_pairs, relations
    
    print(right_num)
    print(EPISODE)
    #print("aver_accuracy:",right_num/EPISODE)




if __name__ == '__main__':
    main()

########################################

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:56:40 2018

@author: a84106805
"""

#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
# Try_Few_Shot
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_try as tg
import os
import math
import argparse
import random

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 64)
parser.add_argument("-w","--class_num",type = int, default = 7)                        # 5 
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 1)
parser.add_argument("-e","--episode",type = int, default= 1)
parser.add_argument("-t","--test_episode", type = int, default = 1)                 # 1000
parser.add_argument("-l","--learning_rate", type = float, default = 0.00001)            #0.001
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),                   #1
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))              
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True))                                    #add by me
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True))                                    #add by me

    def forward(self,x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer2_1(out)
        #print(out.shape)
        out = self.layer2_2(out)
        #print(out.shape)
        #out = self.layer2_3(out)
        #print(out.shape)
        #out = self.layer2_4(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(inplace = True),                                    #add by me
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        #self.drop = nn.Dropout(0.5)

    def forward(self,x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer2_1(out)
        #print(out.shape)
        out = self.layer2_2(out)
        #print(out.shape)
        #out = self.layer2_3(out)
        #print(out.shape)
        #out = self.layer2_4(out)
        #print(out.shape)
        out = out.view(out.size(0),-1)
        #print(out.shape)
        out = F.relu(self.fc1(out))
        #print(out.shape)
        out = F.sigmoid(self.fc2(out))
        #print(out.shape)
        #out = self.drop(out)
        return out
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        '''feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        degrees = random.choice([0,90,180,270])
        task = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)


        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 5x64*5*5
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,73,73)                     #5,5                        
        sample_features = torch.sum(sample_features,1).squeeze(1)
        batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)

        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,73,73)          #5,5
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(relations,one_hot_labels)


        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.data[0])

        if (episode+1)%1000 == 0:                      #original 5000

            # test
            print("Testing...")
            total_rewards = 0'''

        for i in range(10):
            
            
            CLASS_NUM_TEST = 1
    
            feature_encoder_scheduler.step(episode)
            relation_network_scheduler.step(episode)
    
            # init dataset
            # sample_dataloader is to obtain previous samples for compare
            # batch_dataloader is to batch samples for training
            degrees = random.choice([0])
            task1 = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,0)
            task2 = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,0,BATCH_NUM_PER_CLASS)
            
            sample_dataloader = tg.get_data_loader(task1,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",rotation=degrees)
            batch_dataloader = tg.get_data_loader(task2,num_per_class=BATCH_NUM_PER_CLASS,split="test",rotation=degrees)
    
    
            # sample datas
            samples,sample_labels = sample_dataloader.__iter__().next()
            batches_ori,batch_labels_ori = batch_dataloader.__iter__().next()
            
            #print(batch_labels_ori)
            
            #index = random.randint(0,6)
            index = 0
            
            batches = batches_ori[index:index+1][:][:][:]
            batch_labels = batch_labels_ori[index]
            
            #print(samples)
            #print(sample_labels)
            
            #print(batches)
            #print(batch_labels)
    
            # calculate features
            sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 5x64*5*5               print(sample_features)                #GPU
            #print(sample_features)
            sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,16,23)                     #5,5                        
            sample_features = torch.sum(sample_features,1).squeeze(1)
            batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5                               #GPU
    
            # calculate relations
            # each batch sample link to every samples to calculate relations
            # to form a 100x128 matrix for relation network
            sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM_TEST,1,1,1,1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
            batch_features_ext = torch.transpose(batch_features_ext,0,1)
    
            #print(sample_features_ext.shape)
            #print(batch_features_ext.shape)
            
            relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,16,23)          #5,5
            
            #print(relation_pairs)
            
            relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
            
            #print(relations)
            del task1,sample_dataloader,samples,sample_labels,sample_features,sample_features_ext,relation_pairs
            
            task1 = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,0)
            sample_dataloader = tg.get_data_loader(task1,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",rotation=degrees)
            samples,sample_labels = sample_dataloader.__iter__().next()
            sample_features = feature_encoder(Variable(samples).cuda(GPU))
            sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,16,23)                     #5,5                        
            sample_features = torch.sum(sample_features,1).squeeze(1)
            sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM_TEST,1,1,1,1)
            relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,16,23)
            relations_2 = relation_network(relation_pairs).view(-1,CLASS_NUM)
            for index_1, div_relations in enumerate(relations.data) :
              for index_2, div in enumerate(div_relations) :
                relations.data[index_1][index_2] = min(relations.data[index_1][index_2],relations_2.data[index_1][index_2])
            #print(relations)
            del sample_dataloader,samples,sample_labels,sample_features,sample_features_ext,relation_pairs,relations_2
            
            task1 = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,0)
            sample_dataloader = tg.get_data_loader(task1,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",rotation=degrees)
            samples,sample_labels = sample_dataloader.__iter__().next()
            sample_features = feature_encoder(Variable(samples).cuda(GPU))
            sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,16,23)                     #5,5                        
            sample_features = torch.sum(sample_features,1).squeeze(1)
            sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM_TEST,1,1,1,1)
            relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,16,23)
            relations_2 = relation_network(relation_pairs).view(-1,CLASS_NUM)
            for index_1, div_relations in enumerate(relations.data) :
              for index_2, div in enumerate(div_relations) :
                relations.data[index_1][index_2] = min(relations.data[index_1][index_2],relations_2.data[index_1][index_2])
            #print(relations)
            del sample_dataloader,samples,sample_labels,batches_ori,batch_labels_ori,batches,sample_features,batch_features,sample_features_ext,batch_features_ext,relation_pairs
                
            
            one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM_TEST, CLASS_NUM).scatter_(1, torch.LongTensor([[batch_labels]]), 1)).cuda(GPU)
            
            #print(relations)
            #print(one_hot_labels)
            
            predict_labels = torch.max(relations.data,1)
            
            #for index, div_relations in enumerate(relations.data) :
            #  if max(div_relations) <0.85:
            #    predict_labels[index] = 100
            
            #print(batch_labels)
            print(predict_labels)
            
            del task2,batch_dataloader,batch_labels,one_hot_labels,predict_labels,relations,relations_2





if __name__ == '__main__':
    main()

##############################

# code is based on https://github.com/katerakelly/pytorch-maml
# task_generator

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def omniglot_character_folders():
    data_folder = '../datas/test_logo/'            #original: omniglot_resized

    character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)

    num_train = 1200                    #original: 1200
    metatrain_character_folders = character_folders[:]          #[:num_train]  [:0]
    metaval_character_folders = character_folders[:]            #[num_train:]  [0:]

    return metatrain_character_folders,metaval_character_folders

class OmniglotTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, character_folders, num_classes, train_num,test_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')                                         # original: 'L'
        #image = image.resize((299,299), resample=Image.LANCZOS) # per Chelsea's implementation             original: (28,28)
        #image = np.array(image, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train',shuffle=True,rotation=0):
    # NOTE: batch size here is # instances PER CLASS
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

    dataset = Omniglot(task,split=split,transform=transforms.Compose([transforms.Resize(399),transforms.RandomCrop((299,399)),Rotate(rotation),transforms.ToTensor(),normalize]))
    #dataset = Omniglot(task,split=split,transform=transforms.Compose([Rotate(rotation),transforms.ToTensor(),normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

################################

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:48:41 2018

@author: a84106805
"""

# code is based on https://github.com/katerakelly/pytorch-maml
#task_generate_test

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def omniglot_character_folders():
    data_folder = '../datas/test_logo/'            #original: omniglot_resized

    character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)

    num_train = 1200                    #original: 1200
    metatrain_character_folders = character_folders[:]          #[:num_train]  [:0]
    metaval_character_folders = character_folders[:]            #[num_train:]  [0:]

    return metatrain_character_folders,metaval_character_folders

class OmniglotTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, character_folders, num_classes, train_num,test_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = self.character_folders
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')                                         # original: 'L'
        #image = image.resize((299,299), resample=Image.LANCZOS) # per Chelsea's implementation             original: (28,28)
        #image = np.array(image, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train',shuffle=True,rotation=0):
    # NOTE: batch size here is # instances PER CLASS
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

    dataset = Omniglot(task,split=split,transform=transforms.Compose([transforms.Resize(399),transforms.RandomCrop((299,399)),Rotate(rotation),transforms.ToTensor(),normalize]))
    #dataset = Omniglot(task,split=split,transform=transforms.Compose([Rotate(rotation),transforms.ToTensor(),normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler, pin_memory = True)

    return loader

###########################

# code is based on https://github.com/katerakelly/pytorch-maml
#task_generate_try

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def omniglot_character_folders():
    data_folder_1 = '../datas/test_logo/'

    character_folders_1 = [os.path.join(data_folder_1, family, character) \
                for family in os.listdir(data_folder_1) \
                if os.path.isdir(os.path.join(data_folder_1, family)) \
                for character in os.listdir(os.path.join(data_folder_1, family))]
    #random.seed(1)
    #random.shuffle(character_folders)

    #num_train = 1200
    metatrain_character_folders = character_folders_1[:]

    data_folder_2 = '../datas/testb/'                                   #
    character_folders_2 = [os.path.join(data_folder_2, family, character) \
                for family in os.listdir(data_folder_2) \
                if os.path.isdir(os.path.join(data_folder_2, family)) \
                for character in os.listdir(os.path.join(data_folder_2, family))]

    metaval_character_folders = character_folders_2[:]
    
    print("metatrain_character_folders = ", metatrain_character_folders)
    
    print("metaval_character_folders = ", metaval_character_folders)

    return metatrain_character_folders,metaval_character_folders

class OmniglotTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, character_folders, num_classes, train_num,test_num):               # if wanna try, test num is the try num

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = self.character_folders
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))                           #

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]
            
        #print("test_root = ",self.test_roots)

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        #image = image.resize((299,299), resample=Image.LANCZOS) # per Chelsea's implementation
        #image = np.array(image, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        
        batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train',rotation=0):
    # NOTE: batch size here is # instances PER CLASS
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

    dataset = Omniglot(task,split=split,transform=transforms.Compose([transforms.Resize(399),transforms.RandomCrop((299,399)),Rotate(rotation),transforms.ToTensor(),normalize]))
    #dataset = Omniglot(task,split=split,transform=transforms.Compose([Rotate(rotation),transforms.ToTensor(),normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)                     #num_per_class*task.num_classes 

    return loader


