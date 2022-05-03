import sys

import torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from model import *


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=1)
        nn.init.constant_(m.bias.data, 0.1)


def train_ThetaFix(encoder):
    
    W1 = encoder.fc.weight.data # mark parameters that need to be sparse
    d = list(W1.size())[1]
    U = torch.zeros(d)
    for i in range(d):
        U[i] = 1 / (torch.linalg.norm(W1[:,i].float()) + sys.float_info.epsilon)
    U = torch.diag(U)
    
    return U  


def train_UFix(encoder_part1, encoder_part2, 
               classifier, 
               decoder_part1, decoder_part2, 
               data_trl_part1, data_trl_part2, 
               labels_trl, 
               data_tru_part1, data_tru_part2, 
               U_part1, U_part2, 
               L_part1, L_part2, 
               n_epoch_local, lr_local, alpha, beta):
        
    device = torch.device('cpu')
    
    # set computing device for the entire network
    # encoder
    encoder_part1.to(device)
    encoder_part2.to(device)
    # classifier
    classifier.to(device)
    # decoder
    decoder_part1.to(device)
    decoder_part2.to(device)
    
    # set training state for the entire network
    # encoder
    encoder_part1.train()
    encoder_part2.train()
    # classifier
    classifier.train()
    # decoder
    decoder_part1.train()
    decoder_part2.train()
    
    # set classification loss function
    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_rec = nn.MSELoss().to(device)
    
    # mark parameters that need to be sparse
    W1_part1 = encoder_part1.fc.weight.data 
    W1_part2 = encoder_part2.fc.weight.data 
    
    # set optimizer
    optimizer = optim.Adam([{'params': encoder_part1.parameters()},
                            {'params': encoder_part2.parameters()},
                            {'params': classifier.parameters()},
                            {'params': decoder_part1.parameters()},
                            {'params': decoder_part2.parameters()}],
                            lr=lr_local)
    
    # set optimization scheduler
    scheduler = lr_scheduler.StepLR(optimizer, 10, 0.96)
    
    # Make training data and labels variable
    # overlapping samples
    data_trl_part1 = Variable(data_trl_part1).to(device) 
    data_trl_part2 = Variable(data_trl_part2).to(device) 
    # label information for overlapping samples
    labels_trl = Variable(labels_trl).to(device) 
    # non-overlapping samples
    data_tru_part1 = Variable(data_tru_part1).to(device) 
    data_tru_part2 = Variable(data_tru_part2).to(device) 
    
    # training process for local encoder, decoder, and classifier
    # print('>>> Local Network Training ...')

    for epoch in range(n_epoch_local):
        
        # print('epoch'+str(epoch)+'\n')       
              
        # local encoding process
        # overlapping samples
        feat_trl_part1 = encoder_part1(data_trl_part1)
        feat_trl_part2 = encoder_part2(data_trl_part2)
        # non-overlapping samples
        feat_tru_part1 = encoder_part1(data_tru_part1)
        feat_tru_part2 = encoder_part2(data_tru_part2)
        
        # Classification
        cls_feat = classifier(torch.cat((feat_trl_part1, feat_trl_part2), 1))
        # Reconstruction
        data_tru_rec_part1 = decoder_part1(feat_tru_part1)
        data_tru_rec_part2 = decoder_part2(feat_tru_part2)
        
        # Set zero gradients for the optimizer of classification
        optimizer.zero_grad()
        
        # Computer training loss for classification in each epoch
        loss_cls = criterion_cls(cls_feat, torch.squeeze(labels_trl))
        loss_rec = criterion_rec(data_tru_part1, data_tru_rec_part1) + \
                   criterion_rec(data_tru_part2, data_tru_rec_part2)
        loss_sparse = torch.trace(W1_part1.mm(U_part1).mm(W1_part1.t())) + \
                      torch.trace(W1_part2.mm(U_part2).mm(W1_part2.t()))
        feat_part1 = torch.cat((feat_trl_part1, feat_tru_part1), 0)
        feat_part2 = torch.cat((feat_trl_part2, feat_tru_part2), 0)
        loss_graph = torch.trace(feat_part1.t().mm(L_part1).mm(feat_part1)) + \
                     torch.trace(feat_part2.t().mm(L_part2).mm(feat_part2))
        loss = alpha * loss_graph + loss_rec + loss_cls + beta * loss_sparse
        
        # optimize network parameters
        loss.backward()
        optimizer.step()
        scheduler.step() 

    return encoder_part1, encoder_part2, classifier, decoder_part1, decoder_part2


def train(dataset, 
          data_trl_part1, data_trl_part2, 
          labels_trl, 
          data_tru_part1, data_tru_part2, 
          L_part1, L_part2, 
          n_dim_part1, n_dim_part2, 
          size_localout, 
          n_party, n_class, alpha, beta):
    
    n_epoch = 5
    
    if dataset == 'handwritten':    
        n_epoch_local = 400
        lr_local = 1e-3
    elif dataset == 'tox':
        n_epoch_local = 400
        lr_local = 1e-3
        
    # Local Model Initialization
    # print('>>> Local Model Initialization ......')
    # Local Encoder
    encoder_part1 = LocalEncoder(n_dim_part1, size_localout)
    encoder_part1.apply(weights_init)
    encoder_part2 = LocalEncoder(n_dim_part2, size_localout)
    encoder_part2.apply(weights_init)
    # Local Decoder
    decoder_part1 = LocalDecoder(size_localout, n_dim_part1)
    decoder_part1.apply(weights_init)
    decoder_part2 = LocalDecoder(size_localout, n_dim_part2)
    decoder_part2.apply(weights_init)
    # Classifier
    classifier = Classifier(size_localout * n_party, n_class)
    classifier.apply(weights_init)
        
    for i in range(n_epoch):
        
        # Optimizing U with Theta being fixed
        U_part1 = train_ThetaFix(encoder_part1)
        U_part2 = train_ThetaFix(encoder_part2)
        
        # Optimizing Theta with U being fixed
        encoder_part1, encoder_part2, \
        classifier, \
        decoder_part1, decoder_part2 = \
        train_UFix(encoder_part1, encoder_part2, 
                   classifier, 
                   decoder_part1, decoder_part2, 
                   data_trl_part1, data_trl_part2, 
                   labels_trl, 
                   data_tru_part1, data_tru_part2,  
                   U_part1, U_part2, 
                   L_part1, L_part2, 
                   n_epoch_local, lr_local, alpha, beta)
        
    W1_part1 = encoder_part1.fc.weight.data 
    W1_part1 = W1_part1.cpu().detach().numpy()
    W1_part2 = encoder_part2.fc.weight.data 
    W1_part2 = W1_part2.cpu().detach().numpy()
     
    return W1_part1, W1_part2