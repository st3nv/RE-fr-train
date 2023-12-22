import sys, os
os.chdir('/Users/songtengyu/Documents/2023Fall/Information theory/fr-train/')

import numpy as np
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from argparse import Namespace
from FRTrain_arch import Generator, DiscriminatorF, DiscriminatorR, weights_init_normal, test_model
import warnings
warnings.filterwarnings("ignore")
import pandas as pd


def myMIloss(output, target, y_prop, beta=20):
    H_Y = -y_prop * torch.log(y_prop) - (1-y_prop) * torch.log(1-y_prop)
    output = output * beta
    # convert output to probability
    sig_oup = torch.sigmoid(output)
    # convert target to float
    target = target.float()

    # calculate mutual information
    p_s1 = torch.mean(target)
    p_s0 = 1 - p_s1
    p_y1 = torch.mean(sig_oup)
    p_y0 = 1 - p_y1

    p_s1y1 = torch.mean(target * sig_oup)
    p_s1y0 = p_s1 - p_s1y1
    p_s0y1 = p_y1 - p_s1y1
    p_s0y0 = 1 - p_s1y1 - p_s1y0 - p_s0y1
    

    p_s1_given_y1 = p_s1y1 / p_y1
    p_s1_given_y0 = p_s1y0 / p_y0
    p_s0_given_y1 = p_s0y1 / p_y1
    p_s0_given_y0 = p_s0y0 / p_y0

    H_S = -p_s1 * torch.log(p_s1) - p_s0 * torch.log(p_s0)
    
    H_S_given_Y = 0
    
    for tmp, tmp_cond in zip([p_s1y1, p_s1y0, p_s0y1, p_s0y0], [p_s1_given_y1, p_s1_given_y0, p_s0_given_y1, p_s0_given_y0]):
        if tmp != 0:
            H_S_given_Y += -tmp * torch.log(tmp_cond)

    mi = H_S - H_S_given_Y
    mi_normed = mi / H_Y

    return mi_normed


def train_model(train_tensors, val_tensors, tune_tensors, train_opt, lambda_f, lambda_r, seed):
    """
      Trains FR-Train by using the classes in FRTrain_arch.py.
      
      Args:
        train_tensors: Training data.
        val_tensors: Clean validation data.
        test_tensors: Test data.
        train_opt: Options for the training. It currently contains size of validation set, 
                number of epochs, generator/discriminator update ratio, and learning rates.
        lambda_f: The tuning knob for L_2 (ref: FR-Train paper, Section 3.3).
        lambda_r: The tuning knob for L_3 (ref: FR-Train paper, Section 3.3).
        seed: An integer value for specifying torch random seed.
        
      Returns:
        Information about the tuning knobs (lambda_f, lambda_r),
        the test accuracy of the trained model, and disparate impact of the trained model.
    """
    
    XS_train = train_tensors.XS_train
    y_train = train_tensors.y_train
    s1_train = train_tensors.s1_train
    
    XS_val = val_tensors.XS_val
    y_val = val_tensors.y_val
    s1_val = tune_tensors.s1_val
    
    XS_tune = tune_tensors.XS_val
    y_tune = tune_tensors.y_val
    s1_tune = tune_tensors.s1_val
    
    # Saves return values here
    test_result = []
    
    val = train_opt.val # Number of data points in validation set
    k = train_opt.k     # Updates ratio of generator and discriminator (1:k training).
    n_epochs = train_opt.n_epochs  # Number of training epoch
    
    # Changes the input validation data to an appropriate shape for the training
    XSY_val = torch.cat([XS_val, y_val.reshape((y_val.shape[0], 1))], dim=1)  

    # The loss values of each component will be saved in the following lists. 
    # We can draw epoch-loss graph by the following lists, if necessary.
    g_losses =[]
    d_f_losses = []
    d_r_losses = []
    clean_test_result = []

    bce_loss = torch.nn.BCELoss()

    # Initializes generator and discriminator
    generator = Generator()
    discriminator_F = DiscriminatorF()
    discriminator_R = DiscriminatorR()

    # Initializes weights
    torch.manual_seed(seed)
    generator.apply(weights_init_normal)
    discriminator_F.apply(weights_init_normal)
    discriminator_R.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_opt.lr_g)
    optimizer_D_F = torch.optim.SGD(discriminator_F.parameters(), lr=train_opt.lr_f)
    optimizer_D_R = torch.optim.SGD(discriminator_R.parameters(), lr=train_opt.lr_r)

    XSY_val_data = XSY_val[:val]

    train_len = XS_train.shape[0]
    val_len = XSY_val.shape[0]

    # Ground truths using in Disriminator_R
    Tensor = torch.FloatTensor
    valid = Variable(Tensor(train_len, 1).fill_(1.0), requires_grad=False)
    generated = Variable(Tensor(train_len, 1).fill_(0.0), requires_grad=False)
    fake = Variable(Tensor(train_len, 1).fill_(0.0), requires_grad=False)
    clean = Variable(Tensor(val_len, 1).fill_(1.0), requires_grad=False)
    
    r_weight = torch.ones_like(y_train, requires_grad=False).float()
    r_ones = torch.ones_like(y_train, requires_grad=False).float()
    y_prop = torch.mean(y_train == 1).detach().float()

    for epoch in range(n_epochs):

        # -------------------
        #  Forwards Generator
        # -------------------
        if epoch % k == 0 or epoch < 500:
            optimizer_G.zero_grad()

        gen_y = generator(XS_train)
        gen_data = torch.cat([XS_train, gen_y.detach().reshape((gen_y.shape[0], 1))], dim=1)
        
        # -----------------------------
        #  Trains Fairness Discriminator
        # -----------------------------

        optimizer_D_F.zero_grad()

        # Discriminator_F tries to distinguish the sensitive groups by using the output of the generator.
        d_f_loss= bce_loss(discriminator_F(gen_y.detach()).squeeze(), s1_train.squeeze())
        d_f_loss.backward()
        d_f_losses.append(d_f_loss)
        optimizer_D_F.step()
            

        # ---------------------------------
        #  Trains Robustness Discriminator
        # ---------------------------------
        optimizer_D_R.zero_grad()

        # Discriminator_R tries to distinguish whether the input is from the validation data or the generated data from generator.
        clean_loss = bce_loss(discriminator_R(XSY_val_data), clean)
        poison_loss = bce_loss(discriminator_R(gen_data.detach()), fake)
        d_r_loss = 0.5 * (clean_loss + poison_loss)

        d_r_loss.backward()
        d_r_losses.append(d_r_loss)
        optimizer_D_R.step()


        # ---------------------
        #  Updates Generator
        # ---------------------

        # Loss measures generator's ability to fool the discriminators
        if epoch < 500 :
            g_loss = bce_loss((F.tanh(gen_y).squeeze()+1)/2, (y_train+1)/2)
            g_loss.backward()
            g_losses.append(g_loss)
            optimizer_G.step()
    
        elif epoch % k == 0:
            r_decision = discriminator_R(gen_data)
            r_gen = bce_loss(r_decision, generated)
            
            # ------------------------------
            #  Re-weights using output of D_R
            # ------------------------------
            
            if epoch % 100 == 0:
                loss_ratio = (g_losses[-1]/d_r_losses[-1]).detach()
                a = 1/(1+torch.exp(-(loss_ratio-3)))
                b = 1-a
                r_weight_tmp = r_decision.detach().squeeze()
                r_weight = a * r_weight_tmp + b * r_ones

            g_cost = F.binary_cross_entropy_with_logits(gen_y.squeeze(), (y_train.squeeze()+1)/2, reduction="none").squeeze()

            f_gen = myMIloss(gen_y.squeeze(), s1_train, y_prop, beta=20)
            
            g_loss = (1-lambda_f-lambda_r) * torch.mean(g_cost*r_weight) -  lambda_r * r_gen - lambda_f * f_gen

            g_loss.backward()
            optimizer_G.step()

        g_losses.append(g_loss)

    tmp = test_model(generator, XS_tune, y_tune, s1_tune)
    test_result.append([lambda_f, lambda_r, tmp[0].item(), tmp[1]])
    return test_result


def finetune_performance(train_data, clean_data, val_data, test_data, lambda_f_lst, lambda_r_lst, seed=0):
    performance_lst = []
    best_params = [-1, -1]
    best_performance = 0
    
    XS_train = train_data.XS_train
    y_train = train_data.y_train
    s1_train = train_data.s1_train
    XS_val = clean_data.XS_clean
    y_val = clean_data.y_clean
    s1_val = clean_data.s1_clean
    XS_tune = val_data.XS_val
    y_tune = val_data.y_val
    s1_tune = val_data.s1_val
    XS_test = test_data.XS_test
    y_test = test_data.y_test
    s1_test = test_data.s1_test
    

    train_tensors = Namespace(XS_train = XS_train, y_train = y_train, s1_train = s1_train)
    val_tensors = Namespace(XS_val = XS_val, y_val = y_val, s1_val = s1_val) 
    tune_tensors = Namespace(XS_val = XS_tune, y_val = y_tune, s1_val = s1_tune)
    test_tensors = Namespace(XS_val = XS_test, y_val = y_test, s1_val = s1_test)

    train_opt = Namespace(val=len(y_val), n_epochs=4000, k=3, lr_g=0.005, lr_f=0.01, lr_r=0.001)
    
    for lambda_f in lambda_f_lst:
        for lambda_r in lambda_r_lst:
            test_result = train_model(train_tensors, val_tensors, tune_tensors, train_opt, lambda_f, lambda_r, seed)
            test_acc, test_di = test_result[0][2], test_result[0][3]
            performance_lst.append([lambda_f, lambda_r, test_acc, test_di])
            if test_acc + test_di > best_performance:
                best_performance = test_acc + test_di
                best_params = [lambda_f, lambda_r]
    
    test_result = train_model(train_tensors, val_tensors, test_tensors, train_opt, best_params[0], best_params[1], seed)
        
    return performance_lst, best_params, test_result[0][2], test_result[0][3]
 