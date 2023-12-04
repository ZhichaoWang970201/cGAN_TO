## start from here fem
from __future__ import division
## end from here fem

import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator
from utils import *

## start from here fem
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import cvxopt ;import cvxopt.cholmod
import math

nely = 128
nelx = 128
Emin = 1e-9
Emax = 1.0
ndof = 2*(nelx+1)*(nely+1)
nele = nelx*nely

r_s = nelx * 0.075 # within it fixed/empty
r_m = nelx * 0.4 # the starting position of rim
r_l = nelx * 0.5 # the ending position of rim
# external load
rat = 1 # the ratio of normal force and shear force 
num_f = 3600
idx = np.zeros([2, 2*num_f])
data = np.zeros(2*num_f)
F = np.zeros((ndof,1))

for i in range(num_f):
    theta = 2*i/num_f*math.pi
    #c_idx = min( max( math.floor(nely/2+(nely/2-1)*math.cos(theta)) , 0) , nely ) # move inside a little bit
    #r_idx = min( max( math.floor(nely/2-(nely/2-1)*math.sin(theta)) , 0) , nely )
    c_idx = min( max( round(nely/2+(nely/2-1)*math.cos(theta)) , 0) , nely ) # move inside a little bit
    r_idx = min( max( round(nely/2-(nely/2-1)*math.sin(theta)) , 0) , nely )
    row_idx = (nely+1) * c_idx + r_idx
    F[2*row_idx,0] = F[2*row_idx,0] + math.sin(theta)-rat*math.cos(theta)
    F[2*row_idx+1,0] = F[2*row_idx+1,0] + (-math.cos(theta)-rat*math.sin(theta)) 

# boundary condition (BC)
fixed = []
for i in range(nely+1):
    for j in range(nelx+1):
        if (math.sqrt((i-nely/2)**2 + (j-nely/2)**2)<=r_s):
            bc_idx = (nely+1) * j + i
            fixed.append(2*bc_idx)
            fixed.append(2*bc_idx+1)
fixed = np.array(fixed)

dofs=np.arange(2*(nelx+1)*(nely+1))
free=np.setdiff1d(dofs,fixed)

# passive zone
# passive zone 1: 0.1*radius (all zero)
passive1 = []
for i in range(nely-1):
    for j in range(nelx-1):
        if (math.sqrt((i-63.5)**2 + (j-63.5)**2)<=r_s) | (math.sqrt((i-63.5)**2 + (j-63.5)**2)>=r_l):
            passive1.append([i,j])
passive1 = np.array(passive1)
# passive zone 2:
passive2 = []
for i in range(nely-1):
    for j in range(nelx-1):
        if (math.sqrt((i-63.5)**2 + (j-63.5)**2)>r_m) & (math.sqrt((i-63.5)**2 + (j-63.5)**2)<=r_l):
            passive2.append([i,j])        
passive2 = np.array(passive2)
            
nu=0.3
k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
KE = Emax/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);

# FE: Build the index vectors for the for coo matrix format.
edofMat=np.zeros((nelx*nely,8),dtype=int)
for elx in range(nelx):
    for ely in range(nely):
        el = ely+elx*nely
        n1=(nely+1)*elx+ely
        n2=(nely+1)*(elx+1)+ely
        edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
# Construct the index pointers for the coo format
iK = np.kron(edofMat,np.ones((8,1))).flatten()
jK = np.kron(edofMat,np.ones((1,8))).flatten()

# Delete the rows and columns corresponding to fixed nodes
def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form !
    m = A.shape[0]
    keep = np.delete (np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete (np.arange(0, m), delcol)
    A = A[:, keep]
    return A    
    
# create a function to do FEM, compliance and gradient
def fem(images_gpu, iK = iK, jK = jK):
    images = np.array(images_gpu.cpu().detach())
    num = np.shape(images)[0]
    u = np.zeros((num,ndof,1))
    ce = np.zeros((num,nele))
    c = np.zeros((num))
    dc = np.zeros((num,nele))
    dv = np.ones(nely*nelx)
    
    for i in range(num):
        # calculate the stiffness matrix
        penal = 3 # a parameter in TO
        x = np.reshape(images[i,0,:,:], (1,-1))
        sK = ((KE.flatten()[np.newaxis]).T*(Emin+(x)**penal*(Emax-Emin))).flatten(order='F')
        K = csc_matrix((sK,(iK,jK)),shape=(ndof,ndof))

        # get rid of fixed nodes
        K = deleterowcol(K, fixed, fixed).tocoo()
        #print(K)
        K = cvxopt.spmatrix(K.data,K.row.astype(np.int),K.col.astype(np.int))
        B = cvxopt.matrix(F[free,0])
        cvxopt.cholmod.linsolve(K,B)
        u[i,free,0]=np.array(B)[:,0] 
        
        ce[i,:] = (np.dot(u[i,edofMat].reshape(nelx*nely,8),KE) * u[i,edofMat].reshape(nelx*nely,8) ).sum(1)
        c[i] = ( (Emin+x**penal*(Emax-Emin))*ce[i,:] ).sum()
        dc[i,:]=(-penal*x**(penal-1)*(Emax-Emin))*ce[i,:] # original derivative    
        dc[i,:] = dc[i,:] - np.mean(dc[i,:]) # prevent volume fraction change
    
    images_gpu = images_gpu.view(num,-1)
    # move to gpu
    c = torch.sum(torch.tensor(c).to('cuda'))
    dc = torch.tensor(dc).to('cuda')
    c_linear = torch.sum(dc*images_gpu)
    c_final = c_linear - c_linear.detach() + c
    return c_final
## end here fem

class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()



    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = torch.unsqueeze(real_images[:,0,:,:],1)
            real_images = tensor2var(real_images)
            d_out_real,dr1,dr2 = self.D(real_images)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images,gf1,gf2 = self.G(z)
            d_out_fake,df1,df2 = self.D(fake_images)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            
            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake

            #### TO start here
            #fake_images_TO = (fake_images+1)/2
            #c_loss_fake = fem(fake_images_TO) 
            #print("Discriminator")
            #print(step)
            #print(d_loss)
            #print(c_loss_fake)
            #llamda = 10**(-3)
            #d_loss = d_loss_real + d_loss_fake + llamda * c_loss_fake
            d_loss = d_loss_real + d_loss_fake
            #### TO end here

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out,_,_ = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images,_,_ = self.G(z)
            # Compute loss with fake images
            g_out_fake,_,_ = self.D(fake_images)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()
            
            #### TO start here
            #fake_images_TO = (fake_images+1)/2
            #c_loss_fake = fem(fake_images_TO) 
            #print('Generator')
            #print(g_loss_fake)
            #print(c_loss_fake)
            #g_loss_fake = g_loss_fake + llamda*c_loss_fake
            g_loss_fake = g_loss_fake
            #### TO end here

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()


            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                #print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                #      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                #      format(elapsed, step + 1, self.total_step, (step + 1),
                #            #self.total_step , d_loss_real.data[0],
                #            #self.G.attn1.gamma.mean().data[0], self.G.attn2.gamma.mean().data[0] ))
                #            self.total_step , d_loss_real.data,
                #            self.G.attn1.gamma.mean().data, self.G.attn2.gamma.mean().data ))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, ".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                            #self.total_step , d_loss_real.data[0],
                            #self.G.attn1.gamma.mean().data[0], self.G.attn2.gamma.mean().data[0] ))
                            self.total_step , d_loss_real.data ))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images,_,_= self.G(fixed_z)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def build_model(self):

        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))
        
    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
