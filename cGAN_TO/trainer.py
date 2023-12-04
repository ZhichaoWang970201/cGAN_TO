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
from scipy.ndimage.interpolation import rotate

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
rmin = 0.02*nelx
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

F = F / np.sqrt(1+rat*rat) * np.sqrt(2)

# boundary condition (BC)
r_s1 = nelx * 0.1
fixed = []
for i in range(nely+1):
    for j in range(nelx+1):
        if (math.sqrt((i-nely/2)**2 + (j-nely/2)**2)<=r_s1):
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

# Filter: Build (and assemble) the index+data vectors for the coo matrix format
nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
iH = np.zeros(nfilter)
jH = np.zeros(nfilter)
sH = np.zeros(nfilter)
cc=0
for i in range(nelx):
    for j in range(nely):
        row=i*nely+j
        kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
        kk2=int(np.minimum(i+np.ceil(rmin),nelx))
        ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
        ll2=int(np.minimum(j+np.ceil(rmin),nely))
        for k in range(kk1,kk2):
            for l in range(ll1,ll2):
                col=k*nely+l
                fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                iH[cc]=row
                jH[cc]=col
                sH[cc]=np.maximum(0.0,fac)
                cc=cc+1
# Finalize assembly and convert to csc format
H=csc_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely))
Hs = np.ndarray.flatten(np.asarray(H.sum(1)))

# Delete the rows and columns corresponding to fixed nodes
def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form !
    m = A.shape[0]
    keep = np.delete (np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete (np.arange(0, m), delcol)
    A = A[:, keep]
    return A

def rotate_c(orig, spoke):
    orig = np.reshape(orig, (nelx, nely))
    upda = np.zeros((nelx, nely))
    for i in range(spoke):
        upda += rotate(orig, i/spoke*360, reshape=False)
    upda /= spoke
    upda = np.reshape(upda, (nelx*nely))
    return upda
    
        
    
# create a function to do FEM, compliance and gradient
def fem(images_gpu, spoke, iK = iK, jK = jK):    
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
        x = np.reshape(images[i,0,:,:], (-1)) 
        
        x = H * x / Hs
        
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
        
        dc[i,:] = H * dc[i,:] / Hs
        # close rotation
        dc[i,:] = rotate_c(dc[i,:], np.int(12*spoke[i]))
        
        dc[i,:] = dc[i,:] - np.mean(dc[i,:]) # prevent volume fraction change
        #print(dc[i,:])
        
        # optimization criteria
        #volfrac = np.sum(x)/nelx/nely
        #xnew = oc(nelx,nely,x,volfrac,dc[i,:],dv)
        #import pdb; pdb.set_trace()
        #dc[i,:] = (x-xnew)/lr
        #print(dc[i,:])
    
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
        #fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim+2)) ################################################################################################
        fixed_z[:,self.z_dim] = torch.rand(fixed_z.size(0))*0.2+0.4  ####################################################################################################
        fixed_z[:,self.z_dim+1] = torch.randint(3, 13, (fixed_z.size(0),))/12 ###########################################################################################
        

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
            real_images[:,0,:,:] = 2*real_images[:,0,:,:]-1 # transform from [0,1] to [-1,1]
            real_images = tensor2var(real_images)
            d_out_real,dr1,dr2 = self.D(real_images)

            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            
            # apply Gumbel Softmax
            #z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim+2)) ##############################################################################################
            # the last dimension is for volume fraction: 0.4-0.6
            z[:,self.z_dim] = torch.rand(real_images.size(0))*0.2+0.4  ##################################################################################################
            z[:,self.z_dim+1] = torch.randint(3, 13, (real_images.size(0),))/12 #########################################################################################
            
            fake_images,gf1,gf2 = self.G(z)
            # add passive zone
            adjust_low = torch.ones_like(fake_images) ##################################################################################################################
            adjust_low[:,:,passive1[:,0],passive1[:,1]] = -1 ###########################################################################################################
            adjust_high = -torch.ones_like(fake_images) ################################################################################################################
            adjust_high[:,:,passive2[:,0],passive2[:,1]] = 1 ###########################################################################################################
            fake_images = torch.minimum(torch.maximum(fake_images,adjust_high),adjust_low) #############################################################################
            
            fake_vf = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(z[:,self.z_dim],1),2),3)*torch.ones_like(fake_images) #############################################
            fake_spoke = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(z[:,self.z_dim+1],1),2),3)*torch.ones_like(fake_images) ########################################
            fake_images = torch.cat((fake_images, fake_vf, fake_spoke), 1) ##############################################################################################
            d_out_fake,df1,df2 = self.D(fake_images)

            #vf_loss = torch.sum(torch.square(torch.sum((fake_images[:,0,:,:]+1)/2,[1,2]) - fake_images.size(2)*fake_images.size(3)*z[:,self.z_dim])) ####################
            #print(vf_loss) ##############################################################################################################################################
            vf_loss = torch.sum(torch.square(torch.sum((fake_images[:,0,:,:]+1)/2,[1,2]) - fake_images.size(2)*fake_images.size(3)*z[:,self.z_dim])) ###############################
            print(vf_loss) ##############################################################################################################################################
            
            import random ###############################################################################################################################################
            from torchvision.transforms.functional import rotate ########################################################################################################
            rotate_images = torch.ones_like(fake_images)*fake_images.detach() ###########################################################################################
            for ii in range(rotate_images.size(0)): #####################################################################################################################
                rotate_angle = (random.randint(1, (12*z[ii,self.z_dim+1])-1) / (12*z[ii,self.z_dim+1]) * 360).item() ####################################################
                orig_image = torch.unsqueeze((rotate_images[ii,0,:,:]+1)/2, 0) ################################################################################################
                rotate_images[ii,0,:,:] = rotate(orig_image, rotate_angle) ##############################################################################################
                rotate_images[ii,0,:,:] = 2 * rotate_images[ii,0,:,:] - 1 ############################################################################################### 
            spoke_loss = torch.sum(torch.square(rotate_images-fake_images)) #############################################################################################
            print(spoke_loss) ###########################################################################################################################################
            
            spoke_loss2 = torch.sum(torch.square(torch.sum(fake_images[:,2,:,:],[1,2]) - fake_images.size(2)*fake_images.size(3)*z[:,self.z_dim+1])) ###############################

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            
            # Backward + Optimize
            #### TO start here
            fake_images_TO = (fake_images+1)/2
            c_loss_fake = fem(torch.unsqueeze(fake_images_TO[:,0,:,:],1), z[:,self.z_dim+1]) 
            print("Discriminator")
            print(step)
            print(d_loss_real + d_loss_fake)
            print(c_loss_fake)
            #d_loss = d_loss_real + d_loss_fake + 2.5*10**(-5)*vf_loss + 10**(-7)*c_loss_fake + 10**(-2)*spoke_loss + 10**(-3)*spoke_loss2 ############################
            d_loss = d_loss_real + d_loss_fake + 10**(-6) * (1000*vf_loss + c_loss_fake)
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
            #z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim+2)) ##############################################################################################
            # the last dimension is for volume fraction: 0.4-0.6
            z[:,self.z_dim] = torch.rand(real_images.size(0))*0.2+0.4 ###################################################################################################
            z[:,self.z_dim+1] = torch.randint(3, 13, (real_images.size(0),))/12 #########################################################################################
            fake_images,_,_ = self.G(z)
            
            # add passive zone
            adjust_low = torch.ones_like(fake_images) ##################################################################################################################
            adjust_low[:,:,passive1[:,0],passive1[:,1]] = -1 ###########################################################################################################
            adjust_high = -torch.ones_like(fake_images) ################################################################################################################
            adjust_high[:,:,passive2[:,0],passive2[:,1]] = 1 ###########################################################################################################
            fake_images = torch.minimum(torch.maximum(fake_images,adjust_high),adjust_low) #############################################################################
            
            fake_vf = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(z[:,self.z_dim],1),2),3)*torch.ones_like(fake_images) #############################################
            fake_spoke = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(z[:,self.z_dim+1],1),2),3)*torch.ones_like(fake_images) ########################################
            fake_images = torch.cat((fake_images, fake_vf, fake_spoke), 1) ##############################################################################################
            
            # Compute loss with fake images
            g_out_fake,_,_ = self.D(fake_images)  # batch x n
                  
            #vf_loss = torch.sum(torch.square(torch.sum((fake_images[:,0,:,:]+1)/2,[1,2]) - fake_images.size(2)*fake_images.size(3)*z[:,self.z_dim])) ####################
            #print(vf_loss) ##############################################################################################################################################
            vf_loss = torch.sum(torch.square(torch.sum((fake_images[:,0,:,:]+1)/2,[1,2]) - fake_images.size(2)*fake_images.size(3)*z[:,self.z_dim])) ###############################
            print(vf_loss) ##############################################################################################################################################
            
            rotate_images = torch.ones_like(fake_images)*fake_images.detach() ###########################################################################################
            for ii in range(rotate_images.size(0)): #####################################################################################################################
                rotate_angle = (random.randint(1, (12*z[ii,self.z_dim+1])-1) / (12*z[ii,self.z_dim+1]) * 360).item() ##############################################################
                orig_image = torch.unsqueeze((rotate_images[ii,0,:,:]+1)/2, 0) ################################################################################################
                rotate_images[ii,0,:,:] = rotate(orig_image, rotate_angle) ##############################################################################################
                rotate_images[ii,0,:,:] = 2 * rotate_images[ii,0,:,:] - 1 ###############################################################################################
            spoke_loss = torch.sum(torch.square(rotate_images-fake_images)) #############################################################################################
            print(spoke_loss) ###########################################################################################################################################          
            print(z[:,self.z_dim])
            print(torch.sum((fake_images[:,0,:,:]+1)/2,[1,2])/fake_images.size(2)/fake_images.size(3))
            
            spoke_loss2 = torch.sum(torch.square(torch.sum(fake_images[:,2,:,:],[1,2]) - fake_images.size(2)*fake_images.size(3)*z[:,self.z_dim+1])) ###############################

            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()
                
            #### TO start here
            fake_images_TO = (fake_images+1)/2
            c_loss_fake = fem(torch.unsqueeze(fake_images_TO[:,0,:,:],1), z[:,self.z_dim+1]) 
            print('Generator')
            print(g_loss_fake)
            print(c_loss_fake)
            #g_loss_fake = g_loss_fake + 2.5*10**(-5)*vf_loss + 10**(-7)*c_loss_fake + 10**(-1)*spoke_loss + 10**(-3)*spoke_loss2 ######################################################################################
            g_loss_fake = g_loss_fake + 10**(-6) * (1000*vf_loss + c_loss_fake)
            #### TO end here

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            print(fixed_z[:,self.z_dim+1]*12)
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
                # add passive zone
                adjust_low = torch.ones_like(fake_images) ##################################################################################################################
                adjust_low[:,:,passive1[:,0],passive1[:,1]] = -1 ###########################################################################################################
                adjust_high = -torch.ones_like(fake_images) ################################################################################################################
                adjust_high[:,:,passive2[:,0],passive2[:,1]] = 1 ###########################################################################################################
                fake_images = torch.minimum(torch.maximum(fake_images,adjust_high),adjust_low) #############################################################################
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
