"""
DA-RNN model initialization.

@author Zhenye Na 05/21/2018

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).

"""
from ops import *

#from natureGRU import *
from torch.autograd import Variable

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import math
"""
from platypus.types import Real
from platypus.algorithms import NSGAII, SMPSO, OMOPSO
"""
#Need to change the lstm module 
#In place use GRy nad specific gates to nature paper 
#Rest will be same .





class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
#         print(self.weight.data)
#         print(self.bias.data)

    def forward(self, input):
#         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        
        self.T = T


        self.delta_size=input_size
        self.identity = torch.eye(input_size).cuda()
        self.zeros_x = Variable(torch.zeros(self.delta_size).cuda())
        self.zeros_h= Variable(torch.zeros( self.encoder_num_hidden).cuda())
        
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)
        
        self.gamma_h_l = nn.Linear(self.delta_size, self.encoder_num_hidden)

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.encoder_num_hidden)

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + (self.T )*2, out_features=1, bias=True)

        #self.inputDecay = nn.Linear(encoder_num_hidden + 1, 1)
        #self.stateDecay=
    def forward(self, X,X_last):
        """forward.

        Args:
            X

        """

        mask = Variable(X.data.new(
            X.size(0), self.T, self.input_size).zero_()).cuda()
        X_clone=Variable(X.data.new(
            X.size(0), self.T, self.input_size).zero_()).cuda()
        X_mean=Variable(X.data.new(
            X.size(0), self.input_size).zero_()).cuda()
        X_mean_clone=Variable(X.data.new(
            X.size(0), self.input_size).zero_()).cuda()    
       
        
        deltaX=Variable(X.data.new(
            X.size(0), self.T, self.input_size).zero_()).cuda()
        #using the decay
        X_new = Variable(X.data.new(
            X.size(0), self.T , self.input_size).zero_()).cuda()
        X_tilde = Variable(X.data.new(
            X.size(0), self.T, self.input_size).zero_()).cuda()
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_()).cuda()

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # hidden, cell: initial states with dimention hidden_size

        #print(X_tilde.size())ma
        #print("s")
        h_n = self._init_states(X)
        s_n = self._init_states(X)
        #print("s")
        #print(h_n.size())
        #print("s")
        #print("x size",X.size())
        
        for t in range(self.T):
            for bs in range(X.size(0)):
                for fea in range(self.input_size):
                    #print(t," t",bs,"bs",fea," fea")
                    if( (X[bs,t,fea])!=-1):
                        mask[bs][t][fea]=1
                        
                           

                    if( t>0 and (X[bs,t-1,fea])==-1 ):
                        deltaX[bs][t][fea]=deltaX[bs][t-1][fea]+1
                       
                       
                    elif(t>0 and (X[bs,t-1,fea])!=-1):
                        deltaX[bs][t][fea]=1

        for bs in range(X.size(0)):
                for fea in range(self.input_size):       
                    count=0
                    sum_val=0
                    for t in range(self.T):
                        if((X[bs,t,fea])!=-1) :
                            sum_val= sum_val+X[bs,t,fea]
                            
                            count=count+1
                    if(count !=0):        
                        X_mean[bs,fea]=sum_val/count
        """
        X_clone=X
        print("ds",X_clone[X.size(0)-1,:,:])
        for bs in range(X.size(0)):
                for fea in range(self.input_size):
                    for t in range(self.T):
                        if((X[bs,t,fea])!=-1) :
                            if(maxX[bs,fea]!=minX[bs,fea]):
                                X[bs,t,fea]= (X_clone[bs,t,fea]-minX[bs,fea])/( maxX[bs,fea]-minX[bs,fea])
                            else:
                                X[bs,t,fea]=0.5
                    X_mean[bs,fea]=  (X_mean_clone[bs,fea]-minX[bs,fea])/( maxX[bs,fea]-minX[bs,fea]) 
            #print("sds",X_mean[1,:] )               
        """
        for bs in range(X.size(0)):
                for fea in range(self.input_size):       
                    for t in range(self.T):
                        if((X[bs,t,fea])==-1) :
                            X[bs,t,fea]=0
                            
        
        #print("X",X[X.size(0)-1,:,:])
        for t in range(self.T):
            # batch_size * input_size * (2*hidden_size + T - 1)
            
            
            delta_x = torch.exp(-torch.max(self.zeros_x, self.gamma_x_l(deltaX[:,t,:].view(-1, self.input_size))))
            delta_h = torch.exp(-torch.max(self.zeros_h, self.gamma_h_l(deltaX[:,t,:].view(-1, self.delta_size))))
            #print("time t",t)
            
            #print("delta_x",delta_x.size(),"delta_h",delta_h.size(),"mask",mask.size())
            #print("X",X.size(),"X_last",X_last.size(),"X_mean",X_mean.size())
        
            #X_new[:, t, :]=torch.mul(delta_x , X_last[:, t, :])
            #"steps"

            X_new[:, t, :] = torch.mul((1 - mask[:, t, :]),torch.mul(delta_x , X_last[:, t, :])+torch.mul((1 - delta_x) , X_mean))+torch.mul(mask[:, t, :],X[:, t, :])
            
            
                
        
            h_n=torch.squeeze(h_n)
            
            h_n = torch.mul(delta_h,h_n)
            
            h_n=torch.unsqueeze(h_n,0)
            
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X_new.permute(0, 2, 1),
                           mask.permute(0,2,1)), dim=2)
            #print(x.size())
            
            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + (self.T)*2))
            #print("s")
            ##print("s")
            # get weights by softmax
            
            alpha = F.softmax(x.view(-1, self.input_size))

            # get new input for LSTM
            #print("s")
           
            x_tilde = torch.mul(alpha, X[:, t, :])
            #print(x_tilde.size())
            # encoder LSTM
            #print("x_size ",x.size())
            #print("x_tlide",x_tilde.size())
            
        
            self.encoder_lstm.flatten_parameters()
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            #print("s")
            #print(len(final_state))
            h_n = final_state[0]
            
            s_n = final_state[1]
            #if(t==3):
             #   print("t",t,"h_n ",h_n)
            #print("s")
            #print(h_n.size())
            #rint("s")
            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
             X
        Returns:
            initial_hidden_states
        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = Variable(X.data.new(
            1, X.size(0), self.encoder_num_hidden).zero_()).cuda()
        return initial_states

 


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
                                        nn.Tanh(),
                                        nn.Linear(encoder_num_hidden, 1))
        self.lstm_layer = nn.LSTM(
            input_size=1, hidden_size=decoder_num_hidden)
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final1 = nn.Linear(decoder_num_hidden + encoder_num_hidden, decoder_num_hidden)
        self.fc_final2=nn.Linear(decoder_num_hidden , 1)
        self.fc.weight.data.normal_()

    def forward(self, X_encoed, y_prev):
        """forward."""
        d_n = self._init_states(X_encoed)
        c_n = self._init_states(X_encoed)
    
            
        for t in range(self.T - 1):

            x = torch.cat((d_n.repeat(self.T , 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           X_encoed), dim=2)

            
            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T))
            
            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoed)[:, 0, :]
            
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(

                    y_tilde.unsqueeze(0), (d_n, c_n))
                # 1 * batch_size * decoder_num_hidden
                d_n = final_states[0]
                # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]
        # Eqn. 22: final output
        y_predi = self.fc_final1(torch.cat((d_n[0], context), dim=1))
        y_pred =self.fc_final2(y_predi)
       
        
        #print("y",y_pred)    
        
        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstmdecoder_num_hidden
        initial_states = Variable(X.data.new(
            1, X.size(0), self.decoder_num_hidden).zero_()).cuda()
        return initial_states



class DA_rnn(nn.Module):
    """da_rnn."""

    def __init__(self, X, y,X_last,T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 resume,
                 parallel=False):
        """da_rnn initialization."""
        super(DA_rnn, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T                       #time intreval default 10
        self.X = X
        self.y = y
        self.X_last=X_last
        self.resume=resume

        #self.X_last=Variable(torch.from_numpy(extractX_d(X)).type(torch.FloatTensor)).cuda()
        self.Encoder = Encoder(input_size=X.shape[1],                     #d-varible size                 
                               encoder_num_hidden=encoder_num_hidden,
                               T=T)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T)

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)

        # Training set
        self.train_timesteps = int(self.X.shape[0] * 0.7)
        self.input_size = self.X.shape[1]

    def train(self):
        """training process."""
        min_loss=999999
        
        if(self.resume==True):
                """
                checkpointencoder = torch.load("../weights_new/EncoderEpochnew300.pt")
                self.Encoder.load_state_dict(checkpointencoder['model_state_dict'])
                self.encoder_optimizer.load_state_dict(checkpointencoder['optimizer_state_dict'])
                epoch_start = checkpointencoder['epoch']
                loss = checkpointencoder['loss']

                checkpointdecoder = torch.load("../weights_new/DecoderEpochnew300.pt")
                self.Decoder.load_state_dict(checkpointdecoder['model_state_dict'])
                self.decoder_optimizer.load_state_dict(checkpointdecoder['optimizer_state_dict'])
                print("loading checkpoint "+str(epoch_start)+"with loss"+str(loss))
                """
        else:

            epoch_start=0
        iter_per_epoch = int(np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros((self.epochs-epoch_start) * iter_per_epoch)
        self.iter_train_losses = np.zeros((self.epochs-epoch_start) * iter_per_epoch)
        self.epoch_losses = np.zeros((self.epochs-epoch_start))
        self.epoch_test_losses= np.zeros((self.epochs-epoch_start))

        n_iter = 0

        for epoch in range(self.epochs-epoch_start):
            
            
            
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while (idx < self.train_timesteps-self.T):
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                x = np.zeros((len(indices), self.T, self.input_size))    #need to change this 
                #x_last seen matrix
                x_last=np.zeros((len(indices), self.T, self.input_size))
                #print(idx ," batch_size",len(indices))
                #print("ref_idz",ref_idx)
                #print("train timesteps",self.train_timesteps)
                y_prev = np.zeros((len(indices), self.T - 1))
                y_gt = self.y[indices + self.T]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(indices[bs] + self.T ), :]
                    x_last[bs, :, :] = self.X_last[indices[bs]:(indices[bs] + self.T ), :]
                    y_prev[bs, :] = self.y[indices[bs]:(indices[bs] + self.T - 1)]

                loss= self.train_forward(x,x_last, y_prev, y_gt)
                self.iter_losses[epoch * iter_per_epoch + idx // self.batch_size] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 750 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

            self.epoch_losses[epoch] = np.mean(self.iter_losses[range(epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            self.test_loss_val=self.test_loss()

            self.epoch_test_losses[epoch]=self.test_loss_val

            if(epoch>50):
                plt.ioff()
                fig7=plt.figure()
                plt.plot(range(0,epoch ),
                         self.epoch_test_losses[range(0,epoch)], label="True")
                
                plt.legend(loc='upper left')
                plt.savefig("../results_new/7_new"+str(epoch+epoch_start)+"_17_8.png")
                plt.close(fig7)

            if(epoch>50):
                plt.ioff()
                fig8=plt.figure()
                plt.plot(range(0,epoch ),
                         self.epoch_losses[range(0,epoch)], label="True")
                
                plt.legend(loc='upper left')
                plt.savefig("../results_new/8_new"+str(epoch+epoch_start)+"_17_8.png")
                plt.close(fig8)    



            if epoch % 2 == 0:
                print ("Epochs: ", epoch+epoch_start, " Iterations: ", n_iter, " Loss: ", self.epoch_losses[epoch])
            if epoch % 50 == 0 and self.epoch_losses[epoch]<min_loss:
                print("Model's state_dict:")
                for param_tensor in self.Encoder.state_dict():
                    print(param_tensor, "\t", self.Encoder.state_dict()[param_tensor].size())

                # Print optimizer's state_dict
                #print("Optimizer's state_dict:")
                #for var_name in self.encoder_optimizer.state_dict():
                   # print(var_name, "\t",  self.encoder_optimizer.state_dict()[var_name])

                print("Model's state_dict:")
                for param_tensor in self.Decoder.state_dict():
                    print(param_tensor, "\t", self.Decoder.state_dict()[param_tensor].size())

                # Print optimizer's state_dict
                #print("Optimizer's state_dict:")
                #for var_name in self.decoder_optimizer.state_dict():
                  #  print(var_name, "\t",  self.decoder_optimizer.state_dict()[var_name])        
                torch.save({
                            'epoch': epoch+epoch_start,
                            'model_state_dict': self.Encoder.state_dict(),
                            'optimizer_state_dict': self.encoder_optimizer.state_dict(),
                            'loss': loss
                            },( "../weights_new/EncoderEpochnew"+str(epoch+epoch_start)+"_17_8.pt"))
                torch.save({
                            'epoch': epoch+epoch_start,
                            'model_state_dict': self.Decoder.state_dict(),
                            'optimizer_state_dict': self.decoder_optimizer.state_dict(),
                            'loss': loss
                            }, ("../weights_new/DecoderEpochnew"+str(epoch+epoch_start)+"_17_8.pt"))
                min_loss= self.epoch_losses[epoch]      
                      
            if (epoch+epoch_start)%25==0 :
                y_train_pred = self.test(on_train=True)
                y_test_pred = self.test(on_train=False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                plt.ioff()
                fig4=plt.figure()
                plt.plot(range(1, 1 + len(self.y)),
                         self.y, label="True")
                plt.plot(range(self.T, len(y_train_pred) + self.T),
                         y_train_pred, label='Predicted - Train')
                plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
                         y_test_pred, label='Predicted - Test')
                plt.legend(loc='upper left')
                plt.savefig("../results_new/4_new"+str(epoch+epoch_start)+"_17_8.png")
                plt.close(fig4)
                
                plt.ioff()
                fig5 = plt.figure()
                plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
                         self.y[self.train_timesteps:], label="True")
                plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
                         y_test_pred, label='Predicted - Test')
               
                plt.legend(loc = 'upper left')
                plt.savefig("../results_new/5_new"+str(epoch+epoch_start)+"_17_8.png")
                plt.close(fig5)
                plt.ioff()
                fig6 = plt.figure()
                plt.plot(range(1, len(y_train_pred) + self.T),
                         self.y[:self.train_timesteps], label="True")
                plt.plot(range(self.T, len(y_train_pred) + self.T),
                         y_train_pred, label='Predicted - Train')
               
                plt.legend(loc = 'upper left')
                plt.savefig("../results_new/6_new"+str(epoch+epoch_start)+"_17_8.png")
                plt.close(fig6)
                np.savetxt("../results_new/epoch_loss1_"+str(epoch+epoch_start)+"_17_8.csv",  +self.epoch_losses, delimiter=",")


            # Save files in last iterations
            # if epoch == self.epochs - 1:
            #     np.savetxt('../loss.txt', np.array(self.epoch_losses), delimiter=',')
            #     np.savetxt('../y_pred.txt',
            #                np.array(self.y_pred), delimiter=',')
            #     np.savetxt('../y_true.txt',
            #                np.array(self.y_true), delimiter=',')


    def train_forward(self, X,X_last, y_prev, y_gt):
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        #input X not perfect need to change this

        
        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor)).cuda(),Variable(torch.from_numpy(X_last).type(torch.FloatTensor)).cuda())
        y_pred = self.Decoder(input_encoded, Variable(
            torch.from_numpy(y_prev).type(torch.FloatTensor)).cuda())

        y_true = Variable(torch.from_numpy(
            y_gt).type(torch.FloatTensor)).cuda()

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()



    def test_loss(self):
        y_pred = np.zeros((self.X.shape[0] - self.train_timesteps,1))

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T , self.X.shape[1]))
            X_last=np.zeros((len(batch_idx), self.T, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                X[j, :, :] = self.X[range(batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps ), :]
                X_last[j,:,:]=self.X_last[range(batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps), :]
                y_history[j, :] = self.y[range(batch_idx[j] + self.train_timesteps - self.T,  batch_idx[j]+ self.train_timesteps - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor)).cuda()
            _, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)).cuda(),Variable(torch.from_numpy(X_last).type(torch.FloatTensor)).cuda())
            y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded, y_history).detach().cpu().numpy()
            i += self.batch_size

        y_gt=self.y[range(self.train_timesteps,self.X.shape[0])]    
        y_true = Variable(torch.from_numpy(
            y_gt).type(torch.FloatTensor)).cuda()

        y_predg=Variable(torch.from_numpy(
            y_pred).type(torch.FloatTensor)).cuda()

        y_true = y_true.view(-1, 1)

        loss = self.criterion(y_predg, y_true)
        
        return loss
        


    def test(self, on_train=False):
        """test."""

        if on_train:
            y_pred = np.zeros((self.train_timesteps - self.T + 1,1))
           
        else:
            y_pred = np.zeros((self.X.shape[0] - self.train_timesteps,1))
           

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T , self.X.shape[1]))
            X_last=np.zeros((len(batch_idx), self.T, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T ), :]
                    X_last[j,:,:]=self.X[range(batch_idx[j], batch_idx[j] + self.T ), :]
                    y_history[j, :] = self.y[range(batch_idx[j],  batch_idx[j]+ self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps ), :]
                    X_last[j,:,:]=self.X_last[range(batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps), :]
                    y_history[j, :] = self.y[range(batch_idx[j] + self.train_timesteps - self.T,  batch_idx[j]+ self.train_timesteps - 1)]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor)).cuda()
            _, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)).cuda(),Variable(torch.from_numpy(X_last).type(torch.FloatTensor)).cuda())
            y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded, y_history).detach().cpu().numpy()
            i += self.batch_size

        return y_pred


    """
    def evalModel(self,X_pred):
        #test
        checkpointencoder = torch.load("../weights_new/EncoderEpochnew450.pt")
        self.Encoder.load_state_dict(checkpointencoder['model_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpointencoder['optimizer_state_dict'])
        epoch_start = checkpointencoder['epoch']
        loss = checkpointencoder['loss']

        checkpointdecoder = torch.load("../weights_new/DecoderEpochnew450.pt")
        self.Decoder.load_state_dict(checkpointdecoder['model_state_dict'])
        self.decoder_optimizer.load_state_dict(checkpointdecoder['optimizer_state_dict'])
        #print("loading checkpoint "+str(epoch_start)+"with loss"+str(loss))
       
        y_pred = np.zeros(1)

        
          
        X = np.zeros((1, self.T, self.X.shape[1]))
        X_last=np.zeros((1, self.T , self.X.shape[1]))
        y_history = np.zeros((1, self.T - 1))     
            
        X[0, 0:self.T-1, :] = self.X[range(int(self.X.shape[0]) - self.T+1, int(self.X.shape[0])), :]
        X_last[0,0:self.T-1,:]=self.X_last[range(int(self.X.shape[0]) - self.T+1, int(self.X.shape[0])), :]
        y_history[0, :] = self.y[range(int(self.X.shape[0]) - self.T+1, int(self.X.shape[0]))]
        X[0, self.T-1, :]=X_pred
        X_last[0, self.T-1,:]=X_pred
        y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor)).cuda()
        _, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor)).cuda(),Variable(torch.from_numpy(X_last).type(torch.FloatTensor)).cuda())
        y_pred = self.Decoder(input_encoded, y_history).cpu().data.numpy()
        #print("y_pred",y_pred)
        #print("X_pred",X_pred)
        val=y_pred[0,0]
        return val
    """        