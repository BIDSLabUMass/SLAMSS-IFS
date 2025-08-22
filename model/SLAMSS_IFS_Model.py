import os
import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import scipy.io
import glob
import torch.nn.functional as F
from numpy import load

class Encode_micro(nn.Module):
    def __init__(self):
        super(Encode_micro, self).__init__()
        
        self.rnn = nn.GRU(2, 32, 2, bidirectional = True, dropout=0.5, batch_first = True)
        
        self.fc = nn.Sequential( 
            nn.Linear(32*2, 128),
        )
     
    def forward(self, src):
        
        #src = [src sent len, batch size]
        outputs, hidden = self.rnn(src)
        hidden = self.fc(outputs[:,-1,:])
        return outputs, hidden


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.rnn = nn.GRU(2, 64, 3, bidirectional = True, dropout=0.5, batch_first = True)
        
        self.cnn = nn.Sequential( 
        nn.Conv1d(2, 128, 5, stride=1, padding=2),
        nn.LeakyReLU(inplace=False),
        nn.MaxPool1d(6, stride=6),#60
        nn.Conv1d(128, 128, 5, stride=1, padding=2),
        nn.LeakyReLU(inplace=False),
        #nn.Dropout(0.5),
        #nn.MaxPool1d(3, stride=3),#20
        nn.Conv1d(128, 128, 5, stride=1, padding=2),
        nn.LeakyReLU(inplace=False),
        nn.MaxPool1d(5, stride=5),#10
        nn.Conv1d(128, 128, 5, stride=1, padding=2),
        nn.LeakyReLU(inplace=False),
        # #nn.MaxPool1d(2, stride=2),#5
        nn.Conv1d(128, 128, 5, stride=1, padding=2),
        # nn.LeakyReLU(inplace=False),
        # #nn.MaxPool1d(2, stride=2),#2
        # nn.Conv1d(128, 256, 9, stride=1, padding=4),     
        nn.Dropout(0.5),
        #nn.LeakyReLU(inplace=False), #put it back 2020 706
        )
         
        self.cnn_ACT = nn.Sequential( 
        nn.Conv1d(5, 16, 9, stride=1, padding=4),
        nn.LeakyReLU(inplace=False),
        nn.Conv1d(16, 32, 9, stride=1, padding=4),
        nn.LeakyReLU(inplace=False),
        nn.Conv1d(32, 64, 9, stride=1, padding=4),
        nn.Dropout(0.5),
        )
        self.cnn3 = nn.Sequential( 
        nn.Conv1d(128+64+128, 256, 11, stride=1, padding=5),
        nn.LeakyReLU(inplace=False),
        nn.Conv1d(256, 256, 15, stride=1, padding=7),
        nn.LeakyReLU(inplace=False),
        nn.Conv1d(256, 512, 21, stride=1, padding=10),
        nn.Dropout(0.5),
        )

    def forward(self, src, time):
        
        #src = [src sent len, batch size]
        src1 = src.permute(0, 2, 1)
        src11 = time.permute(0, 2, 1)
        #print(src1.size())
        src2 = self.cnn(src1)
        outputs, hidden = self.rnn(src)
        outputs = outputs[:,::30,:]
        src22 = outputs.permute(0, 2, 1)
        src222 = self.cnn_ACT(src11)
        #print src2.size()
        #outputs2 = outputs.permute(0, 2, 1)
        #src3= torch.cat((src11 src2),1)
        src3= torch.cat((src2, src22, src222),1)
        src4 = self.cnn3(src3)
        src4= torch.cat((src4, src11[:,:,:]),1)
        src5 = src4.permute(0, 2, 1)
        

        return src5
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim * 1) + dec_hid_dim, dec_hid_dim, bias=True)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        #repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        

        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        energy = energy.permute(0, 2, 1)
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        

        attention = torch.bmm(v, energy).squeeze(1)
        
        return F.softmax(attention, dim=1)



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        
        self.rnn = nn.GRU(512+5, enc_hid_dim, 1, bidirectional = False, dropout=0.2, batch_first = True)
        
        self.fc = nn.Sequential( 
            nn.Linear(enc_hid_dim * 1, dec_hid_dim, bias=True),
        )
     
    def forward(self, src):
        
        #src = [src sent len, batch size]
        


        outputs, hidden = self.rnn(src)


        hidden = self.fc(hidden[-1,:,:])

        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        self.rnn = nn.GRU((enc_hid_dim * 1 ) + emb_dim, dec_hid_dim,1, bidirectional = False, batch_first = True, dropout=0.2)
        self.out = nn.Sequential( 

            nn.Linear((enc_hid_dim * 1) + dec_hid_dim + emb_dim + 512+5 + 128, output_dim),
            #nn.Dropout(0.5),
        )
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden, encoder_outputs, cnnf, micro):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        
        a = self.attention(hidden, encoder_outputs)
        #a = [batch size, src len]
        a = a.unsqueeze(1)
        #a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(0, 1, 2)
        
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        

        rnn_input = torch.cat((input, weighted), dim = 2)

        rnn_input = rnn_input.permute(1, 0, 2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        output = output.squeeze(1)
        output = output.unsqueeze(0)
        cnnf = cnnf.unsqueeze(0)
        micro=micro.unsqueeze(0)

        output = self.out(torch.cat((output, weighted, input, cnnf, micro), dim = 2))

        return output, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device = 0):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.rnn = Encode_micro()
        print(self.device)

        
    def forward(self, src, src2, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden = self.encoder(src)

        input = trg[:,0]


        

        
        for t in range(0, max_len):
            out, hid = self.rnn(src2[:,t*30:(t+1)*30,:])
            output, hidden = self.decoder(input, hidden, encoder_outputs, src[:,t,:],hid)
            if output.size()[0] == 1:
                outputs[:,t,:] = output[:,:]
            else:
                outputs[:,t,:] = output[:,0,:]
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:,t] if teacher_force else output.squeeze(0)
        

        return outputs