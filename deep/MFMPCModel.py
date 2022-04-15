from os import remove
import torch
import numpy as np
import math


class MF_MPC(torch.nn.Module):
    def __init__(self, params):
        super(MF_MPC, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']
        self.rating_matrix_tensor = params['rating_matrix_tensor']
        self.train_uid_rate_set = params['train_uid_rate_set']
        self.global_avg = torch.sum(self.rating_matrix_tensor) / torch.nonzero(self.rating_matrix_tensor).size(0)
        
        # costruct model parameter
        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)
        self.user_bias = torch.nn.Embedding(self.num_users, 1)
        self.item_bias = torch.nn.Embedding(self.num_items, 1)
        self.Mr_ik = torch.nn.Embedding(self.num_items,self.latent_dim)
        
        #initiate parameter
        self.user_embedding.weight.data = (torch.rand(self.num_users,self.latent_dim) -  0.5) * 0.01
        self.item_embedding.weight.data = (torch.rand(self.num_items,self.latent_dim) -  0.5) * 0.01
        self.user_bias.weight.data = torch.zeros(self.num_users, 1).float()
        self.item_bias.weight.data = torch.zeros(self.num_items, 1).float()
        self.Mr_ik.weight.data = (torch.rand(self.num_items,self.latent_dim) -  0.5) * 0.01

        
    def forward(self, user_idx, item_idx):
        #calculate UuMpc in {u_id,i_id}
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        UuMpc = torch.zeros(user_idx.size(0),self.latent_dim)
        for i in range(user_idx.size(0)):
            for r in range(1,6):
                rate_dict = self.train_uid_rate_set[user_idx[i].item()][r]
                #remove {i} from Iru
                if item_idx[i] in rate_dict:
                    rate_dict.remove(item_idx[i])
                if len(rate_dict) == 0:
                    continue
                UuMpc[i] += self.Mr_ik(torch.LongTensor(rate_dict)).sum() / math.sqrt(len(rate_dict))
        user_vec += UuMpc

        #rui = (Uu + UuMpc) * Vi + bu + bi + mu
        rui = torch.mul(user_vec, item_vec).sum(dim = 1) + self.user_bias(user_idx).view(-1) + self.item_bias(item_idx).view(-1) + self.global_avg
        rui = torch.clamp(rui,1,5)
        return rui
        
        
        

        





