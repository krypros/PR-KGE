import torch
import random
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState
from tqdm import tqdm


class PRDualEVote(Model):
    def __init__(self, config):
        super(PRDualEVote, self).__init__(config)
        self.emb_1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_3 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_4 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_5 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_6 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_7 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_8 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_3 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_4 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_5 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_6 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_7 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_8 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.aug_h = []
        self.aug_hr = []
        self.aug_t = []
        self.aug_tr = []
        self.aug_ent = {}
        self.aug_dict = pkl.load(open(self.config.rdict_path, 'rb'))
        self.h_rel_dict = pkl.load(open(self.config.hdict_path, 'rb'))
        self.t_rel_dict = pkl.load(open(self.config.tdict_path, 'rb'))
        self.filter_lmbda = self.config.filter_lmbda
        self.por_rel_mode = self.config.por_rel_mode
        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.isAugmented = False
        self.init_weights()

    def init_weights(self):
        if self.config.nextStage == 1:
            pre_model = torch.load(self.config.premodel_path, map_location=torch.device('cuda'))
            self.emb_1.weight.data = pre_model['emb_1.weight']
            self.emb_2.weight.data = pre_model['emb_2.weight']
            self.emb_3.weight.data = pre_model['emb_3.weight']
            self.emb_4.weight.data = pre_model['emb_4.weight']
            self.emb_5.weight.data = pre_model['emb_5.weight']
            self.emb_6.weight.data = pre_model['emb_6.weight']
            self.emb_7.weight.data = pre_model['emb_7.weight']
            self.emb_8.weight.data = pre_model['emb_8.weight']
            self.rel_1.weight.data = pre_model['rel_1.weight']
            self.rel_2.weight.data = pre_model['rel_2.weight']
            self.rel_3.weight.data = pre_model['rel_3.weight']
            self.rel_4.weight.data = pre_model['rel_4.weight']
            self.rel_5.weight.data = pre_model['rel_5.weight']
            self.rel_6.weight.data = pre_model['rel_6.weight']
            self.rel_7.weight.data = pre_model['rel_7.weight']
            self.rel_8.weight.data = pre_model['rel_8.weight']
            del pre_model
        else:
            nn.init.xavier_uniform_(self.emb_1.weight.data)
            nn.init.xavier_uniform_(self.emb_2.weight.data)
            nn.init.xavier_uniform_(self.emb_3.weight.data)
            nn.init.xavier_uniform_(self.emb_4.weight.data)
            nn.init.xavier_uniform_(self.emb_5.weight.data)
            nn.init.xavier_uniform_(self.emb_6.weight.data)
            nn.init.xavier_uniform_(self.emb_7.weight.data)
            nn.init.xavier_uniform_(self.emb_8.weight.data)
            nn.init.xavier_uniform_(self.rel_1.weight.data)
            nn.init.xavier_uniform_(self.rel_2.weight.data)
            nn.init.xavier_uniform_(self.rel_3.weight.data)
            nn.init.xavier_uniform_(self.rel_4.weight.data)
            nn.init.xavier_uniform_(self.rel_5.weight.data)
            nn.init.xavier_uniform_(self.rel_6.weight.data)
            nn.init.xavier_uniform_(self.rel_7.weight.data)
            nn.init.xavier_uniform_(self.rel_8.weight.data)


    #Calculate the Dual Hamiltonian product
    def _omult(self, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, c_0, c_1, c_2, c_3, d_0, d_1, d_2, d_3):

        h_0=a_0*c_0-a_1*c_1-a_2*c_2-a_3*c_3
        h1_0=a_0*d_0+b_0*c_0-a_1*d_1-b_1*c_1-a_2*d_2-b_2*c_2-a_3*d_3-b_3*c_3
        h_1=a_0*c_1+a_1*c_0+a_2*c_3-a_3*c_2
        h1_1=a_0*d_1+b_0*c_1+a_1*d_0+b_1*c_0+a_2*d_3+b_2*c_3-a_3*d_2-b_3*c_2
        h_2=a_0*c_2-a_1*c_3+a_2*c_0+a_3*c_1
        h1_2=a_0*d_2+b_0*c_2-a_1*d_3-b_1*c_3+a_2*d_0+b_2*c_0+a_3*d_1+b_3*c_1
        h_3=a_0*c_3+a_1*c_2-a_2*c_1+a_3*c_0
        h1_3=a_0*d_3+b_0*c_3+a_1*d_2+b_1*c_2-a_2*d_1-b_2*c_1+a_3*d_0+b_3*c_0

        return  (h_0,h_1,h_2,h_3,h1_0,h1_1,h1_2,h1_3)

    #Normalization of relationship embedding
    def _onorm(self,r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8):
        denominator_0 = r_1 ** 2 + r_2 ** 2 + r_3 ** 2 + r_4 ** 2
        denominator_1 = torch.sqrt(denominator_0)
        #denominator_2 = torch.sqrt(r_5 ** 2 + r_6 ** 2 + r_7 ** 2 + r_8 ** 2)
        deno_cross = r_5 * r_1 + r_6 * r_2 + r_7 * r_3 + r_8 * r_4

        r_5 = r_5 - deno_cross / denominator_0 * r_1
        r_6 = r_6 - deno_cross / denominator_0 * r_2
        r_7 = r_7 - deno_cross / denominator_0 * r_3
        r_8 = r_8 - deno_cross / denominator_0 * r_4

        r_1 = r_1 / denominator_1
        r_2 = r_2 / denominator_1
        r_3 = r_3 / denominator_1
        r_4 = r_4 / denominator_1
        #r_5 = r_5 / denominator_2
        #r_6 = r_6 / denominator_2
        #r_7 = r_7 / denominator_2
        #r_8 = r_8 / denominator_2
        return r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

    #Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def _calc(self, e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 ):

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )

        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = self._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                             r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)


        score_r = (o_1 * e_1_t + o_2 * e_2_t + o_3 * e_3_t + o_4 * e_4_t
                   +  o_5 * e_5_t + o_6 * e_6_t + o_7 * e_7_t + o_8 * e_8_t)

        return -torch.sum(score_r, -1)


    def loss(self, score, regul, regul2, regul3 = None):
        if regul3 == None:
            return (
                torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul + \
                 self.config.lmbda * regul2
            )
        else:
            return (
                torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul + \
                 self.config.lmbda * regul2 + self.config.lmbda_two * regul3
            )


    def forward(self):
        e_1_h = self.emb_1(self.batch_h)
        e_2_h = self.emb_2(self.batch_h)
        e_3_h = self.emb_3(self.batch_h)
        e_4_h = self.emb_4(self.batch_h)
        e_5_h = self.emb_5(self.batch_h)
        e_6_h = self.emb_6(self.batch_h)
        e_7_h = self.emb_7(self.batch_h)
        e_8_h = self.emb_8(self.batch_h)

        e_1_t = self.emb_1(self.batch_t)
        e_2_t = self.emb_2(self.batch_t)
        e_3_t = self.emb_3(self.batch_t)
        e_4_t = self.emb_4(self.batch_t)
        e_5_t = self.emb_5(self.batch_t)
        e_6_t = self.emb_6(self.batch_t)
        e_7_t = self.emb_7(self.batch_t)
        e_8_t = self.emb_8(self.batch_t)

        r_1 = self.rel_1(self.batch_r)
        r_2 = self.rel_2(self.batch_r)
        r_3 = self.rel_3(self.batch_r)
        r_4 = self.rel_4(self.batch_r)
        r_5 = self.rel_5(self.batch_r)
        r_6 = self.rel_6(self.batch_r)
        r_7 = self.rel_7(self.batch_r)
        r_8 = self.rel_8(self.batch_r)

        score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )
        regul = (torch.mean(torch.abs(e_1_h) ** 2)
                 + torch.mean(torch.abs(e_2_h) ** 2)
                 + torch.mean(torch.abs(e_3_h) ** 2)
                 + torch.mean(torch.abs(e_4_h) ** 2)
                 + torch.mean(torch.abs(e_5_h) ** 2)
                 + torch.mean(torch.abs(e_6_h) ** 2)
                 + torch.mean(torch.abs(e_7_h) ** 2)
                 + torch.mean(torch.abs(e_8_h) ** 2)
                 + torch.mean(torch.abs(e_1_t) ** 2)
                 + torch.mean(torch.abs(e_2_t) ** 2)
                 + torch.mean(torch.abs(e_3_t) ** 2)
                 + torch.mean(torch.abs(e_4_t) ** 2)
                 + torch.mean(torch.abs(e_5_t) ** 2)
                 + torch.mean(torch.abs(e_6_t) ** 2)
                 + torch.mean(torch.abs(e_7_t) ** 2)
                 + torch.mean(torch.abs(e_8_t) ** 2)
                 )
        regul2 = (torch.mean(torch.abs(r_1) ** 2)
                  + torch.mean(torch.abs(r_2) ** 2)
                  + torch.mean(torch.abs(r_3) ** 2)
                  + torch.mean(torch.abs(r_4) ** 2)
                  + torch.mean(torch.abs(r_5) ** 2)
                  + torch.mean(torch.abs(r_6) ** 2)
                  + torch.mean(torch.abs(r_7) ** 2)
                  + torch.mean(torch.abs(r_8) ** 2))

        if self.isAugmented:
            sample_num_h = random.sample(range(0, len(self.aug_h)), len(self.aug_h) // self.config.nbatches)
            sample_num_t = random.sample(range(0, len(self.aug_t)), len(self.aug_t) // self.config.nbatches)
            regul3, r_regul = self.sampled_pseudo_loss(sample_num_h, sample_num_t)
            regul2 += r_regul
            return self.loss(score, regul, regul2, regul3)

        return self.loss(score, regul, regul2)


    def predict(self):
        e_1_h = self.emb_1(self.batch_h)
        e_2_h = self.emb_2(self.batch_h)
        e_3_h = self.emb_3(self.batch_h)
        e_4_h = self.emb_4(self.batch_h)
        e_5_h = self.emb_5(self.batch_h)
        e_6_h = self.emb_6(self.batch_h)
        e_7_h = self.emb_7(self.batch_h)
        e_8_h = self.emb_8(self.batch_h)

        e_1_t = self.emb_1(self.batch_t)
        e_2_t = self.emb_2(self.batch_t)
        e_3_t = self.emb_3(self.batch_t)
        e_4_t = self.emb_4(self.batch_t)
        e_5_t = self.emb_5(self.batch_t)
        e_6_t = self.emb_6(self.batch_t)
        e_7_t = self.emb_7(self.batch_t)
        e_8_t = self.emb_8(self.batch_t)

        r_1 = self.rel_1(self.batch_r)
        r_2 = self.rel_2(self.batch_r)
        r_3 = self.rel_3(self.batch_r)
        r_4 = self.rel_4(self.batch_r)
        r_5 = self.rel_5(self.batch_r)
        r_6 = self.rel_6(self.batch_r)
        r_7 = self.rel_7(self.batch_r)
        r_8 = self.rel_8(self.batch_r)

        score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )
        return score.cpu().data.numpy()


    def triple_augment(self):
        self.isAugmented = True
        # aug_h_list = pkl.load(open('./checkpoint/miss_triple_list.pkl', 'rb'))
        for dh in tqdm(range(self.config.entTotal)):
            selected_hr, selected_tr = self.topK_ent_rel(dh)
            if dh in self.h_rel_dict:
                for dr in selected_hr:
                    if dr not in self.h_rel_dict[dh]:
                        self.aug_h.append(dh)
                        self.aug_hr.append(dr)
                        self.pred_ent_t(dh, dr, 'h')
            else:
                for dr in selected_hr:
                    self.aug_h.append(dh)
                    self.aug_hr.append(dr)
                    self.pred_ent_t(dh, dr, 'h')
            if dh in self.t_rel_dict:
                for dr in selected_tr:
                    if dr not in self.t_rel_dict[dh]:
                        self.aug_t.append(dh)
                        self.aug_tr.append(dr)
                        self.pred_ent_t(dh, dr, 't')
            else:
                for dr in selected_tr:
                    self.aug_t.append(dh)
                    self.aug_tr.append(dr)
                    self.pred_ent_t(dh, dr, 't')
        print("burry entities: ", len(set(self.aug_h + self.aug_t)))
        print("Add Head triples: ", len(self.aug_h))
        print("Add Tail triples: ", len(self.aug_t))
        self.aug_h = torch.from_numpy(np.array(self.aug_h))
        self.aug_h = self.aug_h.cuda()
        self.aug_t = torch.from_numpy(np.array(self.aug_t))
        self.aug_t = self.aug_t.cuda()
        self.aug_hr = torch.from_numpy(np.array(self.aug_hr))
        self.aug_hr = self.aug_hr.cuda()
        self.aug_tr = torch.from_numpy(np.array(self.aug_tr))
        self.aug_tr = self.aug_tr.cuda()


    def _calc_r(self, o_r_1, o_r_2):
        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = o_r_1
        o_1_r, o_2_r, o_3_r, o_4_r, o_5_r, o_6_r, o_7_r, o_8_r = o_r_2
        score_r = (o_1 * o_1_r + o_2 * o_2_r + o_3 * o_3_r + o_4 * o_4_r
                   + o_5 * o_5_r + o_6 * o_6_r + o_7 * o_7_r + o_8 * o_8_r)
        return -torch.sum(score_r, -1)


    def sampled_pseudo_loss(self, sample_num_h, sample_num_t):
        sampled_pseudo_h = self.aug_h[sample_num_h]
        sampled_pseudo_t = self.aug_t[sample_num_t]
        sampled_pseudo_hr = self.aug_hr[sample_num_h]
        sampled_pseudo_tr = self.aug_tr[sample_num_t]
        if self.por_rel_mode == 'a':
            score_h_loss, regul_h_3 = self._calc_pseudo(sampled_pseudo_h, sampled_pseudo_hr, 'h')
            score_t_loss, regul_t_3 = self._calc_pseudo(sampled_pseudo_t, sampled_pseudo_tr, 't')
            score_loss = (score_h_loss + score_t_loss) / 2
            regul_3 = (regul_h_3 + regul_t_3) / 2
        elif self.por_rel_mode == 'h':
            score_h_loss, regul_h_3 = self._calc_pseudo(sampled_pseudo_h, sampled_pseudo_hr, 'h')
            score_loss = score_h_loss
            regul_3 = regul_h_3
        else:
            score_t_loss, regul_t_3 = self._calc_pseudo(sampled_pseudo_t, sampled_pseudo_tr, 't')
            score_loss = score_t_loss
            regul_3 = regul_t_3
        return score_loss, regul_3
        

    def _calc_pseudo(self, sampled_pseudo_h, sampled_pseudo_r, flag):
        sampled_h = sampled_pseudo_h.tolist()
        sampled_r = sampled_pseudo_r.tolist()
        e_1_t = []
        e_2_t = []
        e_3_t = []
        e_4_t = []
        e_5_t = []
        e_6_t = []
        e_7_t = []
        e_8_t = []
        for dh, dr in zip(sampled_h, sampled_r):
            key = flag + str(dh) + "_" + str(dr)
            o_t = self.aug_ent[key]
            o_t_1, o_t_2, o_t_3, o_t_4, o_t_5, o_t_6, o_t_7, o_t_8 = o_t   
            e_1_t.append(o_t_1)
            e_2_t.append(o_t_2)
            e_3_t.append(o_t_3)
            e_4_t.append(o_t_4)
            e_5_t.append(o_t_5)
            e_6_t.append(o_t_6)
            e_7_t.append(o_t_7)
            e_8_t.append(o_t_8)
        e_1_t = torch.cat(e_1_t, 0)
        e_2_t = torch.cat(e_2_t, 0)
        e_3_t = torch.cat(e_3_t, 0)
        e_4_t = torch.cat(e_4_t, 0)
        e_5_t = torch.cat(e_5_t, 0)
        e_6_t = torch.cat(e_6_t, 0)
        e_7_t = torch.cat(e_7_t, 0)
        e_8_t = torch.cat(e_8_t, 0)
        e_1_h = self.emb_1(sampled_pseudo_h)
        e_2_h = self.emb_2(sampled_pseudo_h)
        e_3_h = self.emb_3(sampled_pseudo_h)
        e_4_h = self.emb_4(sampled_pseudo_h)
        e_5_h = self.emb_5(sampled_pseudo_h)
        e_6_h = self.emb_6(sampled_pseudo_h)
        e_7_h = self.emb_7(sampled_pseudo_h)
        e_8_h = self.emb_8(sampled_pseudo_h)
        r_1 = self.rel_1(sampled_pseudo_r)
        r_2 = self.rel_2(sampled_pseudo_r)
        r_3 = self.rel_3(sampled_pseudo_r)
        r_4 = self.rel_4(sampled_pseudo_r)
        r_5 = self.rel_5(sampled_pseudo_r)
        r_6 = self.rel_6(sampled_pseudo_r)
        r_7 = self.rel_7(sampled_pseudo_r)
        r_8 = self.rel_8(sampled_pseudo_r)

        # label = torch.ones_like(candidate_h)
        if flag == 'h':
            score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                  e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
                  r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )
        else:
            score = self._calc(e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
                  e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                  r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 )

        score_loss = torch.mean(self.criterion(score))
        regul3 = (torch.mean(torch.abs(r_1) ** 2)
                  + torch.mean(torch.abs(r_2) ** 2)
                  + torch.mean(torch.abs(r_3) ** 2)
                  + torch.mean(torch.abs(r_4) ** 2)
                  + torch.mean(torch.abs(r_5) ** 2)
                  + torch.mean(torch.abs(r_6) ** 2)
                  + torch.mean(torch.abs(r_7) ** 2)
                  + torch.mean(torch.abs(r_8) ** 2))
        return score_loss, regul3


    def topK_ent_rel(self, dh):
        with torch.no_grad():
            e_1_h = self.emb_1.weight.data
            e_2_h = self.emb_2.weight.data
            e_3_h = self.emb_3.weight.data
            e_4_h = self.emb_4.weight.data
            e_5_h = self.emb_5.weight.data
            e_6_h = self.emb_6.weight.data
            e_7_h = self.emb_7.weight.data
            e_8_h = self.emb_8.weight.data
            selected_h = [dh for i in range(self.config.entTotal)]
            selected_h = torch.from_numpy(np.array(selected_h))
            selected_h = selected_h.cuda()
            o_h_1 = self.emb_1(selected_h)
            o_h_2 = self.emb_2(selected_h)
            o_h_3 = self.emb_3(selected_h)
            o_h_4 = self.emb_4(selected_h)
            o_h_5 = self.emb_5(selected_h)
            o_h_6 = self.emb_6(selected_h)
            o_h_7 = self.emb_7(selected_h)
            o_h_8 = self.emb_8(selected_h)
            score = self._calc_r((e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h),
                                 (o_h_1, o_h_2, o_h_3, o_h_4, o_h_5, o_h_6, o_h_7, o_h_8))
            
            vals, indices = score.topk(k=10, dim=0, largest=False)
            noh_rels = []
            noh_rels_list = []
            not_rels = []
            not_rels_list = []
            
            for h_i in indices.tolist():
                noh_rels_list.extend(tuple(self.h_rel_dict[h_i]))
            for ent in set(noh_rels_list):
                if noh_rels_list.count(ent) >= self.filter_lmbda * 10:
                    noh_rels.append(ent)
            
            for h_i in indices.tolist():
                not_rels_list.extend(tuple(self.t_rel_dict[h_i]))
            for ent in set(not_rels_list):
                if not_rels_list.count(ent) >= self.filter_lmbda * 10:
                    not_rels.append(ent)
        return noh_rels, not_rels

    def pred_ent_t(self, dh, dr, flag):
        with torch.no_grad():
            key = flag + str(dh) + "_" + str(dr)
            if flag == 'h':
                candidate_h = self.aug_dict[dr][0]
                candidate_t = self.aug_dict[dr][1]
            else:
                candidate_h = self.aug_dict[dr][1]
                candidate_t = self.aug_dict[dr][0]
            candidate_h = torch.from_numpy(np.array(candidate_h))
            candidate_h = candidate_h.cuda()
            candidate_t = torch.from_numpy(np.array(candidate_t))
            candidate_t = candidate_t.cuda()
            o_t_1, o_t_2, o_t_3, o_t_4, o_t_5, o_t_6, o_t_7, o_t_8 = self.topK_center_ents(dh, candidate_h, candidate_t)
            self.aug_ent[key] = (o_t_1, o_t_2, o_t_3, o_t_4, o_t_5, o_t_6, o_t_7, o_t_8)

    # Calculate the centroid of t corresponding to the K nearest neighbours h of dh under the relation dr
    def topK_center_ents(self, dh, candidate_h, candidate_t):
        with torch.no_grad():
            e_1_h = self.emb_1(candidate_h)
            e_2_h = self.emb_2(candidate_h)
            e_3_h = self.emb_3(candidate_h)
            e_4_h = self.emb_4(candidate_h)
            e_5_h = self.emb_5(candidate_h)
            e_6_h = self.emb_6(candidate_h)
            e_7_h = self.emb_7(candidate_h)
            e_8_h = self.emb_8(candidate_h)
            selected_h = [dh for i in range(len(candidate_h))]
            selected_h = torch.from_numpy(np.array(selected_h))
            selected_h = selected_h.cuda()
            # Same h
            o_h_1 = self.emb_1(selected_h)
            o_h_2 = self.emb_2(selected_h)
            o_h_3 = self.emb_3(selected_h)
            o_h_4 = self.emb_4(selected_h)
            o_h_5 = self.emb_5(selected_h)
            o_h_6 = self.emb_6(selected_h)
            o_h_7 = self.emb_7(selected_h)
            o_h_8 = self.emb_8(selected_h)
            score = self._calc_r((e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h),
                                 (o_h_1, o_h_2, o_h_3, o_h_4, o_h_5, o_h_6, o_h_7, o_h_8))

            if len(score) > 10:
            # n h-entities under the same r that are closest to the head entity
                vals, indices = score.topk(k=10, dim=0, largest=False)
            elif len(score) > 5:
                vals, indices = score.topk(k=5, dim=0, largest=False)
            elif len(score) > 3:
                vals, indices = score.topk(k=3, dim=0, largest=False)
            else:
                vals, indices = score.topk(k=1, dim=0, largest=False)

            sampled_pseudo_t = candidate_t[indices.tolist()]

            # The ten nearest head entities correspond to the centroids of the tail entities
            o_c_1 = self.emb_1(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_2 = self.emb_2(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_3 = self.emb_3(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_4 = self.emb_4(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_5 = self.emb_5(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_6 = self.emb_6(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_7 = self.emb_7(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_8 = self.emb_8(sampled_pseudo_t).mean(axis=0, keepdim=True)

        return (o_c_1, o_c_2, o_c_3, o_c_4, o_c_5, o_c_6, o_c_7, o_c_8)
