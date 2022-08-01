import torch
import random
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState
from tqdm import tqdm
import numpy as np
import pickle as pkl


class PreQuatE(Model):
    def __init__(self, config):
        super(PreQuatE, self).__init__(config)
        self.emb_s_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_x_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_y_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_z_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_s_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_x_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_y_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_z_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.Softplus()
        self.aug_h = []
        self.aug_hr = []
        self.aug_t = []
        self.aug_tr = []
        self.aug_ent = {}
        self.isAugmented = False
        self.aug_dict = pkl.load(open(self.config.rdict_path, 'rb'))
        self.t_rel_dict = pkl.load(open(self.config.tdict_path, 'rb'))
        self.h_rel_dict = pkl.load(open(self.config.hdict_path, 'rb'))
        self.filter_lmbda = self.config.filter_lmbda
        self.por_rel_mode = self.config.por_rel_mode
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
        if self.config.nextStage == 1:
            pre_model = torch.load(self.config.premodel_path, map_location=torch.device('cuda'))
            self.emb_s_a.weight.data = pre_model['emb_s_a.weight']
            self.emb_x_a.weight.data = pre_model['emb_x_a.weight']
            self.emb_y_a.weight.data = pre_model['emb_y_a.weight']
            self.emb_z_a.weight.data = pre_model['emb_z_a.weight']
            self.rel_s_b.weight.data = pre_model['rel_s_b.weight']
            self.rel_x_b.weight.data = pre_model['rel_x_b.weight']
            self.rel_y_b.weight.data = pre_model['rel_y_b.weight']
            self.rel_z_b.weight.data = pre_model['rel_z_b.weight']
            del pre_model
        else:
            nn.init.xavier_uniform_(self.emb_s_a.weight.data)
            nn.init.xavier_uniform_(self.emb_x_a.weight.data)
            nn.init.xavier_uniform_(self.emb_y_a.weight.data)
            nn.init.xavier_uniform_(self.emb_z_a.weight.data)
            nn.init.xavier_uniform_(self.rel_s_b.weight.data)
            nn.init.xavier_uniform_(self.rel_x_b.weight.data)
            nn.init.xavier_uniform_(self.rel_y_b.weight.data)
            nn.init.xavier_uniform_(self.rel_z_b.weight.data)

    def _calc(self, s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b):
    
        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b

        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a

        score_r = (A * s_c + B * x_c + C * y_c + D * z_c)
        # print(score_r.size())
        # score_i = A * x_c + B * s_c + C * z_c - D * y_c
        # score_j = A * y_c - B * z_c + C * s_c + D * x_c
        # score_k = A * z_c + B * y_c - C * x_c + D * s_c
        return -torch.sum(score_r, -1)

    def _calc_r(self, o_r_1, o_r_2):
        s_a, x_a, y_a, z_a = o_r_1
        s_c, x_c, y_c, z_c = o_r_2
        score_r = (s_a * s_c + x_a * x_c + y_a * y_c + z_a * z_c)
        return -torch.sum(score_r, -1)

    def loss(self, score, regul, regul2, regul3 = None):
        if regul3 == None:
            return (
                    torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul + self.config.lmbda * regul2
            )
        else:
            return (
                    torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul + self.config.lmbda * regul2 + \
                     self.config.lmbda_two * regul3
            )

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

    def forward(self):
        s_a = self.emb_s_a(self.batch_h)
        x_a = self.emb_x_a(self.batch_h)
        y_a = self.emb_y_a(self.batch_h)
        z_a = self.emb_z_a(self.batch_h)

        s_c = self.emb_s_a(self.batch_t)
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)

        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)
        

        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        regul = (torch.mean( torch.abs(s_a) ** 2)
                 + torch.mean( torch.abs(x_a) ** 2)
                 + torch.mean( torch.abs(y_a) ** 2)
                 + torch.mean( torch.abs(z_a) ** 2)
                 + torch.mean( torch.abs(s_c) ** 2)
                 + torch.mean( torch.abs(x_c) ** 2)
                 + torch.mean( torch.abs(y_c) ** 2)
                 + torch.mean( torch.abs(z_c) ** 2)
                 )
        regul2 =  (torch.mean( torch.abs(s_b) ** 2 )
                 + torch.mean( torch.abs(x_b) ** 2 )
                 + torch.mean( torch.abs(y_b) ** 2 )
                 + torch.mean( torch.abs(z_b) ** 2 ))

        if self.isAugmented:
            sample_num_h = random.sample(range(0, len(self.aug_h)), len(self.aug_h) // self.config.nbatches)
            sample_num_t = random.sample(range(0, len(self.aug_t)), len(self.aug_t) // self.config.nbatches)
            regul3, r_regul = self.sampled_pseudo_loss(sample_num_h, sample_num_t)
            regul2 += r_regul
            return self.loss(score, regul, regul2, regul3)

        return self.loss(score, regul, regul2)

    def predict(self):
        s_a = self.emb_s_a(self.batch_h)
        x_a = self.emb_x_a(self.batch_h)
        y_a = self.emb_y_a(self.batch_h)
        z_a = self.emb_z_a(self.batch_h)

        s_c = self.emb_s_a(self.batch_t)
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)

        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)
        
        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        return score.cpu().data.numpy()

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
        for dh, dr in zip(sampled_h, sampled_r):
            key = flag + str(dh) + "_" + str(dr)
            o_t = self.aug_ent[key]
            o_t_1, o_t_2, o_t_3, o_t_4 = o_t   
            e_1_t.append(o_t_1)
            e_2_t.append(o_t_2)
            e_3_t.append(o_t_3)
            e_4_t.append(o_t_4)
        e_1_t = torch.cat(e_1_t, 0)
        e_2_t = torch.cat(e_2_t, 0)
        e_3_t = torch.cat(e_3_t, 0)
        e_4_t = torch.cat(e_4_t, 0)
        e_1_h = self.emb_s_a(sampled_pseudo_h)
        e_2_h = self.emb_x_a(sampled_pseudo_h)
        e_3_h = self.emb_y_a(sampled_pseudo_h)
        e_4_h = self.emb_z_a(sampled_pseudo_h)
        r_1 = self.rel_s_b(sampled_pseudo_r)
        r_2 = self.rel_x_b(sampled_pseudo_r)
        r_3 = self.rel_y_b(sampled_pseudo_r)
        r_4 = self.rel_z_b(sampled_pseudo_r)

        # label = torch.ones_like(candidate_h)
        if flag == 'h':
            score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h,
                  e_1_t, e_2_t, e_3_t, e_4_t, 
                  r_1, r_2, r_3, r_4)
        else:
            score = self._calc(e_1_t, e_2_t, e_3_t, e_4_t,
                  e_1_h, e_2_h, e_3_h, e_4_h,
                  r_1, r_2, r_3, r_4)

        score_loss = torch.mean(self.criterion(score))
        regul3 = (torch.mean(torch.abs(r_1) ** 2)
                  + torch.mean(torch.abs(r_2) ** 2)
                  + torch.mean(torch.abs(r_3) ** 2)
                  + torch.mean(torch.abs(r_4) ** 2))
        return score_loss, regul3


    def topK_ent_rel(self, dh):
        with torch.no_grad():
            e_1_h = self.emb_s_a.weight.data
            e_2_h = self.emb_x_a.weight.data
            e_3_h = self.emb_y_a.weight.data
            e_4_h = self.emb_z_a.weight.data
            selected_h = [dh for i in range(self.config.entTotal)]
            selected_h = torch.from_numpy(np.array(selected_h))
            selected_h = selected_h.cuda()
            o_h_1 = self.emb_s_a(selected_h)
            o_h_2 = self.emb_x_a(selected_h)
            o_h_3 = self.emb_y_a(selected_h)
            o_h_4 = self.emb_z_a(selected_h)
            score = self._calc_r((e_1_h, e_2_h, e_3_h, e_4_h),
                                 (o_h_1, o_h_2, o_h_3, o_h_4))
            
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
            o_t_1, o_t_2, o_t_3, o_t_4 = self.topK_center_ents(dh, candidate_h, candidate_t)
            self.aug_ent[key] = (o_t_1, o_t_2, o_t_3, o_t_4)

    # 计算关系dr下dh的K个最邻近h对应的t的中心点
    def topK_center_ents(self, dh, candidate_h, candidate_t):
        with torch.no_grad():
            e_1_h = self.emb_s_a(candidate_h)
            e_2_h = self.emb_x_a(candidate_h)
            e_3_h = self.emb_y_a(candidate_h)
            e_4_h = self.emb_z_a(candidate_h)
            selected_h = [dh for i in range(len(candidate_h))]
            selected_h = torch.from_numpy(np.array(selected_h))
            selected_h = selected_h.cuda()
            # 相同的h
            o_h_1 = self.emb_s_a(selected_h)
            o_h_2 = self.emb_x_a(selected_h)
            o_h_3 = self.emb_y_a(selected_h)
            o_h_4 = self.emb_z_a(selected_h)
            score = self._calc_r((e_1_h, e_2_h, e_3_h, e_4_h),
                                 (o_h_1, o_h_2, o_h_3, o_h_4))

            if len(score) > 10:
            # 与头实体最相近的同r下的5个h实体
                vals, indices = score.topk(k=10, dim=0, largest=False)
            elif len(score) > 5:
                vals, indices = score.topk(k=5, dim=0, largest=False)
            elif len(score) > 3:
                vals, indices = score.topk(k=3, dim=0, largest=False)
            else:
                vals, indices = score.topk(k=1, dim=0, largest=False)

            sampled_pseudo_t = candidate_t[indices.tolist()]

            # 最邻近的十个头实体对应尾实体的中心点
            o_c_1 = self.emb_s_a(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_2 = self.emb_x_a(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_3 = self.emb_y_a(sampled_pseudo_t).mean(axis=0, keepdim=True)
            o_c_4 = self.emb_z_a(sampled_pseudo_t).mean(axis=0, keepdim=True)

        return (o_c_1, o_c_2, o_c_3, o_c_4)
