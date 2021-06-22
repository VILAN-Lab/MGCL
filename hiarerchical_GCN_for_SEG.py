import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F


class GCN_Module(nn.Module):
    def __init__(self, args, vocab):
        super(GCN_Module, self).__init__()

        self.args = args
        self.node_number = 40

        self.src_word_emb = nn.Embedding(vocab, 300, padding_idx=0)
        src_embed = torch.load('data/embedding/embedding_enc.pt')

        self.src_word_emb.weight = nn.Parameter(src_embed)
        self.src_word_emb.weight.requires_grad = True

        self.intra_sent_gcn1 = Intra_sentence_GCN(args.hidden_size, 2, self.node_number)
        self.intra_sent_gcn2 = Intra_sentence_GCN(args.hidden_size, 2, self.node_number+1)
        self.intra_sent_gcn3 = Intra_sentence_GCN(args.hidden_size, 2, self.node_number+2)
        self.intra_sent_gcn4 = Intra_sentence_GCN(args.hidden_size, 2, self.node_number+3)

        self.sent1_level_feat = AttFlatten(args)
        self.sent2_level_feat = AttFlatten(args)
        self.sent3_level_feat = AttFlatten(args)
        self.sent4_level_feat = AttFlatten(args)



    def forward(self, sent1, sent2, sent3, sent4, adj1, adj2, adj3, adj4):

        batch_size, n_words = sent1.size()

        sent1_mask = self.mask_for_sentence(sent1.unsqueeze(-1), 'sent1')
        sent2_mask = self.mask_for_sentence(sent2.unsqueeze(-1), 'sent2')
        sent3_mask = self.mask_for_sentence(sent3.unsqueeze(-1), 'sent3')
        sent4_mask = self.mask_for_sentence(sent4.unsqueeze(-1), 'sent4')

        sent1 = self.src_word_emb(sent1)[:, :n_words]
        sent2 = self.src_word_emb(sent2)[:, :n_words]
        sent3 = self.src_word_emb(sent3)[:, :n_words]
        sent4 = self.src_word_emb(sent4)[:, :n_words]

        sent1_feat_s = self.intra_sent_gcn1(sent1, adj1)

        sent1_feat = self.sent1_level_feat(sent1_feat_s, sent1_mask).unsqueeze(1)

        sent2 = torch.cat((sent2, sent1_feat), dim=1)

        sent2_feat_s = self.intra_sent_gcn2(sent2, adj2)

        sent2_feat = self.sent2_level_feat(sent2_feat_s, sent2_mask).unsqueeze(1)

        sent3 = torch.cat((sent3, sent1_feat, sent2_feat), dim=1)

        sent3_feat_s = self.intra_sent_gcn3(sent3, adj3)

        sent3_feat = self.sent3_level_feat(sent3_feat_s, sent3_mask).unsqueeze(1)

        sent4 = torch.cat((sent4, sent1_feat, sent2_feat, sent3_feat), dim=1)
        sent4_feat = self.intra_sent_gcn4(sent4, adj4)

        encoder_output = sent4_feat

        return sent1_feat_s, sent2_feat_s, sent3_feat_s, sent4_feat

    def mask_for_sentence(self, sentence, sent):

        return (torch.sum(torch.abs(sentence), dim=-1) == 0).unsqueeze(1).unsqueeze(2)


class Intra_sentence_GCN(nn.Module):
    def __init__(self, mem_dim, layers, number_node):
        super(Intra_sentence_GCN, self).__init__()

        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(0.1)

        self.linear_output = weight_norm(nn.Linear(self.mem_dim, self.mem_dim), dim=None)

        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(weight_norm(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim), dim=None))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()

        self.linear_node = nn.Linear(number_node, number_node).cuda()
        self.relu = nn.ReLU()

    def forward(self, gcn_inputs, adj):
        # gcn layer
        denom = (adj.sum(2).unsqueeze(2)).float()

        outputs = gcn_inputs.float()
        cache_list = [outputs]
        output_list = []
        adj = adj.float().cuda()

        for i in range(self.layers):

            Ax = adj.bmm(outputs)
            AxW = self.weight_list[i](Ax)
            AxW = AxW + self.weight_list[i](outputs)
            AxW = AxW / denom
            gAxW = self.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out

class Inter_sentence_GCN(nn.Module):
    def __init__(self, mem_dim, layers, number_node):
        super(Inter_sentence_GCN, self).__init__()

        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(0.1)

        self.linear_output = weight_norm(nn.Linear(self.mem_dim, self.mem_dim), dim=None)

        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(weight_norm(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim), dim=None))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()

        self.linear_node = nn.Linear(number_node, number_node).cuda()
        self.relu = nn.ReLU()

    def forward(self, gcn_inputs, adj):
        # gcn layer
        denom = (adj.sum(2).unsqueeze(2)).float()

        outputs = gcn_inputs.float()
        cache_list = [outputs]
        output_list = []
        adj = adj.float().cuda()

        for i in range(self.layers):

            relative_score = outputs.bmm(outputs.transpose(1, 2))
            relative_score = self.linear_node(relative_score).cuda()
            relative_score = torch.softmax(relative_score, dim=-1)

            adj = adj.bmm(relative_score)
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[i](Ax)
            AxW = AxW + self.weight_list[i](outputs)
            AxW = AxW / denom
            gAxW = self.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out


class AttFlatten(nn.Module):
    def __init__(self, args):
        super(AttFlatten, self).__init__()
        self.args = args

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.d_inner,
            out_size=1,
            dropout_r=0.1,
            use_relu=True
        )

        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, x, x_mask):

        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)

        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.args.flat_glimpses):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        self.fc = FullyConnectedLayer(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FullyConnectedLayer, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x