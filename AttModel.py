from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils_m as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from CaptionModel import CaptionModel
from hiarerchical_GCN_for_SEG import GCN_Module

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        print('tgt_vocab_size_att:', self.vocab_size)
        self.input_encoding_size = opt.tgt_vocab_dim
        self.n_src_vocab = 27849
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 40) or opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.encoder_hid_size = opt.hidden_size

        self.bos_idx = getattr(opt, 'bos_idx', 0)
        self.eos_idx = getattr(opt, 'eos_idx', 0)
        self.pad_idx = getattr(opt, 'pad_idx', 0)

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0

        self.gcn = GCN_Module(opt, self.n_src_vocab)

        self.embed = nn.Embedding(self.vocab_size+1, self.input_encoding_size)
        embed = torch.load("./data/embedding/embedding_enc.pt")
        self.embed.weight = nn.Parameter(embed)
        self.embed.weight.requires_grad = True

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        self.encoder_embed = nn.Sequential(nn.Linear(self.encoder_hid_size, self.rnn_size))

        self.encoder_att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.encoder_hid_size),) if self.use_bn else ()) +
                                    (nn.Linear(self.encoder_hid_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x, y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k,v in self.vocab.items() if v in bad_endings]

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        fc_feats = self.fc_embed(fc_feats)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _prepare_encoder_out(self, encoder_output):

        batch, seq, dim = encoder_output.size()
        m_encoder = torch.mean(encoder_output, 1, True)
        m_encoder = m_encoder.view(-1, dim)
        p_m_encoder = self.encoder_embed(m_encoder)
        p_att_encoder = self.encoder_att_embed(encoder_output)
        return p_m_encoder, p_att_encoder

    def _forward(self, fc_feats, att_feats, seq, src1, src2, src3, src4, adj1, adj2, adj3, adj4, att_masks=None):

        sent1, sent2, sent3, sent4 = self.gcn(src1, src2, src3, src4, adj1, adj2, adj3, adj4)


        p_enc_output1, p_enc_att_output1 = self._prepare_encoder_out(sent1)
        p_enc_output2, p_enc_att_output2 = self._prepare_encoder_out(sent2)
        p_enc_output3, p_enc_att_output3 = self._prepare_encoder_out(sent3)
        p_enc_output4, p_enc_att_output4 = self._prepare_encoder_out(sent4)

        p_enc_output = p_enc_output1, p_enc_output2, p_enc_output3, p_enc_output4
        p_enc_att_output = p_enc_att_output1, p_enc_att_output2, p_enc_att_output3, p_enc_att_output4



        batch_size = fc_feats.size(0)

        if seq.ndim == 3:
            seq = seq.reshape(-1, seq.shape[2])
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size*seq_per_img)

        outputs = fc_feats.new_zeros(batch_size*seq_per_img, seq.size(1), self.vocab_size+1)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if seq_per_img > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(seq_per_img,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )

        for i in range(seq.size(1)):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample 采样SA
                sample_prob = fc_feats.new(batch_size*seq_per_img).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(p_enc_output, p_enc_att_output, it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, enc_output, p_enc_att_output, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):

        xt = self.embed(it)

        output, state = self.core(enc_output, p_enc_att_output, xt, fc_feats, att_feats, p_att_feats, state, att_masks)   # 此处引入updown模型
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def _old_sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)


        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks = utils.repeat_tensors(beam_size,
                [p_fc_feats[k:k+1], p_att_feats[k:k+1], pp_att_feats[k:k+1], p_att_masks[k:k+1] if att_masks is not None else None]
            )

            for t in range(1):
                if t == 0:
                    it = fc_feats.new_full([beam_size], self.bos_idx, dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.old_beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq[k*sample_n+_n, :] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n, :] = self.done_beams[k][_n]['logps']
            else:
                seq[k, :] = self.done_beams[k][0]['seq']
                seqLogprobs[k, :] = self.done_beams[k][0]['logps']
        return seq, seqLogprobs


    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)

        self.done_beams = [[] for _ in range(batch_size)]
        
        state = self.init_hidden(batch_size)

        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
            [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
        )
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        return seq, seqLogprobs

    def _sample(self, fc_feats, att_feats, src1, src2, src3, src4, adj1, adj2, adj3, adj4, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size*sample_n)

        sent1, sent2, sent3, sent4 = self.gcn(src1, src2, src3, src4, adj1, adj2, adj3, adj4)

        p_enc_output1, p_enc_att_output1 = self._prepare_encoder_out(sent1)
        p_enc_output2, p_enc_att_output2 = self._prepare_encoder_out(sent2)
        p_enc_output3, p_enc_att_output3 = self._prepare_encoder_out(sent3)
        p_enc_output4, p_enc_att_output4 = self._prepare_encoder_out(sent4)

        p_enc_output = p_enc_output1,p_enc_output2,p_enc_output3,p_enc_output4
        p_enc_att_output = p_enc_att_output1,p_enc_att_output2,p_enc_att_output3,p_enc_att_output4


        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )

        trigrams = []
        
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)



        for t in range(self.seq_length + 1):
            if t == 0:
                it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(p_enc_output, p_enc_att_output, it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state, output_logsoftmax=output_logsoftmax)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:, t-1].data.cpu().numpy(), self.bad_endings_ix)
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp


            if block_trigrams and t >= 3:
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3:
                        trigrams.append({prev_two: [current]})
                    elif t > 3:
                        if prev_two in trigrams[i]:
                            trigrams[i][prev_two].append(current)
                        else:
                            trigrams[i][prev_two] = [current]
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                alpha = 2.0
                logprobs = logprobs + (mask * -0.693 * alpha)

            if t == self.seq_length:
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx
                logprobs = logprobs * unfinished.unsqueeze(1).float()
                unfinished = unfinished * (it != self.eos_idx)
            seq[:,t] = it
            seqLogprobs[:,t] = logprobs
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams_table = [[] for _ in range(group_size)]

        seq_table = [fc_feats.new_full((batch_size, self.seq_length), self.pad_idx, dtype=torch.long) for _ in range(group_size)]
        seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in range(self.seq_length + group_size):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.seq_length-1:
                    if t == 0:
                        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
                    else:
                        it = seq[:, t-1]

                    logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state_table[divm]) # changed
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    if divm > 0:
                        unaug_logprobs = logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda
                    
                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    if remove_bad_endings and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                        tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                        logprobs = logprobs + tmp

                    if block_trigrams and t >= 3:
                        prev_two_batch = seq[:,t-3:t-1]
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current  = seq[i][t-1]
                            if t == 3:
                                trigrams.append({prev_two: [current]})
                            elif t > 3:
                                if prev_two in trigrams[i]:
                                    trigrams[i][prev_two].append(current)
                                else:
                                    trigrams[i][prev_two] = [current]
                        prev_two_batch = seq[:,t-2:t]
                        mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i,j] += 1

                        alpha = 2.0
                        logprobs = logprobs + (mask * -0.693 * alpha)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        unfinished = seq[:,t-1] != self.pad_idx & seq[:,t-1] != self.eos_idx
                        it[~unfinished] = self.pad_idx
                        unfinished = unfinished & (it != self.eos_idx) # changed
                    seq[:,t] = it
                    seqLogprobs[:,t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table, 1).reshape(batch_size * group_size, -1)


class UpDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(UpDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.tgt_vocab_dim + opt.rnn_size * 3, opt.rnn_size)
        self.att_lstm_sent = nn.LSTMCell(opt.rnn_size*3, opt.rnn_size)

        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size)
        self.attention = Attention(opt)

    def forward(self, enc_output, enc_att_output, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):

        sent_att1, sent_att2, sent_att3, sent_att4 = enc_att_output

        dec_input_att = torch.cat([sent_att1, sent_att2, sent_att3, sent_att4], dim=1)
        dec_input = torch.mean(dec_input_att, dim=1)

        prev_h = state[0][-1]
        att_lstm_sent1_input = torch.cat([dec_input, prev_h, fc_feats, xt], 1)
        h_att_sent1, c_att_sent1 = self.att_lstm(att_lstm_sent1_input, (state[0][0], state[1][0]))

        att_img_sent1 = self.attention(h_att_sent1, att_feats, p_att_feats, att_masks)
        att_sent1 = self.attention(h_att_sent1, sent_att1, sent_att1, att_masks=None)
        lstm_input_sent2 = torch.cat([att_sent1, h_att_sent1, att_img_sent1], 1)

        h_att_sent2, c_att_sent2 = self.att_lstm_sent(lstm_input_sent2, (h_att_sent1, c_att_sent1))

        att_img_sent2 = self.attention(h_att_sent2, att_feats, p_att_feats, att_masks)
        att_sent2 = self.attention(h_att_sent2, sent_att2, sent_att2, att_masks=None)
        lstm_input_sent3 = torch.cat([ att_sent2, h_att_sent2,att_img_sent2], 1)

        h_att_sent3, c_att_sent3 = self.att_lstm_sent(lstm_input_sent3, (h_att_sent2, c_att_sent2))

        att_img_sent3 = self.attention(h_att_sent3, att_feats, p_att_feats, att_masks)
        att_sent3 = self.attention(h_att_sent3, sent_att3, sent_att3, att_masks=None)
        lstm_input_sent4 = torch.cat([att_sent3, h_att_sent3, att_img_sent3], 1)


        h_att_sent4, c_att_sent4 = self.att_lstm_sent(lstm_input_sent4, (h_att_sent3, c_att_sent3))

        att_img_sent4 = self.attention(h_att_sent4, att_feats, p_att_feats, att_masks)
        att_sent4 = self.attention(h_att_sent4, sent_att4, sent_att4, att_masks=None)
        lang_lstm_input = torch.cat([att_img_sent4, att_sent4, h_att_sent4], 1)


        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att_sent4, h_lang]), torch.stack([c_att_sent4, c_lang]))

        return output, state

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)
        
        weight = F.softmax(dot, dim=1)
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)

        return att_res

class UpDownModel(AttModel):
    def __init__(self, opt):
        super(UpDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = UpDownCore(opt)
