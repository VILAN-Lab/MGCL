from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import utils.opts as opts
# import models as models
from AttModel import UpDownModel
from data_d.dataloader import *
from data_d.dataloaderraw import *
import utils.eval_utils as eval_utils
import argparse
import utils.misc as utils
import modules.losses as losses
import torch




parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='./log_/model-best.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet152',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='./log_/infos_-best.pkl',
                help='path to infos to evaluate')
parser.add_argument('--only_lang_eval', type=int, default=0,
                help='lang eval on saved results')
parser.add_argument('--force', type=int, default=0,
                help='force to evaluate no matter if there are results available')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

pred_fn = os.path.join('eval_results/', '.saved_pred_'+ opt.id + '_' + opt.split + '.pth')
result_fn = os.path.join('eval_results/', opt.id + '_' + opt.split + '.json')

# Setup the model
opt.vocab = vocab
model = UpDownModel(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = losses.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.dataset.ix_to_word = infos['vocab']


# Set sample options
opt.dataset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
        vars(opt))


path_gen = './data/gen/gen_ending'+'.txt'
file_txt = open(path_gen, 'w')
end_gen = []
for i, story in enumerate(split_predictions):
    story_gen = story['caption']
    end_gen.append(story['caption'])
    file_txt.write(story_gen)
    file_txt.write('\r\n')

# print(end_gen)
file_txt.close()

print('loss: ', loss)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 2:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
