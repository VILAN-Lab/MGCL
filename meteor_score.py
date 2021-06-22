import numpy as np
from nltk.translate import meteor_score
import time



def compute_score(goldfile, predfile):

    score = []
    with open(goldfile, 'r', encoding='utf-8') as gf, open(predfile, 'r', encoding="utf-8") as pf:
        for gs, ps in zip(gf, pf):
            print('gs:', gs)

            ps = ps.lower()
            print('ps:', ps)
            score_m = meteor_score.single_meteor_score(gs, ps)
            score.append(score_m)
    me_score = round(np.average(score), 4)


    return me_score

if __name__=="__main__":
    print("Preparation for evaluation")

    gold_file = "./data/gen/test.txt"
    pred_file = "./data/gen/gen_story.txt"

    score = compute_score(gold_file, pred_file)
    print('Meteor_score:', score)

