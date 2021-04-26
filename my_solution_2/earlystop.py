import torch

import config as cnf


class EarlyStop(object):
    def __init__(self, patience=7):
        super(EarlyStop, self).__init__()
        self.best_res = {
            "acc": 0.0,
            "prec": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "ap": 0.0,
        }
        self.patience = patience
        self.count = 0

    def check_earlystop(self, res, model):
        if res['auc'] >= self.best_res['auc']:
            self.best_res = res
            path = cnf.checkpoint_path + '{}_lr_{}.pth'.format(cnf.run_type, cnf.lr)
            torch.save(model, path)
            self.count = 0
        else:
            self.count += 1

        if self.count >= self.patience:
            print("best_res: {}".format(self.best_res))
            print("------------------Early Stop Train------------------")
            return True
        else:
            return False
