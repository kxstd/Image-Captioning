import sys
from loguru import logger
import numpy as np

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def get_logger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


def ptb_tokenize(key_to_captions):
    captions_for_image = {}
    for key, caps in key_to_captions.items():
        captions_for_image[key] = []
        for idx, cap in enumerate(caps):
            captions_for_image[key].append(
                {
                    # "image_id": key
                    # "id": idx,
                    "caption": cap
                }
            )
    tokenizer = PTBTokenizer()
    key_to_captions = tokenizer.tokenize(captions_for_image)
    return key_to_captions

# class epsilon:
#     def __init__(self, num_epoch, num_batch, eps_mode):
#         self.num_epoch = num_epoch
#         self.num_batch = num_batch
#         self.total = num_epoch * num_batch
#         self.exp = pow(100.0,1.0/self.total)
#         self.linear = 1.0/self.total
#         self.eps_mode = eps_mode

#     def get_epsilon(self, epoch, batch):
#         num_iter = epoch*self.num_batch + batch
#         if self.eps_mode == "None":
#             eps = 1.0
#         if self.eps_mode == "exp":
#             eps = self.exp**num_iter
#         if self.eps_mode == "linear":
#             eps = 1- linear*num_iter
#         if self.eps_mode == "sigmod":
#             eps = 1.0/(1.0 + np.exp(8.0-num_iter*16.0/self.total))
        
#         eps = min(max(0,eps),1)
#         return eps

class epsilon:
    def __init__(self, num_epoch, num_batch, eps_mode):
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.total = num_epoch
        self.exp = pow(0.01,1.0/self.total)
        self.linear = 1.0/self.total
        self.eps_mode = eps_mode

    def get_epsilon(self, epoch, batch):
        num_iter = epoch
        if self.eps_mode == 'None':
            eps = 1.0
        elif self.eps_mode == 'exp':
            eps = self.exp**num_iter
        elif self.eps_mode == 'linear':
            eps = 1- self.linear*num_iter
        elif self.eps_mode == 'sigmod':
            eps = 1.0/(1.0 + np.exp(num_iter*16.0/self.total-8.0))
        else:
            print(self.eps_mode)
        eps = min(max(0,eps),1)
        return eps

