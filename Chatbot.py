# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:36:20 2020

@author: Will
"""

import nltk
import numpy as mp
import random 
import string

f = open('chatbot.txt', 'r', errors = 'ignore')

raw = f.read()

raw = raw.lower()

#nltk.download('punkt')
#nltk.download('wordnet')

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

print (sent_tokens[:2])
