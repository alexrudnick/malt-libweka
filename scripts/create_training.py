#!/usr/bin/env python3

SIZES="1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000".split()

from subprocess import call

pairs = \
[("swedish","/home/alexr/corpora/conll2006/data/swedish/talbanken05/train/swedish_talbanken05_train.conll"),
 ("portuguese","/home/alexr/corpora/conll2006/data/portuguese/bosque/treebank/portuguese_bosque_train.conll"),
 ("danish","/home/alexr/corpora/conll2006/data/danish/ddt/train/danish_ddt_train.conll"),
 ("dutch","/home/alexr/corpora/conll2006/data/dutch/alpino/train/dutch_alpino_train.conll")]

for (langname, treebank) in pairs:
    for size in SIZES:
        call(["scripts/first_n_sentences.py", treebank, size, langname])
