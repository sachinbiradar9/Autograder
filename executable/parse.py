
import csv
import enchant
import nltk
import numpy as np
import os
import re
from pattern.en import parsetree


# Parser
os.environ['STANFORD_PARSER']='/Users/sachin/Downloads/stanford-parser-full-2018-02-27/'
os.environ['STANFORD_MODELS']='/Users/sachin/Downloads/stanford-parser-full-2018-02-27/'
#parser = nltk.parse.stanford.StanfordParser(model_path="/Users/sachin/Downloads/stanford-english-corenlp-2018-02-27-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
parser = nltk.parse.corenlp.CoreNLPParser(url="http://localhost:9000")

def get_modal(tree):
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree and 'S' in subtree.label():
            return get_modal(subtree)
        elif type(subtree) == nltk.tree.Tree and subtree.label() == 'VP':
            if subtree[0].label() == 'MD':
                return ('MD', subtree.leaves()[0])
            elif subtree.height > 2:
                return get_modal(subtree)
            else:
                return False
sent = "I came because I sick"
parse_tree = next(parser.raw_parse(sent))
parse_tree.pretty_print()
modal = get_modal(parse_tree)
print modal
