import csv
import enchant
import nltk
import numpy as np
import os
import re
from pattern.en import parsetree

test = True

#nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Parser
parser = nltk.parse.corenlp.CoreNLPParser(url="http://localhost:9000")

# Enchant
dictionary_us = enchant.Dict("en_US")
dictionary_uk = enchant.Dict("en_UK")
for words_file in os.listdir('./resources/word_lists/'):
     words = open('./resources/word_lists/' + words_file, 'r').read().splitlines()
     for word in words:
         dictionary_us.add_to_session(word.lower())


class EssayData:
    def __init__(self):
        if test:
            self.X, self.P, self.files  = self.load_essays('../input/testing/')
        else:
            self.X, self.P, self.Y, self.files  = self.load_essays('../input/training/')


    def load_essays(self, filename):
        X = []; P = []; Y = []; files = []
        label_dict = {}

        with open(filename + 'index.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                prompt = re.search('\t\t(.*)\t\t', row['prompt']).group(1)
                if test:
                    label_dict[row['filename']] = [prompt]
                else:
                    label_dict[row['filename']] = [prompt, row['grade']]

        for essay_file in os.listdir(filename + 'essays/'):
            essay = open(filename + '/essays/' + essay_file, 'r').read()
            X.append(essay)
            P.append(label_dict[essay_file][0])
            if not test:
                Y.append(1 if label_dict[essay_file][1] == 'high' else -1)
            files.append(essay_file)
        if test:
            return np.array(X), np.array(P), files
        else:
            return np.array(X), np.array(P), np.array(Y), files




class Essay:
    def __init__(self, essay):
        self.essay = essay

        self.sents = self.get_sentences()
        self.sentence_score = self.compute_sentence_score()

        self.misspells = self.get_misspells()
        self.misspell_score = self.compute_misspell_score()

        self.agreement_errors = self.get_agreement_errors()
        self.agreement_score = self.compute_agreement_score()

        self.verb_errors = self.count_verb_errors()
        self.verb_score = self.compute_verb_score()


    def apply_hash(self, text, reverse=False):
        if not reverse:
            replace_dict = {"e.g.": "e#g#", "i.e.": "i#e#", "Dr.": "Dr#", "Mr.": "Mr#", "Mrs.": "Mrs#", "Sr.": "Sr#", "Jr.": "Jr#", "etc.": "etc#"}
        else:
            replace_dict = {"e#g#": "e.g.", "i#e#": "i.e.", "Dr#": "Dr.", "Mr#": "Mr.", "Mrs#": "Mrs.", "Sr#": "Sr.", "Jr#": "Jr.", "etc#": "etc."}
        replace = dict((re.escape(k), v) for k, v in replace_dict.iteritems())
        pattern = re.compile("|".join(replace.keys()))
        text = pattern.sub(lambda m: replace[re.escape(m.group(0))], text)
        return text


    def get_sentences(self):
        sents = []
        essay = self.apply_hash(self.essay, reverse=False)
        essay = essay.replace('.', ' . ').replace('\n', ' . ')
        nltk_sents = nltk.sent_tokenize(essay)

        for nltk_sent in nltk_sents:
            sent = nltk_sent.strip()
            sents.extend([sent])

        sents = list(filter(lambda x: x!= '.', sents))
        sents = list(map(lambda x: self.apply_hash(x,reverse=True), sents))
        sentences = []
        for sent in sents:
            if "," in sent:
                comma_seaprated = sent.split(",")
                temp = ""
                for chunk in comma_seaprated:
                    if temp:
                        chunk = temp + ", " + chunk
                    temp = ""
                    if chunk.count(" ") > 4:
                        sentences.append(chunk.strip())
                    else:
                        temp = chunk
                if temp:
                    sentences[-1] = sentences[-1] + ", " +temp

            else:
                sentences.append(sent.strip())

        return sentences


    def compute_sentence_score(self):
        no_of_sents = len(self.sents)
        if no_of_sents <= 5:
            return 1
        elif no_of_sents <= 12:
            return 2
        elif no_of_sents <= 18:
            return 3
        elif no_of_sents <= 22:
            return 4
        else:
            return 5


    def get_misspells(self):
        misspells = []
        for sent in self.sents:
            words = nltk.word_tokenize(sent)
            for word in words:
                word = word.lower()
                if not dictionary_us.check(word) and not dictionary_uk.check(word):
                    misspells.append(word)
        return misspells


    def compute_misspell_score(self):
        no_of_mispells = len(self.misspells)
        if no_of_mispells <= 5:
            return 0
        elif no_of_mispells <= 10:
            return 1
        elif no_of_mispells <= 15:
            return 2
        elif no_of_mispells <= 20:
            return 3
        else:
            return 4

    def traverse_tree(self, tree, tag):
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree and (subtree.label().find(tag) == 0 or 'S' in subtree.label() or 'PRP' in subtree.label()):
                if subtree.height() == 2:
                    if 'PRP' in subtree.label() and subtree.leaves()[0].lower() in ['it', 'she', 'he']:
                        return "PRP1", subtree.leaves()[0]
                    if tag == 'V':
                        return (subtree.label(), subtree.leaves()[0])
                    else:
                        return (tree[-1].label(), tree[-1].leaves()[0])
                else:
                    return self.traverse_tree(subtree, tag)


    def get_modal(self, tree):
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree and 'S' in subtree.label():
                return self.get_modal(subtree)
            elif type(subtree) == nltk.tree.Tree and subtree.label() == 'VP':
                if subtree[0].label() == 'MD':
                    return ('MD', subtree.leaves()[0])
                elif subtree.height > 2:
                    return self.get_modal(subtree)
                else:
                    return False


    def get_agreement_errors(self):
        agreement_errors = []
        for sent in self.sents:
            parse_tree = next(parser.raw_parse(sent))
            subject = self.traverse_tree(parse_tree, 'N')
            verb = self.traverse_tree(parse_tree, 'V')
            modal = self.get_modal(parse_tree)
            if not subject or not verb:
                continue

            sv_agreement = ["NNS,VBZ", "NNS,MD,VBZ", "NNPS,VBZ", "NN,VBP", "NNPS,MD,VBP", "NN,MD,VBP", "NN,VB", "NNP,VB", 'PRP,VBZ', 'PRP,MD,VBZ', "PRP$,MD,VB", "PRP$,VB", 'PRP$,VBP', 'PRP$,MD,VBP', 'PRP$,VBZ', 'PRP$,MD,VBZ', 'PRP1,VBP', 'PRP1,MD,VBP', 'PRP1,MD,VBZ']
            if modal:
                tag_sequence = subject[0] + ",MD," + verb[0]
            else:
                tag_sequence = subject[0] + "," + verb[0]

            if subject[1].lower() == "it" and verb[1].lower() == 'is':
                continue
            if tag_sequence in sv_agreement:
                agreement_errors.append(str(subject) + " " + str(modal) + " " + str(verb))

        return agreement_errors


    def compute_agreement_score(self):
        no_of_agreement_errors = len(self.agreement_errors)
        if no_of_agreement_errors <= 1:
            return 5
        elif no_of_agreement_errors <= 2:
            return 4
        elif no_of_agreement_errors <= 3:
            return 3
        elif no_of_agreement_errors <= 4:
            return 2
        else:
            return 1


    def count_verb_errors(self):
        verb_errors = []
        verb_list=["VB","VBZhas","VBZwas","VBZis","MD","VBD","VBG","VBN","VBP"]
        bigram_agreement=["VBD,VBP","MD,VBZis","MD,VBZhas","MD,VBD","MD,VBN","MD,VBG","MD,VBP","VBP,VBP","MD,MD",
                          "MD,VBZis","MD,VBZhas","VB,MD","VB,VBZhas","VBD,MD","VBZis,MD","VBZhas,MD","VBN,MD",
                          "VBN,JJ","VBZ,VB","VBZis,VBZis","VBZis,VB","VBZis,VBP","VBZis,VBD","VBZhas,VBZhas","VBZhas,VBZis","VBZhas,VB",
                          "VBZhas,VBP","VBZhas,VBD","VBZhas,VBG","WDT,VBN"]

        trigram_agreement=["MD,VB,VBD","MD,VB,VBP","MD,VB,VBZ","MD,VB,VBZis","MD,VB,VBZhas","MD,VBZhas,VBN",
                           "MD,VBZhas,VB","MD,VBZhas,VBD","MD,VBZhas,VBN","MD,VBZhas,VBP","MD,VBZhas,VBZhas","MD,VBZis,VBN",
                           "MD,VBZis,VB","MD,VBZis,VBD","MD,VBZis,VBN","MD,VBZis,VBP","MD,VBZis,VBZis","MD,VBN,VBN","MD,VBN,VB",
                           "MD,VBN,VBD","MD,VBN,VBN","MD,VBN,VBP","MD,VBN,VBZis","MD,VBN,VBZhas"]

        tense_agreement=["MD,VB,VB", "VBP,VB", "VBZ,VBG,VBZ", "VB,RB,VBZ", "VBZ,VBG,RG,VB", "VBP,VBP", "TO,VBG", "TO,VBD", "VBP,VB", "VB,VBN", "TO,VB,TO,TO,VB", "VBZ,MD,VB,RP", "VBP,TO,VBG", "VBD,VBG,VBN", "MD,TO,VB", "VBP,VB", "MD,VBN", "MD,VBD", "VBP,RB,VBP,TO,VB", "VBP,RB,VBP,RB,VBP", "VBZ,MD,VB", "RB,VBZ,VB", "RB,VBN,VBG", "TO,VBZ", "MD,MD, VB,TO,VBG", "TO,VBZ,VBZ", "MD,VBZ,TO,VB", "VBP,RB,MD,VB", "VRP,RB,MD,VB" ,"RB,VB,RB,VBN", "VBP,MD,VB", "VBP,VBZ,VBG", "VB,VBP,VBN,TO,VBG", "VBP,VB,VBD", "TO,VBG", "VBP,VBZ,VBG", "VBP,VBG,TO,VB,TO,VB", "MD,VB,VB"]

        sents = self.sents
        for sent in sents:
            x=nltk.word_tokenize(sent)

            tagged = nltk.pos_tag(x)

            disambi=[]
            for i in range(0,len(tagged)):

                if (tagged[i][0] == 'is' and tagged[i][1] == 'VBZ'):
                    tu=(tagged[i][0],'VBZis')
                    disambi.append(tu)

                elif (tagged[i][1] == 'has' and tagged[i][1] == 'VBZ'):
                    tu=(tagged[i][0],'VBZhas')
                    disambi.append(tu)

                elif (tagged[i][1] in ['was','were'] and tagged[i][1] == 'VBD'):
                    tu=(tagged[i][0],'VBDwas')
                    disambi.append(tu)

                else:
                    disambi.append(tagged[i])
            #print disambi

            tags1=[]
            for i in range(0, len(disambi)):
                if "'" in disambi[i][0] or "," in disambi[i][0]:
                    continue
                else:
                    tags1.append(disambi[i])

            #print tags1

            #bigrams of verbs
            bigrams1=zip(tags1,tags1[1:])

            #if tense_error:
                #print tense_error
                #print tags1

            bigrams=[]
            for i in range(0,len(bigrams1)):
                if (bigrams1[i][0][1] in verb_list and  bigrams1[i][1][1] in verb_list):
                    bigrams.append(bigrams1[i])

            #print bigrams

            #trigrams of verbs

            trigrams1=zip(tags1,tags1[1:],tags1[2:])
            #print trigrams1
            trigrams=[]
            for i in range(0,len(trigrams1)):
                if (trigrams1[i][0][1] in verb_list and  trigrams1[i][1][1] in verb_list and  trigrams1[i][2][1] in verb_list):
                    trigrams.append(trigrams1[i])

            #print trigrams

            #Check agreement of bigram and trigram rules w.r.t wrong rules
            rule_error = []
            for i in range(0,len(bigrams)):
                if (bigrams[i][0][1] + "," + bigrams[i][1][1]) in bigram_agreement:
                    rule_error.append(bigrams[i])

            for j in range(0,len(trigrams)):
                if (trigrams[j][0][1] + "," + trigrams[j][1][1] + "," + trigrams[j][2][1]) in trigram_agreement:
                    rule_error.append(trigrams[j])

            verb_errors.extend(rule_error)

            # check tense errors
            tree = parsetree(sent, relations=True)
            chunks = map(lambda x : x.verbs, tree)
            chunks = chunks[0]
            for chunk in chunks:
                verb_chunk = ",".join([x.tag for x in chunk])
                word_chunk = ",".join([x.string for x in chunk])
                if verb_chunk == 'VBP,VB':
                    if chunk[0].string != 'do':
                        verb_errors.append((word_chunk, verb_chunk))
                elif verb_chunk == 'VB,VBN':
                    if chunk[0].string.lower() not in ['be', 'are']:
                        verb_errors.append((word_chunk, verb_chunk))
                elif verb_chunk in tense_agreement:
                    verb_errors.append((word_chunk, verb_chunk))


        return verb_errors

    def compute_verb_score(self):
        no_of_verb_errors = len(self.verb_errors)
        if no_of_verb_errors <= 1:
            return 5
        elif no_of_verb_errors <= 2:
            return 4
        elif no_of_verb_errors <= 3:
            return 3
        elif no_of_verb_errors <= 4:
            return 2
        else:
            return 1


essay_data = EssayData()
file_contents = []

for index,essay_x in enumerate(essay_data.X):
    essay = Essay(essay_x)
    score = 2*essay.sentence_score - essay.misspell_score + essay.agreement_score + essay.verb_score
    grade = 'unknown'

    file_contents.append(
        essay_data.files[index] + ';' +
        str(essay.sentence_score) + ';' +
        str(essay.misspell_score) + ';' +
        str(essay.agreement_score) + ';' +
        str(essay.verb_score) + ';' +
        '0;0;0;' +
        str(score) + ';' +
        grade + '\n'
    )


results_file_path = '../output/results.txt'
with open(results_file_path,'w+') as f:
    for content in file_contents:
        f.write(content)

"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
bins = list(range(6))
plt.title('Agreement error distribution for high class')
plt.hist(sv1, bins=bins)
plt.figure()
plt.title('Agreement error distribution for low class')
plt.hist(sv0, bins=bins)
plt.figure()
plt.title('Verb error distribution for high class')
plt.hist(cv1, bins=bins)
plt.figure()
plt.title('Verb error distribution for low class')
plt.hist(cv0, bins=bins)
plt.show()
"""
