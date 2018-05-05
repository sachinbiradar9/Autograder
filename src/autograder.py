import nltk
from data import EssayData
from score import Essay

# nltk downloads check wn
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')




# specifies if data picked from training or testing folder
test = True

# Load Essays
essay_data = EssayData(test)

file_contents = []
count = len(essay_data.X)
sv0 = []
sv1 = []
for index,essay_x in enumerate(essay_data.X):
    print index+1, '/', count

    # Make Essay object which calculates all the subscores
    essay = Essay(essay_x, essay_data.P[index])

    # put subscore in formula and get score
    score = (
        2 * essay.sentence_score
        - 2 * essay.misspell_score
        + 0.2 * essay.agreement_score
        + 0.8 * essay.verb_score
        + 2 * essay.sentence_formation_score
        + 2 * essay.text_coherence_score
        + 3 * essay.topic_coherence_score
    )
    # get grade based on score
    grade = 'high' if score >= 40 else 'low'


    # used to plot distribution
    """
    if essay_data.Y[index] == 1:
        sv1.append(score)
    else:
        sv0.append(score)
    """

    file_contents.append(
        essay_data.files[index] + ';' +
        str(essay.sentence_score) + ';' +
        str(essay.misspell_score) + ';' +
        str(essay.agreement_score) + ';' +
        str(essay.verb_score) + ';' +
        str(essay.sentence_formation_score) + ';' +
        str(essay.text_coherence_score) + ';' +
        str(essay.topic_coherence_score) + ';' +
        str(score) + ';' +
        grade + '\n'
    )




# Write the subscores and grades to file
results_file_path = '../output/results.txt'
with open(results_file_path,'w+') as f:
    for content in file_contents:
        f.write(content)


"""
# Plot distribution of attributes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
bins = list(range(50))
plt.title('Final score distribution for high class')
plt.hist(sv1)
plt.figure()
plt.title('Final score distribution for low class')
plt.hist(sv0)
plt.show()
"""
