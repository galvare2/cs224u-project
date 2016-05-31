import matplotlib.pyplot as plt
import numpy as np

RANDOM = 0.500
INTERPLAY = 0.565
DISCOURSE = 0.504
PRONOUNS = 0.515
FORMATTING = 0.534
BEST = 0.604
PAPER = 0.650

print "hi"
N = 7
results = (RANDOM, INTERPLAY, DISCOURSE, PRONOUNS, FORMATTING, BEST, PAPER)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, results, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score by Feature Set')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Random', 'Interplay', 'Discourse', 'Pronouns', 'Formatting', 'Best', 'Paper'))

plt.show()