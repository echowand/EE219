import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from sklearn.datasets import fetch_20newsgroups as f20


def plot_by_categories(cat_count, categories):
    train = f20(subset='train', shuffle=True, random_state=42)
    if (cat_count != 20):
        train = f20(subset='train', categories=categories, shuffle=True, random_state=42)
    x = np.arange(0, cat_count + 1, 1)
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(train.target, bins=x, edgecolor='gray')
    ax.set_xticks(bins)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Classes', x=1)
    ax.set_ylabel('Count')
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # Label the raw counts
        ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -18), textcoords='offset points', va='top', ha='center')

        # Label the percentages
        percent = '%0.0001f%%' % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -32), textcoords='offset points', va='top', ha='center')
    plt.title('Distribution of Documents')
    plt.subplots_adjust(bottom=0.15)
    plt.show()


categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

# plot a histogram of the number of documents per topic
plot_by_categories(20, [])
# plot the number of documents in Computer Technology and Recreational Activity
plot_by_categories(8, categories)
