import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def cmtable(y_true, y_pred, label_map, title=None):
    """Returns a confusion matrix table
    """
    cm = confusion_matrix(y_true, y_pred)
    cm = 100. * cm / cm.sum(axis=1)
    # Invert label map
    d = {}
    for k,v in label_map.items():
        d[v] = k
    labels = map(d.get, range(len(d)))
    #ax = sns.heatmap(cm / cm.sum(axis=1, dtype=float),
    ax = sns.heatmap(cm,
                     annot=True,
                     fmt='0.2f',
                     cmap=plt.cm.afmhot_r,
                     #cmap=plt.cm.Blues,
                     xticklabels=labels,
                     yticklabels=labels,
                     linewidths=0.5,
                     cbar=False)
    ax.tick_params(length=0, labeltop=True, labelbottom=False)
    ax.set_frame_on(True)
    if title: ax.set_title(title, fontweight='bold')
    return ax

if __name__ == '__main__':
    sns.set()
    y_true = np.random.randint(0,3, 100)
    y_pred = np.random.randint(0,3, 100)
    label_map = {'foo':0, 'bar':1, 'baz':2}
    cmtable(y_true, y_pred, label_map, 'Confusion Matrix\n')
    plt.show()
    #cm = confusion_matrix(y_true, y_pred)
    #df = pd.DataFrame(cm, columns=['foo','bar','baz'], index=['foo','bar','baz'])
    #probs = 1. * df / df.sum(axis=1)
    #df['pcc'] = probs.values.diagonal()
    #s = df.style
    #with open('foo.html', 'w') as f:
    #    f.write(s._repr_html_())
    exit(0)
