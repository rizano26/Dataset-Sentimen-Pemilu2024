import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

y_true = ['Neutral','Neutral','Positive','Neutral','Neutral','Negative','Positive','Neutral','Neutral','Neutral','Negative','Neutral','Positive','Neutral','Neutral','Neutral','Positive','Positive','Neutral','Positive','Negative','Neutral','Negative','Positive','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Positive','Neutral','Neutral','Neutral','Positive','Neutral','Neutral','Neutral','Neutral','Negative','Neutral','Positive','Neutral','Neutral','Neutral','Positive','Neutral','Neutral','Neutral','Neutral','Neutral','Positive','Neutral','Neutral','Neutral','Neutral','Neutral','Positive','Positive','Neutral','Positive','Positive','Neutral','Positive','Positive','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Positive','Positive','Neutral','Neutral','Negative','Neutral','Neutral','Positive','Positive','Positive','Positive','Positive','Positive','Neutral','Neutral','Positive','Positive','Positive','Positive','Neutral','Positive','Positive','Positive','Positive','Positive','Positive','Neutral','Positive','Neutral','Positive','Neutral','Positive','Positive','Positive','Positive','Positive','Positive','Neutral','Neutral','Positive','Positive','Positive','Positive','Negative','Positive','Neutral','Positive','Positive','Positive','Negative','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Neutral','Positive','Neutral','Neutral','Positive','Positive','Neutral','Neutral','Neutral','Positive','Neutral','Neutral','Neutral','Positive','Positive','Positive','Neutral','Positive','Positive','Positive','Negative','Neutral','Positive','Positive','Positive','Positive','Neutral','Positive','Positive','Negative','Positive','Positive','Negative','Negative','Neutral','Negative','Neutral','Neutral','Negative','Negative','Negative','Negative','Neutral','Negative','Negative','Negative','Negative','Neutral','Negative','Negative','Negative','Negative','Negative','Neutral','Neutral','Negative','Negative','Negative','Positive','Negative','Neutral','Negative','Negative','Neutral','Neutral','Neutral','Positive','Negative','Negative','Negative','Positive','Neutral','Neutral','Negative','Positive','Negative','Neutral','Neutral','Negative','Positive','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Positive','Negative','Negative','Positive','Negative','Negative','Neutral','Negative','Negative','Negative','Negative','Neutral','Positive','Neutral','Negative','Negative','Negative','Negative','Neutral','Negative','Positive','Negative','Negative','Negative','Negative','Negative','Neutral','Neutral']
y_pred = ['Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Neutral','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Positive','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Neutral','Negative']

classes = ['Postive', 'Negative', 'Neutral']

confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative', 'Neutral'])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cnf_matrix = confusion_matrix(y_true, y_pred, labels=['Positive', 'Negative', 'Neutral'])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Positive', 'Negative', 'Neutral'],
                      title='Confusion matrix, without normalization')

plt.show()
