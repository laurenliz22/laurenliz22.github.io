---
layout: post
title:      "Three Visualization Techniques"
date:       2020-01-22 18:45:27 +0000
permalink:  three_visualization_techniques
---


It's true, a picture tells a thousand words!

Visualization in data science is an extremely important tool.  It allows you to tell your data's story as well as illustrate how well your Machine Learning model is working.   

Visualization techniques not only help you as a data scientist explore and analyze your dataset, it also is helpful for non-technical audiences to understand your methodology and results as you explain your work to them. 

There are a ton of types of data visualization techniques you can use.  I'll focus on a few of my favorites here.

### The Bar Chart 
<blockquote class="imgur-embed-pub" lang="en" data-id="a/g6a3ixp" data-context="false" ><a href="//imgur.com/a/g6a3ixp"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

The Bar Chart is a clean and easy to understand chart when you are presenting your findings to a non-technical audience.  I highly recommend that if you can display your findings in a bar chart, you should do it.  There are a few options to code charts in Python, below is one of them using Matplotlib, but Seaborn is another great option.

```
import matplotlib.pyplot as plt
plt.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
```
* Information on the parameters can be found at this link: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.bar.html

### The Histogram
<blockquote class="imgur-embed-pub" lang="en" data-id="a/QUPeykT" data-context="false" ><a href="//imgur.com/a/QUPeykT"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

The Histogram Chart allows you to see if you have a 'normal distribution' through a bell curve shape, as illustrated in the chart above.  For Linear Regression, the model residuals should follow a normal distribution.  The normal distribution is also essential for Statistical Modeling.  

```
import matplotlib.pyplot as plt
plt.hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, *, data=None, **kwargs)
```
* Information on the parameters can be found at this link: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html

### The Confusion Matrix
<blockquote class="imgur-embed-pub" lang="en" data-id="a/WBcdfbp" data-context="false" ><a href="//imgur.com/a/WBcdfbp"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

The confusion matrix tells us the number of True Positives, True Negatives, False Positives and False Negatives when comparing the "True Value" and the "Predicted Value" from your classification model's fit.  It is a great visualization to see how good your model is fitting your data.   

```
# Import confusion_matrix
from sklearn.metrics import confusion_matrix

#includes option for normalization to return percentages for
#each class label in the visual rather than raw counts
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    # Check if normalize is set to True
    # If so, normalize the raw confusion matrix before visualizing
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, cmap=cmap)
    
    # Add title and axis labels 
    plt.title('Confusion Matrix') 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    
    # Add appropriate axis scales
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Text formatting
    fmt = '.2f' if normalize else 'd'
    # Add labels to each cell
    thresh = cm.max() / 2.
    # Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    # Add a legend
    plt.colorbar()
    plt.show() 

# Plot a normalized confusion matrix
class_names = set(y)
cnf_matrix = confusion_matrix(y_test, y_hat_test)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
```

* A great article to learn more about True Positives, False Positives, False Negatives and True Negatives can be found here: https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative


And there you have it, those are my 3 favorite types of visualization techniques!  As mentioned at the beginning of this blog, the visualization techniques you can use are really endless, so keep exploring.  I will be as well!



