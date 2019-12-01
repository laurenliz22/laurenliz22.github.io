---
layout: post
title:      "ROC Curve / Multiclass Predictions / Random Forest Classifier"
date:       2019-12-01 23:27:55 +0000
permalink:  roc_curve_multiclass_predictions_random_forest_classifier
---


While working through my first modeling project as a Data Scientist, I found an excellent way to compare my models was using a ROC Curve!  However, I ran into a bit of a glitch because for the first time I had to create a ROC Curve using a dataset with multiclass predictions instead of binary predictions.  I also had to learn how to create a ROC Curve using a Random Forest Classifier for the first time.  Since it took me an entire afternoon googling to figure these things out, I thought I would blog about them to hopefully help someone in the future, that being you!

Let's begin!

After running my random forest classifier, I realized there is no `.decision function` to develop the y_score, which is what I thought I needed to produce my ROC Curve.  However, for a random forest classifier I learned you must instead use `.predict_proba` instead.

```
#construct baseline pipeline
pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state=123))])
```
```
# Fit the model
model = pipe_rf.fit(X_train, y_train)
```
```
#Calculate the y_score
y_score = model.predict_proba(X_test)
```

Using `.predict_proba` provides you with a y_score that will need to be binarized using label_binarize from sklearn.preprocessing.  In my case, I had 7 classes ranging from 1-7.

```
#Binarize the output
y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4, 5, 6,7])
n_classes = y_test_bin.shape[1]
```

Now you can finally create a ROC Curve (and calculate your AUC values) for your multiple classes using the code below!

```
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
  fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
  plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
  print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.show()
```

And that's it!  I hope this saved you an afternoon of googling!


