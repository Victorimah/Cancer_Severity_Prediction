# Cancer_Severity_Prediction
Predict whether a mammogram mass is benign or malignant
I'll be using the "mammographic masses" public dataset from the UCI repository (source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)

This data contains 961 instances of masses detected in mammograms, and contains the following attributes:

BI-RADS assessment: 1 to 5 (ordinal)
1. Age: patient's age in years (integer)
2. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
3. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
4. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
5. Severity: benign=0 or malignant=1 (binomial)

BI-RADS is an assesment of how confident the severity classification is; it is not a "predictive" attribute and so we will discard it. The age, shape, margin, and density attributes are the features that we will build our model with, and "severity" is the classification we will attempt to predict based on those attributes.

Although "shape" and "margin" are nominal data types, which sklearn typically doesn't deal with well, they are close enough to ordinal that I shouldn't just discard them. The "shape" for example is ordered increasingly from round to irregular.

A lot of unnecessary anguish and surgery arises from false positives arising from mammogram results. If I can build a better way to interpret them through supervised machine learning, it could improve a lot of lives.

I will be applying several different supervised machine learning techniques to this data set, and see which one yields the highest accuracy as measured with K-Fold cross validation (K=10). Apply:

1. Decision tree
2. Random forest
3. KNN
4. Naive Bayes
5. SVM
6. Logistic Regression

And, as a bonus challenge, a neural network using Keras. The data needs to be cleaned; many rows contain missing data, and there may be erroneous data identifiable as outliers as well.

Many techniques also have "hyperparameters" that need to be tuned. Once I identify a promising approach, I will see if I can make it even better by tuning its hyperparameters. The algorithm with the greatest numerical value will be transformed into a pickle file for future prediction purposes.
