# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model used is a Random Forest Classifier that was drained on a U.S. Census dataset. The goal was to predit if an individual's income is greater or less than $50,000 per year based on demographics. The model was developed using the Scikit-learn library.

## Intended Use
The model is intended to be used by researchers or organizations that wish to better understand income distribution patterns. For instance, it could be used by a non-profit to predict which individuals may have higher income and be more likely to donate based on demographic information. It is not intended for high-stakes decision making, like legal or hiring decisions. 

## Training Data
The model was trained on a U.S. census dataset that included information on race, sex, native country, marital status, education, occupation, workclass, and age. The dataset has features representing these attributes. The target variable is "salary" which indicates whether a person's income is greater than or less than $50,000 per year. The data was split 80/20, with 80% used for training and 20% used for testing.

## Evaluation Data
The evaluation data was split from the same dataset as the training. It included 20% of the dataset which was withheld during the training. It has the same features and target variables as the training data. 

## Metrics
The model's performance was evaluated using the following metrics:

Precision: 0.7391. This measures the accuracy of the positive predictions made by the model. Of all the individuals the model predicted as earning more than $50K, 73.91% actually earned more than $50K.
Recall: 0.6384. This measures the model's ability to identify all the individuals who actually earn more than $50K. The model correctly identified 63.84% of the positive cases.
F1 Score: 0.6851. The F1 score is the harmonic mean of precision and recall, providing a balance between the two metrics. The model achieved an F1 score of 0.6851, indicating a reasonable balance between precision and recall.

## Ethical Considerations
The model could reinforce or perpetuate biases that already exist in the census data. For instance, the census data may reflect systemic, historical inequalities related to race or gender. The model may in turn produce biased predictions for underrepresented or marginalized groups. 

## Caveats and Recommendations
The census data that the model was trained on may not represent the full diversity of the actual population. It should not be generalized beyond the datatset's scope, especially to populations that have different demographics than the U.S. The model could be improved by addressing biases, perhaps by testing it on different demographic groups or training it based on different datasets. It could also benefit from a more complex modeling approach that improves its accuracy. 
