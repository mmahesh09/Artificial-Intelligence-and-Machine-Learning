# Machine Learning 
![image](https://github.com/mmahesh09/Artificial-Intelligence-and-Machine-Learning/blob/9f49da32cfc3b1d7e699dce890c60e691a4c1401/Machine%20Learning/images/Credit-Card%20fraud%20detection%20(4).png)

## Index
* Prerequisites of Machine Learning
* What is Machine Learning?
* Why Machine Learning
* Types of Machine Learning
* Terminologies of Machine Learning
* Applications of Machine Learning

  ## Prerequisites for Machine Learning

| **Prerequisite**       | **Description**                                                                                   | **Recommended YouTube Video**                                                                                   |
|------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **Linear Algebra**     | Understanding vectors, matrices, and operations involving them is crucial for algorithms and data representation in machine learning. | [Machine Learning Prerequisites](https://www.youtube.com/watch?v=AqGtnEM7S6U)                                    |
| **Calculus**           | Grasping derivatives and integrals aids in understanding optimization algorithms used in training models. | [Machine Learning Prerequisites](https://www.youtube.com/watch?v=AqGtnEM7S6U)                                    |
| **Probability & Statistics** | Knowledge of probability distributions, statistical tests, and data analysis is essential for making inferences and predictions. | [Machine Learning Prerequisites](https://www.youtube.com/watch?v=AqGtnEM7S6U)                                    |
| **Programming Skills** | Proficiency in programming languages, particularly Python, is necessary for implementing algorithms and handling data. | [Machine Learning Prerequisites](https://www.youtube.com/watch?v=AqGtnEM7S6U)                                    |
| **Data Structures & Algorithms** | Understanding data organization and algorithmic problem-solving enhances efficiency in model implementation. | [Machine Learning Prerequisites](https://www.youtube.com/watch?v=AqGtnEM7S6U)                                    |
| **Domain Knowledge**   | Familiarity with the specific field of application helps in selecting appropriate models and interpreting results effectively. | [Machine Learning Prerequisites](https://www.youtube.com/watch?v=AqGtnEM7S6U)                                    |


# Machine Learning
Machine learning (ML) is a subset of artificial intelligence (AI) that focuses on developing systems capable of learning and improving from experience without being explicitly programmed. ML algorithms use data to identify patterns, make predictions, or take decisions, adapting their performance as they are exposed to more data.

# Why Machine Learning?

Machine learning is crucial because it enables systems to automatically learn and improve from data, making them more adaptable, efficient, and intelligent over time. It drives innovation by solving complex problems across industries like healthcare (disease detection), finance (fraud detection), and technology (personalized recommendations). Unlike traditional programming, ML excels in tasks where explicit rules are hard to define, such as image recognition, language translation, and predictive analytics. With the explosion of big data, ML transforms vast information into actionable insights, enhancing decision-making and automating repetitive tasks. Its ability to uncover hidden patterns makes it a cornerstone of modern AI advancements.

# Types of Machine Learning

* Supervised Learning
* Unsupervised Learning
* Reinforcement Learning
---
  ### 1. **Supervised Learning**  
Supervised learning involves training a model on labeled data, where inputs have corresponding outputs. The algorithm learns to map inputs to the correct outputs and generalize to unseen data. Tasks include classification (categorizing data) and regression (predicting continuous values).  
**Examples**:  
- **Classification**: Email spam detection (spam or not spam).  
- **Regression**: Predicting house prices based on features like size and location.  
Common algorithms include Decision Trees, Support Vector Machines, and Neural Networks.

---

### 2. **Unsupervised Learning**  
Unsupervised learning works on unlabeled data to identify patterns, groupings, or structures. It’s used when no explicit output is provided, often for clustering or dimensionality reduction.  
**Examples**:  
- **Clustering**: Customer segmentation in marketing (grouping customers by behavior).  
- **Dimensionality Reduction**: Reducing features in large datasets, like PCA for image compression.  
Key algorithms include K-Means, Hierarchical Clustering, and Autoencoders.

---

### 3. **Reinforcement Learning**  
Reinforcement learning focuses on training an agent to make decisions in an environment by maximizing cumulative rewards through trial and error. The agent learns from feedback (rewards or penalties).  
**Examples**:  
- Training robots to walk.  
- Optimizing strategies in games like chess or Go.  
Popular algorithms include Q-Learning, Deep Q-Networks (DQN), and Policy Gradient Methods.  

## Machine Learning Terminologies

| **Term**               | **Definition**                                                                                   | **Example**                                                                                       |
|-------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Dataset**            | A collection of data used for training, testing, and validating machine learning models.        | A CSV file containing features like age, income, and labels like "yes/no" for loan approval.      |
| **Feature**            | An individual measurable property or characteristic of the data.                               | Age, income, and education level in a dataset predicting loan approval.                           |
| **Label**              | The target variable that the model aims to predict or classify.                                | "Spam" or "Not Spam" in an email classification problem.                                          |
| **Model**              | A mathematical representation of the relationships in the data learned by an algorithm.        | A neural network trained to identify handwritten digits.                                          |
| **Training**           | The process of teaching a model using labeled data to find patterns and relationships.         | Feeding images of cats and dogs with labels to train a classifier.                                |
| **Testing**            | Evaluating the model's performance on unseen data to check its generalization.                 | Using a separate dataset to assess a trained fraud detection model.                               |
| **Overfitting**        | When a model learns the training data too well, including noise, and fails to generalize.       | A decision tree that perfectly fits training data but performs poorly on new data.                |
| **Underfitting**       | When a model is too simple to capture the underlying patterns in the data.                     | A linear regression model failing to fit a quadratic relationship in the data.                   |
| **Hyperparameters**    | External parameters set before training to control the learning process.                       | Learning rate, number of layers in a neural network, or K in K-Means clustering.                  |
| **Epoch**              | One complete pass through the entire training dataset.                                         | Training a neural network for 10 epochs.                                                         |
| **Learning Rate**      | A hyperparameter controlling how much the model adjusts weights during training.               | A lower learning rate may result in slower but more stable convergence.                           |
| **Loss Function**      | A function measuring the difference between predicted and actual outputs.                      | Mean Squared Error (MSE) for regression tasks or Cross-Entropy for classification.                |
| **Regularization**     | Techniques to prevent overfitting by adding constraints or penalties to the model.             | L1 or L2 regularization to limit model complexity.                                                |
| **Gradient Descent**   | An optimization algorithm to minimize the loss function by updating model weights.             | Stochastic Gradient Descent (SGD) for deep learning models.                                       |
| **Supervised Learning**| Learning with labeled data to predict outcomes.                                                | Predicting housing prices using historical data.                                                  |
| **Unsupervised Learning**| Learning patterns or structures in unlabeled data.                                           | Clustering customers into groups based on purchasing behavior.                                    |
| **Reinforcement Learning**| Learning to make decisions through rewards and penalties.                                   | Training a robot to navigate a maze by maximizing its rewards.                                    |
| **Confusion Matrix**   | A table summarizing a classification model's predictions against true labels.                  | True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).             |
| **Accuracy**           | The ratio of correct predictions to the total predictions made by the model.                   | (TP + TN) / Total Predictions.                                                                    |
| **Precision**          | The ratio of true positives to all predicted positives.                                        | TP / (TP + FP).                                                                                   |
| **Recall**             | The ratio of true positives to all actual positives.                                           | TP / (TP + FN).                                                                                   |
| **F1 Score**           | The harmonic mean of precision and recall.                                                    | 2 * (Precision * Recall) / (Precision + Recall).                                                  |

