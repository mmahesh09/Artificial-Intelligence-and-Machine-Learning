# Machine Learning Algorithms

Machine learning (ML) algorithms are mathematical and computational models that enable computers to learn from and make predictions or decisions based on data without being explicitly programmed for specific tasks. These algorithms analyze data, identify patterns, and build models that generalize to new, unseen data

# Type of ML Algorithms
1. Supervised Algorithm
2. Unsupervised Algorithm
3. Reinforcement Algorithm

# Supervised Machine Learning Algorithms ♻️

This table summarizes popular supervised ML algorithms, their functionality, and real-world examples.

| **Algorithm**            | **Description**                                                                 | **Type**            | **Real-World Example**                                                   |
|---------------------------|---------------------------------------------------------------------------------|---------------------|---------------------------------------------------------------------------|
|[ **Linear Regression**](https://github.com/mmahesh09/Artificial-Intelligence-and-Machine-Learning/blob/eee29dda0ffb309d834be81db6a732442ec22ba0/Machine%20Learning/Linear%20Regression.md)     | Predicts a continuous target variable based on input features.                  | Regression          | Predicting house prices based on size, location, and amenities.          |
| **Logistic Regression**   | Predicts binary or multi-class outcomes using a logistic function.              | Classification      | Email spam detection (spam vs. not spam).                                |
| **Decision Trees**        | Splits data into branches based on feature values for predictions.              | Regression, Classification | Loan approval based on credit score, income, and age.               |
| **Random Forest**         | Ensemble of decision trees that improves accuracy and reduces overfitting.      | Regression, Classification | Fraud detection in online transactions.                            |
| **Support Vector Machine (SVM)** | Finds the optimal hyperplane to separate data into classes.                      | Classification      | Classifying handwritten digits in the MNIST dataset.                     |
| **K-Nearest Neighbors (KNN)** | Predicts labels based on the majority vote of k-nearest data points.                 | Regression, Classification | Recommending similar products in e-commerce.                           |
| **Naive Bayes**           | Based on Bayes' theorem; assumes feature independence.                          | Classification      | Sentiment analysis of product reviews (positive, neutral, negative).     |
| **Gradient Boosting (e.g., XGBoost)** | Sequentially improves predictions by minimizing errors in previous models.            | Regression, Classification | Predicting customer churn in a subscription business.                   |
| **AdaBoost**              | Combines weak learners into a strong ensemble for better predictions.           | Regression, Classification | Diagnosing diseases based on medical test results.                       |
| **Neural Networks**       | Mimics the human brain; used for complex data patterns.                         | Regression, Classification | Predicting stock prices based on historical data.                        |
| **Linear Discriminant Analysis (LDA)** | Reduces dimensionality while maintaining class separability.                           | Classification      | Face recognition in security systems.                                    |
| **Ridge Regression**      | Linear regression with L2 regularization to prevent overfitting.                | Regression          | Predicting sales trends while avoiding overfitting to noisy data.        |
| **Lasso Regression**      | Linear regression with L1 regularization for feature selection.                 | Regression          | Predicting housing prices while identifying the most important features. |
| **ElasticNet Regression** | Combines L1 and L2 regularization for better flexibility.                       | Regression          | Predicting patient recovery rates from hospital data.                    |
| **CatBoost**              | Gradient boosting algorithm optimized for categorical features.                 | Regression, Classification | Predicting customer lifetime value in e-commerce.                        |
| **LightGBM**              | Gradient boosting algorithm optimized for speed and efficiency.                 | Regression, Classification | Energy consumption forecasting for smart grids.                          |

---

# Unsupervised Machine Learning Algorithms⚙️

| **Algorithm**                  | **Description**                                                                                  | **Use Cases**                                   |
|---------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------------|
| **K-Means Clustering**         | Partitions data into `k` clusters based on similarity.                                           | Customer segmentation, image compression.      |
| **Hierarchical Clustering**    | Creates a tree-like structure of clusters (dendrogram).                                          | Gene analysis, market segmentation.            |
| **DBSCAN (Density-Based)**     | Groups data points close to each other, identifying noise as outliers.                          | Anomaly detection, spatial data analysis.      |
| **Gaussian Mixture Models (GMM)** | Assumes data is generated from a mixture of Gaussian distributions.                              | Image segmentation, density estimation.        |
| **Mean-Shift Clustering**      | Identifies clusters by locating peaks in a data density function.                               | Image processing, object tracking.             |
| **Principal Component Analysis (PCA)** | Reduces data dimensions while preserving variance.                                           | Data visualization, noise reduction.           |
| **t-SNE (t-Distributed Stochastic Neighbor Embedding)** | Reduces dimensions for visualization by preserving local similarities.                    | Data exploration, high-dimensional datasets.    |
| **UMAP (Uniform Manifold Approximation and Projection)** | Reduces dimensions while preserving global and local data structures.                     | Data visualization, cluster analysis.          |
| **Autoencoders**               | Neural networks that learn compressed representations of data.                                  | Anomaly detection, feature extraction.         |
| **Independent Component Analysis (ICA)** | Decomposes a multivariate signal into independent components.                              | Signal separation (e.g., blind source separation). |
| **Self-Organizing Maps (SOMs)** | Neural networks that map high-dimensional data to a 2D grid.                                    | Feature mapping, pattern recognition.          |
| **Isolation Forest**           | Identifies anomalies by isolating observations in a forest of random trees.                    | Fraud detection, anomaly detection.            |
| **Affinity Propagation**       | Clusters data by identifying exemplars (representative points).                                | Document clustering, image segmentation.        |
| **Spectral Clustering**        | Uses eigenvalues of similarity matrices to reduce dimensions for clustering.                   | Graph partitioning, image segmentation.        |
| **Deep Belief Networks (DBNs)** | Stacked neural networks used for feature learning and unsupervised tasks.                      | Pre-training for supervised tasks.             |
| **Factor Analysis**            | Explains data variability using a few latent variables (factors).                              | Psychometrics, social science research.        |

---

### Notes:
- **Clustering Algorithms:** Group data points into meaningful clusters (e.g., K-Means, DBSCAN).
- **Dimensionality Reduction:** Simplify data while preserving essential structures (e.g., PCA, t-SNE).
- **Neural Network-Based:** Use deep learning methods for feature extraction and pattern recognition (e.g., Autoencoders, DBNs).

---
## Reinforcement Learning Algorithms 🌐

| **Algorithm**               | **Type**             | **Description**                                                                 | **Key Use Cases**                            |
|------------------------------|----------------------|---------------------------------------------------------------------------------|---------------------------------------------|
| **Q-Learning**               | Model-Free, Value-Based | Learns an action-value function to find the optimal policy using Q-values.      | Game AI, pathfinding, robotics              |
| **SARSA**                    | Model-Free, Value-Based | Similar to Q-Learning but considers the current policy (on-policy).            | Safer policies in dynamic environments       |
| **Deep Q-Networks (DQN)**    | Model-Free, Deep RL  | Combines Q-Learning with deep neural networks to handle high-dimensional states.| Atari games, autonomous systems             |
| **Double DQN**               | Model-Free, Deep RL  | Addresses overestimation bias in Q-values by using two networks.                | Advanced game AI, robotics                  |
| **Dueling DQN**              | Model-Free, Deep RL  | Separates state-value estimation and action advantages for better learning.     | Complex environments with sparse rewards    |
| **Policy Gradient (PG)**     | Model-Free, Policy-Based | Directly optimizes the policy to maximize expected rewards.                     | Continuous control, robot locomotion        |
| **REINFORCE**                | Model-Free, Policy-Based | A Monte Carlo policy gradient method.                                          | Episodic tasks, simple control problems     |
| **Actor-Critic**             | Model-Free, Hybrid   | Combines policy gradients (Actor) with value functions (Critic).                | Stable training in continuous environments  |
| **A3C (Asynchronous Advantage Actor-Critic)** | Model-Free, Hybrid | Trains multiple agents asynchronously to improve stability and efficiency.      | Multi-threaded RL tasks, resource optimization |
| **PPO (Proximal Policy Optimization)** | Model-Free, Policy-Based | Balances exploration and exploitation with clipped objective functions.         | Robotics, video games                       |
| **TRPO (Trust Region Policy Optimization)** | Model-Free, Policy-Based | Ensures stable updates by constraining policy changes.                          | Robotics, continuous action spaces          |
| **DDPG (Deep Deterministic Policy Gradient)** | Model-Free, Policy-Based | Extends DPG with deep learning for continuous action spaces.                    | Autonomous driving, robot arm control       |
| **TD3 (Twin Delayed DDPG)**  | Model-Free, Policy-Based | Improves DDPG by addressing overestimation bias and instability.                | Continuous control, robotics                |
| **SAC (Soft Actor-Critic)**  | Model-Free, Policy-Based | Maximizes entropy to encourage exploration while optimizing rewards.            | High-dimensional continuous control         |
| **Monte Carlo Tree Search (MCTS)** | Model-Based        | Uses a tree search to optimize actions based on simulated outcomes.             | Board games (e.g., Go, Chess)               |
| **Dyna-Q**                   | Model-Based         | Integrates planning with learning by simulating experience in a model.          | Planning and simulation tasks               |
| **AlphaGo/AlphaZero**        | Hybrid              | Combines MCTS and deep learning for strategic decision-making.                  | Board games, strategy games                 |
| **Rainbow DQN**              | Model-Free, Deep RL  | Combines multiple DQN improvements (e.g., Double DQN, Dueling DQN).             | Complex RL tasks                            |
| **Hierarchical RL (HRL)**    | Hybrid              | Decomposes tasks into sub-tasks for structured learning.                        | Complex, long-horizon tasks                 |

---

### Key Notes:
- **Model-Free Algorithms:** Do not rely on an explicit model of the environment.
- **Model-Based Algorithms:** Use a model of the environment for planning and decision-making.
- **Policy-Based Algorithms:** Optimize policies directly.
- **Value-Based Algorithms:** Learn value functions to derive policies.
- **Hybrid Algorithms:** Combine multiple approaches for better performance.




