# HackerEarth Deep Learning Challenge: Tag-Recommendation-System （Rank 5）

https://www.hackerearth.com/challenge/competitive/deep-learning-challenge-4/leaderboard/

### Problem Statement
HackerEarth wants to improve its customer experience by suggesting tags for any idea submitted by a participant for a given hackathon. Currently, tags can only be manually added by a participant. HackerEarth wants to automate this process with the help of machine learning. To help the machine learning community grow and enhance its skills by working on real-world problems, HackerEarth challenges all the machine learning developers to build a model that can predict or generate tags relevant to the idea/ article submitted by a participant.

You are provided with approximately 1 million technology-related articles mapped to relevant tags. You need to build a model that can generate relevant tags from the given set of articles.

### Data Description
The dataset consists of ‘train.csv ’, ‘test.csv’ and ‘sample_submission.csv’. Description of the columns in the dataset is given below:

id: Unique id for each article

title: Title of the article

article: Description of the article (raw format)

tags: Tags associated with the respective article. If multiple tags are associated with an article then they are seperated by '|'.

### Submission

The submission file submitted by the candidate for evaluation has to be in the given format. The submission file is in .csv format. Check sample_submission for details. Remember, incase of multiple tags for a given article, they are seperated by '|'. 

### Evaluation Metric

The predicted tags will be evaluated on the F1 score metrics. For each article, the detail formula can be found in the challenge website.

### Pipeline
1. Cleaned text data and padded sequences for titles and articles data respectively 
2. Built a pre-trained Word2vec word embedding with gensim package as inputs for neural networks in order to perform NLP tasks
3. Implemented TextCNN model and Bidirectional LSTM model with titles and articles data as input using Pytorch
4. Implemented model ensemble for TextCNN and Bidirectional LSTM model with equal weights
5. Trained the model with approximately 1 million technology-related articles mapped to more than 30,000 different relevant tags and generated relevant tags from the given test set of articles
6. Achieved 0.6072 overall F1 score and won the 5th place in the competition

### Models

![image](https://github.com/moqi112358/Tag-Recommendation-System/blob/master/image/model.png)
