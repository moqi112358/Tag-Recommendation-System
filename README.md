# Tag-Recommendation-System

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

<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial \theta_k^{(j)}}=\sum_{i:r(i,j)=1}{\big((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\big)x_k^{(i)}}+\lambda \theta_k^{(j)}" />

作者：Deep Reader
链接：https://www.zhihu.com/question/26887527/answer/43166739
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
### Pipeline
1. Cleaned text data and padded sequences for titles and articles data respectively 
2. Built a pre-trained Word2vec word embedding with gensim as inputs for neural networks in order to perform NLP tasks
3. Implemented TextCNN model and Bidirectional LSTM model with titles and articles data as input using Pytorch
4. Implemented model ensemble for TextCNN and Bidirectional LSTM model with equal weights
5. Trained the model with approximately 1 million technology-related articles mapped to more than 30,000 different relevant tags and generated relevant tags from the given test set of articles
6. Achieved 0.6072 overall F1 score and won the fifth place in the competition
