# Using data visualization to analyze fraudulent reviewing behavior on the Google Play store to inform machine learning
MSc Project using VADER Sentiment for lexicon based sentiment analysis and Latent Dirichlet Allocation for topic modelling, on a dataset of Google Play review to discover fraudulent behaviour in reviews

## Abstract.
Before installing new applications from the Google Play Store consumers explore the application’s reviews and ratings in order to form an opinion and confirm their decision. Knowing that reviews can be influenced by outside parties, whether to increase or decrease an application’s standing in the market, we shape a database of application reviews and use qualitative data analysis to identify key features, using exploratory visualizations to discover the characteristics of fraudulent behaviour. We use Natural Language Processing techniques such as Sentiment Analysis and Latent Dirichlet Allocation to further distinguish and classify reviews in order to create a complete dataset for fraudulent review behavior analysis to the benefit of machine learning algorithms.

## Introduction

Reviews represent an individuals feelings towards a product, usually accompanied by a rating system, offering feedback that increases or decreases the standing of a product in their respective market. Consumers rely on reviews to determine if a product is worth purchasing, or using. This creates the possibility of influencing the review system, for the benefit of a product or the detriment of a competitor.

Fraud detection is a popular subject for machine learning. A large amount of review data is used in fraud detection algorithm development, the Play Store platform reaching a total of 3.6 million reviews in March 2018, to the benefit of their development.
In fraud detection, machine learning algorithms train using review features. These review features are extract from the data using existent techniques, such as Sentiment Analysis, or are already existent in the dataset, such as reviewer account details.  Following training, and testing, these algorithms are then deployed in the review environment to function.

To facilitate the development of these algorithms, our aim was to apply qualitative data analysis techniques to uncover key review features from the review dataset.
To do so we used Natural Language Processing(NLP) techniques to extract hidden information from our shaped data. We use a lexicon and rule-based sentiment analysis tool named VADER, using its own dictionary of sentiments.

Another NLP technique we used was the creation of statistical models using Latent Dirichlet Allocation(LDA), proposed by David Blei. LDA models contain topic distributions, where each review gravitates towards a specific topic. 

From our research we discover the relationship between topic, sentiment and rating features found in reviews and display it using visualizations.

Considering this relationship between features, we discuss the possible implementation of topic modeling, as of the machine learning process in fraud detection algorithms.

## Objectives

In the project proposal document we proposed the following research questions and objectives for our project:

RQ1: What visualizations of review data are effective in showing the different features of reviews in order to better classify them?  
RO1: To design visualizations of fraudulent reviews on the Google play market that are effective in uncovering the different features of a review that are omitted in existing machine learning algorithms.  

RQ2:  Can existing machine learning algorithms be adapted to account for the topics of a review in their classification of reviews? (features illustrated in the data visualization of review)
RO2: To apply topic modeling techniques on reviews datasets to learn how topics are formed and how they can be used to identify review intent in machine learning algorithms.

## Methods

### 1. Data Shaping

To clear and and shape our dataset, we’ve used OpenRefine[11], a free and open source tool for working with messy data. OpenRefine offers a simple Graphical User Interface with features for cell, row and column editing from the beginning, such as filtering, faceting, splitting, joining, transposing.

For advanced transforms, OpenRefine allowed us to use the Python programming language, used throughout the entire project, for the creation of custom transform expressions.

Changes to the dataset involved:
•	Removal of rows that contained corrupted data/characters(Ã°Å¸â€˜Å) in any of their columns, making data invalid
•	Standardizing the review date to dd/mm/yyyy  format followed by the separation of the new format into Day/Month/Year columns, having made possible the creation of monthly timelines
•	Standardization of rating scores in the dataset. All review scores were changed from percentages(100%, 80%, 60%, 40%, 20%) to the current Google Play star system (5, 4, 3, 2, 1)

We make use of previously collected Google Play metadata, containing review data from, and including, 2010 to 2016, containing a total of 212877 entries of textual data in the English language. 

### 2. VADER Sentiment Analysis

The functionality of VADER is available on the Python 3 package named vaderSentiment, used in our project. This version, compared to Python 3 has improved modularity, it’s credibility being confirmed after the inclusion of VADER into the Natural Language Toolkit(NLTK) Python package[15]. 

We modify VADER, specifically the polarity_scores() method of the SentimentIntensityAnalyzer class to return all used words and emoticons alongside the initial score return.
For information regarding VADER, visit: https://github.com/cjhutto/vaderSentiment

### 3. Natural Language Processing - Topic Modelling

Before using NLP techniques to analyze our data, we’ve split out review texts into tokens using punctuation and white space as split boundaries, removed punctuation,  and reduced each token to it’s root form, resulting in a list of tokens.

We use the Latent Dirichlet Allocation algorithm to describe each one of our review texts using a set of topics, and each topic can be described by a set of words.

Once we know the sets of topics each review is composed of, we use K-means clustering to find regions composed of reviews with similar topics.

### 4. Visualizations

To support their analysis and display results we use different visualizations of our dataset.

To develop our visualizations we’ve used the Matplotlib plotting library, for  the Python programming language. Both Matplotlib and Vega-Lite support the creation of detailed visualizations, as Matplotlib is capable of creating the same visualizations as Vega-Lite, without it’s interactive visualization features, which are not a necessity of our project.

Additionally, Matplotlib has  more detailed API documentation and a larger example gallery using Python, as Vega-Lite’s examples and documentation uses JavaScript Object Notation(JSON) formats in their definition.

Used visualization designs:
•	Word clouds
•	Bar charts
•	Line charts
•	Histograms
•	Donut charts
•	Dendrograms

### 5. Conclusion

The results of this study reveal the similarities between Play Store reviews and texts from social media platforms, such as Twitter.

We applied sentiment analysis, using VADER, and topic modeling, using the LDA algorithm, to find the different features of our review texts, which are added to our datasets.

VADER is used to analyze the word lists available in each review, and classify them into positive, negative and neutral. For each dataset, multiple LDA models are built in order to find the best model, giving us our review-topic probability matrix from which we extract the topics of our sets.

We’ve created multiple visualizations which helped us find relationships between their features. We discovered an increasing trend in our data where reviews increased in length across the years, suspicious behaviour in the top 10 most reviewed applications for the years 2015 and 2016, containing the largest count of 5 score ratings and Generic topic reviews  

For future work, we discussed the use of LDA in algorithms that detect the intent with which reviews are written, by creating a system that makes use of a topic lexicon for better topic identification. Additionally we describe what other features can be discovered from our datasets to aid in this process.

This project has taught and increased our knowledge in multiples areas of study. The following is a list of personal achievements:
•	Enhanced knowledge of Object programming by learning Python throughout the project
•	Improved usage of test routines in Object programming
•	Learned and applied new design principles for data visualizations
•	Learned different NLP techniques including sentiment analysis and topic modeling 
•	Acquired an understand of how large the area of NLP is in the area of computer science, from our research of techniques that could be applied on our datasets, such as bag-of-words, lexicons and machine learning 
•	Time management 
•	Task prioritisation

### 6. References

References
1.	A. Mukherjee, A. Kumar, B. Liu, J. Wang, M. Hsu, M. Castellanos, R. Ghosh, "Spotting opinion spammers using behavioral footprints", Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining
2.	Blei, D.M., Ng, A.Y. and Jordan, M.I., 2003. Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), pp.993-1022.[Accessed 5 August 2019]
3.	Carbunar, B. and Potharaju, R. (2019). A longitudinal study of the Google app market - IEEE Conference Publication. [online] Ieeexplore.ieee.org. Available at: https://ieeexplore.ieee.org/document/7403546 [Accessed 6 Jun. 2019].
4.	Dandannavar P., Jain P., (2016), “Sentiment Classification using Machine Learning Techniques”, International Journal of Science and Research(IJSR), Pages 819-821
5.	Fontanarava, J., Pasi, G. and Viviani, M. (2017). Feature Analysis for Fake Review Detection through Supervised Classification. 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA).
6.	Gordon, M. (2019). 80% of UK users access Twitter via their mobile. [online] Blog.twitter.com. Available at: https://blog.twitter.com/marketing/en_gb/a/en-gb/2014/80-of-uk-users-access-twitter-via-their-mobile.html [Accessed 1 Aug. 2019].
7.	Hutto, C. and Gilbert, E. (2019). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. [online] Semanticscholar.org. Available at: https://www.semanticscholar.org/paper/VADER%3A-A-Parsimonious-Rule-Based-Model-for-Analysis-Hutto-Gilbert/a6e4a2532510369b8f55c68f049ff11a892fefeb [Accessed 28 Jul. 2019].
8.	Ihara, I. and Aliza, R. (2017). Giving you more characters to express yourself. [online] Blog.twitter.com. Available at: https://blog.twitter.com/en_us/topics/product/2017/Giving-you-more-characters-to-express-yourself.html [Accessed 28 Jul. 2019].
9.	Kumar, K., Desai, J. and Majumdar, J. (2016). Opinion mining and sentiment analysis on online customer review. 2016 IEEE International Conference on Computational Intelligence and Computing Research (ICCIC).
10.	Lingras, P. and Triff, M. (2015). Fuzzy and Crisp Recursive Profiling of Online Reviewers and Businesses. IEEE Transactions on Fuzzy Systems, 23(4), pp.1242-1258.
11.	OpenRefine. (2019). OpenRefine/OpenRefine. [online] Available at: https://github.com/OpenRefine/OpenRefine [Accessed 29 Jun. 2019].
12.	Osborne, C. (2019). Google Play bids adieu to anonymous reviews. [online] CNET. Available at: https://www.cnet.com/news/google-play-bids-adieu-to-anonymous-reviews/ [Accessed 28 Jun. 2019].
13.	Reviewmeta.com. (2019). Frequently Asked Questions - ReviewMeta Blog. [online] Available at: https://reviewmeta.com/blog/faq/ [Accessed 25 Jun. 2019].
14.	Nelli, F. (2019). Circular Dendrograms - Meccanismo Complesso. [online] Meccanismo Complesso. Available at: https://www.meccanismocomplesso.org/en/circular-dendrograms-3/ [Accessed 26 Jul. 2019].
15.	NLTK. (2019). nltk.sentiment.vader — NLTK 3.4.5 documentation. [online] Available at: http://www.nltk.org/_modules/nltk/sentiment/vader.html [Accessed 19 Jul. 2019].
16.	Norvig, P. (2012). English Letter Frequency Counts: Mayzner Revisited or ETAOIN SRHLDCU. [online] Norvig.com. Available at: http://norvig.com/mayzner.html [Accessed 28 Jul. 2019].
17.	PowerReviews, Survey Confirms the Value of Reviews, Provides New Insights. Available at: https://www.powerreviews.com/blog/surveyconfirms-the-value-of-reviews/ [Accessed 27th of March 2019)]
18.	Scikit-learn.org. (2019). scikit-learn: machine learning in Python — scikit-learn 0.16.1 documentation. [online] Available at: https://scikit-learn.org/ [Accessed 10 Aug. 2019].
19.	Shivagangadhar K., Sagar H., Sathyan S., Vanipriya C.H.(2015) “Fraud Detection in Online Reviews using Machine Learning Techniques”, International Journal of Computational Engineering Research, Volume 5, Issue 5. Available at:http://docplayer.net/15165970-Fraud-detection-in-online-reviews-using-machinelearning-techniques.html
20.	Statistica(2019), “Number of available applications in the Google Play Store from December 2009 to December 2018”. Available from: https://www.statista.com/statistics/266210/number-of-available-applications-in-thegoogle-play-store/ [Accessed 29th of March 2019]
21.	The Python Graph Gallery (2019). Stacked barplot. [image] Available at: https://python-graph-gallery.com/12-stacked-barplot-with-matplotlib/ [Accessed 18 Jul. 2019].
22.	Umamaheswaran, V. (2019). Comprehending K-means and KNN Algorithms. [online] Medium. Available at: https://becominghuman.ai/comprehending-k-means-and-knn-algorithms-c791be90883d [Accessed 25 Jul. 2019].
23.	Vanaja, S. and Belwal, M. (2018). Aspect-Level Sentiment Analysis on E-Commerce Data. 2018 International Conference on Inventive Research in Computing Applications (ICIRCA).
24.	Wayasti, R., Surjandari, I. and Zulkamain (2018). Mining Customer Opinion for Topic Modeling Purpose: Case Study of Ride-Hailing Service Provider. 2018 6th International Conference on Information and Communication Technology (ICoICT).
25.	Xu, J. (2019). LDA Model. [image] Available at: https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05 [Accessed 25 Jul. 2019].
