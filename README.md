# Malware Classification
## Team - Gregory

`Project Description`: The ultimate objective of this project is the classification of documents among 9 different categories given the large uncompressed Microsoft Malware Classification Challenge dataset. The 9 Maleware categories are as follow:

<ul><li><i>Ramnit</i></li>
<li><i>Gatak</i></li>
<li><i>Lollipop</i></li>
<li><i>Kelihos_ver3</i></li>
<li><i>Simda</i></li>
<li><i>Tracur</i></li>
<li><i>Kelihos_verl</i></li>
<li><i>Vundo</i></li>
<li><i>Obfuscator.ACY</i></li></ul>

The files in the dataset contains only hexadecimal codes. The challenge is to design and develop a Classification model that can classify around 2721 test documents into the above mentioned 9 Malware categories. 
## Installation
* [Dask](https://docs.dask.org/en/latest/install.html)
* [Google Cloud Platform](https://cloud.google.com/) or alternatively you can use [Coiled](https://cloud.coiled.io/)

## Approach
* Create a Dask cluster with the required configuration as per your dataset volume
* Connect it through Web Interface / SSH and open Jupyter Notebook
* Parse the file and extract all the words
* Remove stopwords, punctuations
* Calculate TF-IDF values and create a dataframe
* Separate the dataset into training and testing datasets
* Take the training dataset and separate it by the target values
* Calculate statistical values such as mean, standard deviation for the dataset
* Summarize the data by class
* Calculate the Gaussian Probability Density Function
* Estimate the class probabilities
## Improvements
We estimated the probability of the documents by testing it against the trained Naive Bayes classifier and got the accuracy around 66%. By changing the classifier to Logistic Regression, we almost got 84% accuracy which is an improvement from our previous step.

## Contributions
<ul> <li><a href= "https://github.com/yogeshchaudhari"> Yogesh Chaudhari</a></li>
<li><a href = "https://github.com/shophine"> Shophine Sivaraja</a></li>
<li><a href ="https://github.com/zirakachakzai" > Zirak Khan </a></li></ul>

## License
This project is licensed under the MIT License - see the <a href="https://github.com/dsp-uga/gregory-p1/blob/main/LICENSE">LICENSE</a> file for the details.

## References
<ul> <li><a href = "https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/"> https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/ </a></li> 
<li><a href = "https://docs.dask.org/en/latest/"> https://docs.dask.org/en/latest/ </a></li> 
  <li><a href = "https://cloud.coiled.io/"> Coiled Cloud Platform </a></li> 
<li><a href = "https://towardsdatascience.com/naive-bayes-document-classification-in-python-e33ff50f937e"> https://towardsdatascience.com/naive-bayes-document-classification-in-python-e33ff50f937e </a></li>  
</ul> 
