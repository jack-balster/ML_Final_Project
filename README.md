# ML_Final_Project

This is a final project for an Intro to Machine Learning course. In this project, we attempt to perform sentiment analysis on IMDb moview reviews. 

To accomplish this natural language processing task, we utilized Hugging Face's distilBERT tokenizer and model to tokenize and encode the reviews to a anumerical format. Additionally, to ensure that the length of the reviews do not impact the lengths of its numerical output, we attatched an average pooling layer to the model to standardize the vector lengths.

The stucture of the code is simple as each step/section of the process is separated by a cells due to the orignal code being made in Google Colab. In total, there are 12 cells of code which runs as follows:

1. Asks the user for permission to connect to the user's Google Drive.
2. Loads the IMSb dataset which the user is expected to have already downloaded and uploaded to their Google Drive.
3. Downloads the necessary transformer packages to perform the preprocessing tasks needed for the data.
4. Randomly selects 10000 positve samples and 10000 negative samples from the original dataset, concatinates these two subsets together to one dataset, then shuffles the rows to ensure ordering does not affect training. The reason to lower the size of the dataset is due to our restricted computational capacity. Google Colab provides up to 12 GB of RAM, which can quickly disapate when working in NLP.
5. 
![image](https://github.com/Alyssa1918/ML_Final_Project/assets/123338206/3411f944-6d13-4434-85cd-c84786f434ea)
