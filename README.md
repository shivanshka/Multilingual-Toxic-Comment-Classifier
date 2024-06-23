---
title: Toxic Comment Classifier
emoji: 🌍
colorFrom: blue
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# MultiLingual_Toxic_Comment_Classification

## Problem Statement and Description

It only takes one toxic comment to sour an online discussion. Identifing the toxicity in online conversations, where toxicity is defined as anything `rude, disrespectful or otherwise likely to make someone leave a discussion`, in an automated way using machine learning, at very early stage, is one way to protect voices in online conversations.

The MultiLingual Toxic Comment Classification project is a powerful tool for content platforms operating in various domains, including social media, news websites, forums, and on the internet. By leveraging the latest Natural Language Processing techniques, it automatically detects and filters toxic comments in multiple languages, addressing the pressing issues of cyberbullying, hate speech and harassment.

## Dataset Description

The dataset has been taken from `Kaggle` competition organized by the [`Jigsaw/Conversation AI`](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/overview) team. The dataset contains several files:

- jigsaw-toxic-comment-train.csv - data from our [first competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The dataset is made up of English comments from Wikipedia’s talk page edits.
- jigsaw-unintended-bias-train.csv - data from our [second competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). This is an expanded version of the Civil Comments dataset with a range of additional labels.
- sample_submission.csv - a sample submission file in the correct format.
- test.csv - comments from Wikipedia talk pages in different non-English languages (Spanish, Portuguese, Italian, Turkish, French, Russian).
- validation.csv - comments from Wikipedia talk pages in different non-English languages.

Columns

- id - identifier within each file.
- comment_text - the text of the comment to be classified.
- lang - the language of the comment.
- toxic - whether or not the comment is classified as toxic. (Does not exist in test.csv.)

Here, we're predicting the probability that a comment is `toxic`. A toxic comment would receive a `1.0`. A benign, non-toxic comment would receive a`0.0`. In the test set, all comments are classified as either a `1.0` or `0.0`.

## Repository Structure

The project has the following structure:

- input: This repository would contain the all the csv files mentioned above which can be downloaded from [here](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data) and put them in it.
- demos/multilingual_toxic_comment_files: This repository would contain all the necessary files to build our `Gradio` application to deploy on `HuggingFace Spaces`. All the files required to deploy on `HuggingFace Spaces` are [here](https://huggingface.co/spaces/shivansh-ka/Toxic-Comment-Classifier/tree/main). Note: Ignore git related files and `Multlingual_toxic_comment_classifier/` repository contains our model binary files.
- notebooks: This repository contains jupyter notebooks for all the experiments we ran for model building. Note: To train the models we've used `Kaggle TPUs` and `Kaggle Kernels`. The final/chosen model is in `Multilingual_Toxic_Comment_Classification_Final.ipynb` notebook and this notebook maynot render many times so may have to download locally this repository to view it.
- output/working - This repository contains the saved files while training the model on Kaggle kernel.

## Deployment

The application is deployed on `HuggingFace Spaces` using `Gradio` at [Here](https://huggingface.co/spaces/shivansh-ka/Toxic-Comment-Classifier).
