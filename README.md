**Safety Observation Comment Classifier**

This repository contains a program designed to classify safety observation comments into hazard categories based on their severity. The program utilizes natural language processing (NLP) capabilities provided by large language models (LLMs) to identify and categorize comments efficiently.

**Features**

Hazard Categorization:
The program classifies comments into predefined hazard categories, such as "High Concern" and "Low Concern," to prioritize attention based on severity.

Sentiment Analysis Using Pre-trained Models:
The program employs the BERT-based facebook/bart-large-mnli model for zero-shot classification. This approach eliminates the need for additional model training while delivering accurate results.

Confidence Scoring:
Each classification includes a confidence score, helping users evaluate the reliability of the assigned hazard category.

CSV Input and Output:

Input: Reads safety observation comments from a CSV file.
Output: Outputs the classified comments, along with their hazard category and confidence score, into a new CSV file.
