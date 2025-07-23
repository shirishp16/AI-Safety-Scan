import transformers
from scipy import stats
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

print(dir(scipy))

# Download stopwords from NLTK
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

# Function to process the sentence
def process_sentence(sentence):
    # Convert to lowercase
    sentence = sentence.lower()
    # Split sentence into words
    words = sentence.split()
    # Remove stop words and punctuation
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return words  # You can return the list if you want an array of words instead of a string

# Load the CSV file into a DataFrame
df = pd.read_csv('/Users/aanyatummalapalli/Desktop/HACKATHON 24-25/hackathon/hackdata.csv')

# Process the target column (replace 'column_name' with the actual name of the column)
df['processed_column'] = df['PNT_ATRISKNOTES_TX'].apply(process_sentence)

# Save the updated DataFrame to a new CSV file
df[['processed_column']].to_csv('output_file.csv', index=False, header=False)

print("Processing complete. Output saved to output_file.csv")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
dataset = load_dataset('csv', data_files={'train': 'keywords.csv'})

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['keywords'], padding="max_length", truncation=True)

tokenized_dataset = dataset['train'].map(preprocess_function, batched=True)

# Specify the input features (input IDs and attention masks) and labels for PyTorch
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(
    output_dir='./results',              # Output directory
    evaluation_strategy="epoch",         # Evaluate after each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()