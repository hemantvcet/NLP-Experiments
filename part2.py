import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('/home/hemant/Desktop/NLP_Exp')
print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("\n" + "="*60)
print("LOADING IMDB DATASET")
print("="*60)

try:
    df = pd.read_csv('IMDB Dataset.csv')
    print("✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nFirst few rows:")
    print(df.head(2))
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
except FileNotFoundError:
    print("❌ Error: 'IMDB Dataset.csv' not found in the current directory!")
    print("Please make sure the file is in /home/hemant/Desktop/NLP_Exp/")
    exit()

print("\n" + "="*60)
print("TEXT PREPROCESSING - TOKENIZATION ONLY")
print("="*60)

print("Taking a sample of 1000 reviews for demonstration...")
df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)
print(f"Sample dataset shape: {df_sample.shape}")

print("\n1. TOKENIZATION TECHNIQUES:")
print("-" * 30)

sample_text = df_sample['review'].iloc[0]
print(f"Sample review: {sample_text[:200]}...")
print(f"Sentiment: {df_sample['sentiment'].iloc[0]}")

def word_tokenization(text):
    tokens = word_tokenize(str(text).lower())
    return tokens

def whitespace_tokenization(text):
    return str(text).lower().split()

def regex_tokenization(text):
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    return tokens

def sentence_tokenization(text):
    sentences = sent_tokenize(str(text))
    return sentences

word_tokens = word_tokenization(sample_text)
whitespace_tokens = whitespace_tokenization(sample_text)
regex_tokens = regex_tokenization(sample_text)
sentence_tokens = sentence_tokenization(sample_text)

print(f"\nTokenization comparison on sample text:")
print(f"Word tokenization ({len(word_tokens)} tokens): {word_tokens[:10]}...")
print(f"Whitespace tokenization ({len(whitespace_tokens)} tokens): {whitespace_tokens[:10]}...")
print(f"Regex tokenization ({len(regex_tokens)} tokens): {regex_tokens[:10]}...")
print(f"Sentence tokenization ({len(sentence_tokens)} sentences): {sentence_tokens[:2]}...")

print("\n2. APPLYING TOKENIZATION TO IMDB DATASET:")
print("-" * 45)

def tokenize_text(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [token for token in tokens if token.isalpha() and len(token) > 1]
    
    return cleaned_tokens

print("Tokenizing IMDB reviews...")
df_sample['tokenized_review'] = df_sample['review'].apply(tokenize_text)
df_sample['token_count'] = df_sample['tokenized_review'].apply(len)
df_sample['tokenized_text'] = df_sample['tokenized_review'].apply(lambda x: ' '.join(x))

print("✅ Tokenization completed!")
print(f"\nSample of tokenized data:")
for i in range(3):
    original = df_sample['review'].iloc[i][:100]
    tokenized = df_sample['tokenized_text'].iloc[i][:100]
    print(f"\nReview {i+1}:")
    print(f"Original: {original}...")
    print(f"Tokenized: {tokenized}...")
    print(f"Token count: {df_sample['token_count'].iloc[i]}")

print("\n3. TOKENIZATION ANALYSIS:")
print("-" * 25)

print(f"Tokenization statistics:")
print(f"Average tokens per review: {df_sample['token_count'].mean():.2f}")
print(f"Maximum tokens: {df_sample['token_count'].max()}")
print(f"Minimum tokens: {df_sample['token_count'].min()}")
print(f"Standard deviation: {df_sample['token_count'].std():.2f}")

sentiment_stats = df_sample.groupby('sentiment')['token_count'].agg(['mean', 'std', 'count']).round(2)
print(f"\nToken count by sentiment:")
print(sentiment_stats)

all_tokens = [token for tokens in df_sample['tokenized_review'] for token in tokens]
word_freq = Counter(all_tokens)
most_common = word_freq.most_common(15)

print(f"\nMost frequent tokens:")
for i, (word, count) in enumerate(most_common, 1):
    print(f"{i:2d}. {word}: {count}")

print("\n4. VISUALIZATIONS:")
print("-" * 20)

plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist(df_sample['token_count'], bins=30, edgecolor='black', alpha=0.7, color='lightblue')
axes[0, 0].set_title('Distribution of Token Counts (After Tokenization)')
axes[0, 0].set_xlabel('Number of Tokens')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

sentiment_groups = [df_sample[df_sample['sentiment'] == sent]['token_count'] for sent in ['positive', 'negative']]
axes[0, 1].boxplot(sentiment_groups, labels=['Positive', 'Negative'])
axes[0, 1].set_title('Token Count Distribution by Sentiment')
axes[0, 1].set_ylabel('Number of Tokens')
axes[0, 1].grid(True, alpha=0.3)

sentiment_counts = df_sample['sentiment'].value_counts()
axes[1, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'])
axes[1, 0].set_title('Sentiment Distribution in Sample')

words, counts = zip(*most_common[:10])
axes[1, 1].barh(range(len(words)), counts, color='orange')
axes[1, 1].set_yticks(range(len(words)))
axes[1, 1].set_yticklabels(words)
axes[1, 1].set_title('Top 10 Most Common Tokens')
axes[1, 1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('tokenization_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Tokenization visualizations saved as 'tokenization_analysis.png'")

print("\n5. EXPORTING TOKENIZED DATA:")
print("-" * 30)

export_df = df_sample[['review', 'sentiment', 'tokenized_text', 'token_count']].copy()
export_df.to_csv('tokenized_imdb_sample.csv', index=False)
print("✅ Tokenized dataset exported as 'tokenized_imdb_sample.csv'")

token_freq_df = pd.DataFrame(most_common, columns=['token', 'frequency'])
token_freq_df.to_csv('token_frequency.csv', index=False)
print("✅ Token frequency data exported as 'token_frequency.csv'")

print("\n" + "="*60)
print("TOKENIZATION SUMMARY")
print("="*60)

print(f"""
📁 Dataset: IMDB Movie Reviews (Tokenization Only)
📊 Sample size: {len(df_sample):,} reviews
🔤 Total unique tokens: {len(word_freq):,}
📈 Average tokens per review: {df_sample['token_count'].mean():.1f}
💾 Files created:
   - tokenized_imdb_sample.csv
   - token_frequency.csv
   - tokenization_analysis.png

✅ Tokenization completed successfully!
   - Word-level tokenization applied using NLTK
   - HTML tags removed
   - Alphabetic tokens only (length > 1)
   - Sentence tokenization demonstrated
""")

print("="*60)