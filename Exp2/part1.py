import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('/home/hemant/Desktop/NLP_Exp')
print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
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
print("TEXT PREPROCESSING - STOP WORD REMOVAL")
print("="*60)

print("Taking a sample of 1000 reviews for demonstration...")
df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)
print(f"Sample dataset shape: {df_sample.shape}")

print("\n1. STOP WORD ANALYSIS:")
print("-" * 25)

stop_words = set(stopwords.words('english'))
print(f"Number of English stop words: {len(stop_words)}")
print(f"Sample stop words: {list(stop_words)[:20]}")

sample_text = df_sample['review'].iloc[0]
print(f"\nSample review: {sample_text[:200]}...")
print(f"Sentiment: {df_sample['sentiment'].iloc[0]}")

def tokenize_text(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [token for token in tokens if token.isalpha() and len(token) > 1]
    return cleaned_tokens

tokens_with_stopwords = tokenize_text(sample_text)
print(f"\nTokens with stop words ({len(tokens_with_stopwords)}): {tokens_with_stopwords[:15]}...")

def remove_stopwords(tokens):
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def identify_stopwords_in_text(tokens):
    stopwords_found = [word for word in tokens if word in stop_words]
    return stopwords_found

tokens_without_stopwords = remove_stopwords(tokens_with_stopwords)
stopwords_in_sample = identify_stopwords_in_text(tokens_with_stopwords)

print(f"Tokens without stop words ({len(tokens_without_stopwords)}): {tokens_without_stopwords[:15]}...")
print(f"Stop words found in sample ({len(stopwords_in_sample)}): {stopwords_in_sample[:15]}...")

reduction_percentage = ((len(tokens_with_stopwords) - len(tokens_without_stopwords)) / len(tokens_with_stopwords)) * 100
print(f"Reduction in sample text: {reduction_percentage:.1f}%")

print("\n2. STOP WORD REMOVAL METHODS:")
print("-" * 35)

def method1_nltk_stopwords(tokens):
    stop_words_nltk = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words_nltk]

def method2_custom_stopwords(tokens):
    custom_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    return [word for word in tokens if word not in custom_stopwords]

def method3_frequency_based(tokens, min_freq=2):
    word_freq = Counter(tokens)
    return [word for word in tokens if word_freq[word] >= min_freq]

sample_tokens = tokenize_text(sample_text)
method1_result = method1_nltk_stopwords(sample_tokens)
method2_result = method2_custom_stopwords(sample_tokens)
method3_result = method3_frequency_based(sample_tokens)

print(f"Original tokens: {len(sample_tokens)}")
print(f"Method 1 (NLTK stopwords): {len(method1_result)} tokens")
print(f"Method 2 (Custom stopwords): {len(method2_result)} tokens")
print(f"Method 3 (Frequency-based): {len(method3_result)} tokens")

print("\n3. APPLYING STOP WORD REMOVAL TO DATASET:")
print("-" * 45)

def preprocess_remove_stopwords(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
    filtered_tokens = remove_stopwords(cleaned_tokens)
    
    return filtered_tokens

print("Applying stop word removal to IMDB reviews...")
df_sample['tokens_with_stopwords'] = df_sample['review'].apply(tokenize_text)
df_sample['tokens_without_stopwords'] = df_sample['review'].apply(preprocess_remove_stopwords)
df_sample['original_token_count'] = df_sample['tokens_with_stopwords'].apply(len)
df_sample['filtered_token_count'] = df_sample['tokens_without_stopwords'].apply(len)
df_sample['stopwords_removed'] = df_sample['original_token_count'] - df_sample['filtered_token_count']
df_sample['reduction_percentage'] = (df_sample['stopwords_removed'] / df_sample['original_token_count'] * 100).round(1)
df_sample['processed_text'] = df_sample['tokens_without_stopwords'].apply(lambda x: ' '.join(x))

print("✅ Stop word removal completed!")
print(f"\nSample of processed data:")
for i in range(3):
    original = df_sample['review'].iloc[i][:80]
    processed = df_sample['processed_text'].iloc[i][:80]
    print(f"\nReview {i+1}:")
    print(f"Original: {original}...")
    print(f"After removal: {processed}...")
    print(f"Tokens before: {df_sample['original_token_count'].iloc[i]}, after: {df_sample['filtered_token_count'].iloc[i]}, reduction: {df_sample['reduction_percentage'].iloc[i]}%")

print("\n4. STOP WORD REMOVAL ANALYSIS:")
print("-" * 35)

print(f"Stop word removal statistics:")
print(f"Average original tokens per review: {df_sample['original_token_count'].mean():.2f}")
print(f"Average tokens after removal: {df_sample['filtered_token_count'].mean():.2f}")
print(f"Average stop words removed per review: {df_sample['stopwords_removed'].mean():.2f}")
print(f"Average reduction percentage: {df_sample['reduction_percentage'].mean():.1f}%")

sentiment_analysis = df_sample.groupby('sentiment').agg({
    'original_token_count': 'mean',
    'filtered_token_count': 'mean',
    'stopwords_removed': 'mean',
    'reduction_percentage': 'mean'
}).round(2)

print(f"\nStop word removal by sentiment:")
print(sentiment_analysis)

all_filtered_tokens = [token for tokens in df_sample['tokens_without_stopwords'] for token in tokens]
filtered_word_freq = Counter(all_filtered_tokens)
most_common_filtered = filtered_word_freq.most_common(15)

print(f"\nMost frequent words after stop word removal:")
for i, (word, count) in enumerate(most_common_filtered, 1):
    print(f"{i:2d}. {word}: {count}")

all_original_tokens = [token for tokens in df_sample['tokens_with_stopwords'] for token in tokens]
original_word_freq = Counter(all_original_tokens)
most_common_original = original_word_freq.most_common(15)

stopwords_in_dataset = [word for word, count in most_common_original if word in stop_words]
print(f"\nMost common stop words in dataset: {stopwords_in_dataset[:10]}")

print("\n5. VISUALIZATIONS:")
print("-" * 20)

plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist([df_sample['original_token_count'], df_sample['filtered_token_count']], 
               bins=20, alpha=0.7, label=['With Stopwords', 'Without Stopwords'],
               color=['lightcoral', 'lightblue'], edgecolor='black')
axes[0, 0].set_title('Token Count Distribution: Before vs After Stop Word Removal')
axes[0, 0].set_xlabel('Number of Tokens')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(df_sample['reduction_percentage'], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
axes[0, 1].set_title('Distribution of Reduction Percentages')
axes[0, 1].set_xlabel('Reduction Percentage (%)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

sentiment_groups_before = [df_sample[df_sample['sentiment'] == sent]['original_token_count'] for sent in ['positive', 'negative']]
sentiment_groups_after = [df_sample[df_sample['sentiment'] == sent]['filtered_token_count'] for sent in ['positive', 'negative']]

box_data = sentiment_groups_before + sentiment_groups_after
labels = ['Pos (Before)', 'Neg (Before)', 'Pos (After)', 'Neg (After)']
axes[1, 0].boxplot(box_data, labels=labels)
axes[1, 0].set_title('Token Count by Sentiment: Before vs After')
axes[1, 0].set_ylabel('Number of Tokens')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

words, counts = zip(*most_common_filtered[:10])
axes[1, 1].barh(range(len(words)), counts, color='purple', alpha=0.7)
axes[1, 1].set_yticks(range(len(words)))
axes[1, 1].set_yticklabels(words)
axes[1, 1].set_title('Top 10 Words After Stop Word Removal')
axes[1, 1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('stopword_removal_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Stop word removal visualizations saved as 'stopword_removal_analysis.png'")

print("\n6. EFFECTIVENESS COMPARISON:")
print("-" * 30)

total_original_words = df_sample['original_token_count'].sum()
total_filtered_words = df_sample['filtered_token_count'].sum()
total_stopwords_removed = total_original_words - total_filtered_words
overall_reduction = (total_stopwords_removed / total_original_words) * 100

print(f"Overall effectiveness:")
print(f"{'='*30}")
print(f"Total original tokens: {total_original_words:,}")
print(f"Total tokens after removal: {total_filtered_words:,}")
print(f"Total stop words removed: {total_stopwords_removed:,}")
print(f"Overall reduction: {overall_reduction:.1f}%")

print("\n7. EXPORTING RESULTS:")
print("-" * 25)

export_df = df_sample[['review', 'sentiment', 'processed_text', 'original_token_count', 
                      'filtered_token_count', 'stopwords_removed', 'reduction_percentage']].copy()
export_df.to_csv('stopword_removed_imdb.csv', index=False)
print("✅ Stop word removed dataset exported as 'stopword_removed_imdb.csv'")

filtered_word_freq_df = pd.DataFrame(most_common_filtered, columns=['word', 'frequency'])
filtered_word_freq_df.to_csv('filtered_word_frequency.csv', index=False)
print("✅ Filtered word frequency data exported as 'filtered_word_frequency.csv'")

print("\n" + "="*60)
print("STOP WORD REMOVAL SUMMARY")
print("="*60)

print(f"""
📁 Dataset: IMDB Movie Reviews (Stop Word Removal)
📊 Sample size: {len(df_sample):,} reviews
🔤 Unique words after removal: {len(filtered_word_freq):,}
📈 Average tokens before removal: {df_sample['original_token_count'].mean():.1f}
📉 Average tokens after removal: {df_sample['filtered_token_count'].mean():.1f}
🎯 Overall reduction: {overall_reduction:.1f}%
💾 Files created:
   - stopword_removed_imdb.csv
   - filtered_word_frequency.csv
   - stopword_removal_analysis.png

✅ Stop word removal completed successfully!
   - {len(stop_words)} English stop words removed
   - Multiple removal methods demonstrated
   - HTML tags cleaned
   - Effectiveness analysis completed
""")

print("="*60)