import re
from collections import defaultdict

def get_stats(vocab):
    """
    Given a vocabulary (dictionary mapping words to frequency counts), returns a 
    dictionary of tuples representing the frequency count of pairs of characters 
    in the vocabulary.
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """
    Given a pair of characters and a vocabulary, returns a new vocabulary with the 
    pair of characters merged together wherever they appear.
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_vocab(data):
    """
    Given a list of strings, returns a dictionary of words mapping to their frequency 
    count in the data.
    """
    vocab = defaultdict(int)
    for line in data:
        word = line.strip()
        vocab[' '.join(list(word))] += 1  # No end-of-word token
    return vocab

def compute_scores(pairs, token_freq):
    """
    Computes the score for each pair based on the frequency of the bigram divided by 
    the product of the frequencies of the individual tokens.
    """
    scores = {}
    for (x, y), freq_xy in pairs.items():
        score = freq_xy / (token_freq[x] * token_freq[y])
        scores[(x, y)] = score
    return scores

def get_token_frequencies(vocab):
    """
    Computes the frequency of each token in the vocabulary.
    """
    token_freq = defaultdict(int)
    for word, freq in vocab.items():
        tokens = word.split()
        for token in tokens:
            token_freq[token] += freq
    return token_freq

def byte_pair_encoding(data, n):
    """
    Given a list of strings and an integer n, performs n iterations of byte-pair encoding.
    """
    vocab = get_vocab(data)
    for i in range(n):
        pairs = get_stats(vocab)
        token_freq = get_token_frequencies(vocab)
        scores = compute_scores(pairs, token_freq)
        
        # Print bigram frequencies and scores
        print(f"Iteration {i + 1} Bigram Values and Scores:")
        for bigram, score in scores.items():
            print(f"  Bigram: {bigram} | Frequency: {pairs[bigram]} | Score: {score:.4f}")
        
        # Select the best pair based on the highest score
        best_pair = max(scores, key=scores.get)
        best_score = scores[best_pair]
        
        # Merge the best pair in the vocabulary
        vocab = merge_vocab(best_pair, vocab)
        
        # Print results of the iteration
        print(f"\nIteration {i + 1} Summary:")
        print(f"  Best Merge: {best_pair[0]}{best_pair[1]} → {''.join(best_pair)}")
        print(f"  Best Score: {best_score:.4f}")
        print(f"  Updated Vocabulary: {vocab}")
        print("-" * 50)
        
    return vocab

# Define the corpus
corpus = 'crazy_, hazy_, day_'
data = corpus.split(',')

# Run BPE for 4 iterations
n = 4
bpe_vocab = byte_pair_encoding(data, n)
print("\nFinal Vocabulary:", bpe_vocab)
