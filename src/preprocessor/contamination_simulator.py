# Text contamination functions
import random

from datasets import load_dataset

# Lazy, module-level cache for Gutenberg sentences to avoid repeated loads
_GUTENBERG_SENTENCES = None


def _get_gutenberg_sentences():
    global _GUTENBERG_SENTENCES
    if _GUTENBERG_SENTENCES is None:
        ds = load_dataset("sedthh/gutenberg_english", split="train")
        _GUTENBERG_SENTENCES = ds["TEXT"]
    return _GUTENBERG_SENTENCES


def swap_words(text):
    words = text.split()
    if len(words) < 2:
        return text
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return ' '.join(words)

def add_char_noise(text, noise_level=0.05):
    def insert(s): return s[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + s[i:]
    def delete(s): return s[:i] + s[i+1:] if len(s) > 1 else s
    def substitute(s): return s[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + s[i+1:]
    def swap(s): return s[:i] + s[i+1] + s[i] + s[i+2:] if i < len(s)-1 else s
    ops = [insert, delete, substitute, swap]
    words = text.split()
    for w in range(len(words)):
        if random.random() < noise_level and len(words[w]) > 0:
            i = random.randint(0, len(words[w])-1)
            words[w] = random.choice(ops)(words[w])
    return ' '.join(words)

def insert_irrelevant_text(text, gutenberg_sentences=None):
    if gutenberg_sentences is None:
        gutenberg_sentences = _get_gutenberg_sentences()
    irrelevant = random.choice(gutenberg_sentences)
    words = text.split()
    insert_pos = random.randint(0, len(words))
    return ' '.join(words[:insert_pos] + irrelevant.split() + words[insert_pos:])

def contaminate_text(text):
    """Contaminate a single text sample using simple perturbations and a random
    Gutenberg sentence insertion. The Gutenberg dataset is loaded once and cached.
    """
    text = swap_words(text)
    text = add_char_noise(text)
    text = insert_irrelevant_text(text)
    return text
