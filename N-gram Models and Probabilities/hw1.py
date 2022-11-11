from collections import Counter, defaultdict

from typing import Iterable, TypeVar, Sequence

# DO NOT MODIFY
T = TypeVar("T")

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


def counts_to_probs(counts: Counter[T]) -> defaultdict[T, float]:
    """Return a defaultdict with the input counts converted to probabilities."""
    sum = 0
    d = defaultdict(float)
    for val in counts.values():
        #print(val)
        sum += val
    for key, val in counts.items():
        d[key] = (val / sum)
    return d


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    """Return the bigrams contained in a sequence."""
    l = []
    l.append((START_TOKEN, sentence[0]))
    for x in range(len(sentence) - 1):
        l.append((sentence[x], sentence[x + 1]))
    l.append((sentence[len(sentence) - 1], END_TOKEN))
    return l


def trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return the trigrams contained in a sequence."""
    l =[]
    l.append((START_TOKEN,START_TOKEN, sentence[0]))
    if len(sentence) > 1:
        l.append((START_TOKEN,sentence[0], sentence[1]))
        for x in range(len(sentence)-2):
            l.append((sentence[x],sentence[x+1],sentence[x+2]))
        l.append((sentence[len(sentence)-2],sentence[len(sentence)-1],END_TOKEN))
    else:
        l.append((START_TOKEN,sentence[0],END_TOKEN))
    l.append((sentence[len(sentence)-1],END_TOKEN,END_TOKEN))
    return l


def count_unigrams(sentences: Iterable[list[str]], lower: bool = False) -> Counter[str]:
    """Count the unigrams in an iterable of sentences, optionally lowercasing."""
    c = Counter()
    count = 0
    for x in sentences:
        if lower == True:
            for y in x:
                x[count] = x[count].lower()
                count+=1
            count = 0
        c.update(x)
    return c


def count_bigrams(
    sentences: Iterable[list[str]], lower: bool = False
) -> Counter[tuple[str, str]]:
    """Count the bigrams in an iterable of sentences, optionally lowercasing."""
    c = Counter()
    count = 0
    for x in sentences:
        if lower == True:
            for y in x:
                x[count] = x[count].lower()
                count+=1
            count=0
        x = bigrams(x)
        c.update(x)
    return c


def count_trigrams(
    sentences: Iterable[list[str]], lower: bool = False
) -> Counter[tuple[str, str, str]]:
    """Count the trigrams in an iterable of sentences, optionally lowercasing."""
    c = Counter()
    count = 0
    for x in sentences:
        if lower == True:
            for y in x:
                x[count] = x[count].lower()
                count+=1
            count=0
        x = trigrams(x)
        c.update(x)
    return c
