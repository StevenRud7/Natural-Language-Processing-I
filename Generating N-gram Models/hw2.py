import random
from collections import defaultdict, Counter
from math import log
from typing import Sequence, Iterable, Generator, TypeVar

# hw2.py
# Version 1.1
# 9/26/2022

############################################################
# The following constants and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
random.seed(0)

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"
# DO NOT MODIFY
NEG_INF = float("-inf")
# DO NOT MODIFY (needed if you copy code from HW 1)
T = TypeVar("T")


# DO NOT MODIFY
def load_tokenized_file(path: str) -> Generator[Sequence[str], None, None]:
    """Yield sentences as sequences of tokens."""
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.rstrip("\n")
            tokens = line.split(" ")
            yield tuple(tokens)


# DO NOT MODIFY
def sample(probs: dict[str, float]) -> str:
    """Return a sample from a distribution."""
    # To avoid relying on the dictionary iteration order, sort items
    # This is very slow and should be avoided in general, but we do
    # it in order to get predictable results
    items = sorted(probs.items())
    # Now split them back up into keys and values
    keys, vals = zip(*items)
    # Choose using the weights in the values
    return random.choices(keys, weights=vals)[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


#####
#HW1 Code:
def counts_to_probs(counts: Counter[T]) -> dict[T, float]:
    """Return a dict with the input counts converted to probabilities."""
    #s = 0
    d ={}
    for key, val in counts.items():
        d[key] = (val / sum(counts.values()))
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

####

def bigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[str, dict[str, float]]:
    """Return bigram probabilities computed from the provided sequences."""
    c = Counter()
    d = defaultdict(Counter)
    mainD = {}
    for s in sentences:
        c.update(bigrams(s))
    #d.update(c)
    for key,val in c.items():
        #print(key)
        d[key[0]][key[1]] = val
    for key,val in d.items():
        #print(val)
        mainD[key] = counts_to_probs(val)
    return mainD


def trigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Return trigram probabilities computed from the provided sequences."""
    c = Counter()
    d = defaultdict(Counter)
    mainD = {}
    for s in sentences:
        c.update(trigrams(s))
    for key,val in c.items():
        d[key[:2]][key[2]] = val
    for key,val in d.items():
        #print(key)
        mainD[key] = counts_to_probs(val)
    return mainD


def sample_bigrams(probs: dict[str, dict[str, float]]) -> list[str]:
    """Generate a sequence by sampling from the provided bigram probabilities."""
    s = START_TOKEN
    l = []
    while s != END_TOKEN:
        s = sample(probs[s])
        #s = s[0]
        if s == END_TOKEN:
            break
        #print(s)
        l.append(s)
    return l


def sample_trigrams(probs: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    """Generate a sequence by sampling from the provided trigram probabilities."""
    s = (START_TOKEN,START_TOKEN)
    l = []
    s2 = ""
    while s2 != END_TOKEN:
        s2 = sample(probs[s])
        s = (s[1],s2)
        if s2 == END_TOKEN:
            break
        l.append(s2)
    return l


def bigram_sequence_prob(
    sequence: Sequence[str], probs: dict[str, dict[str, float]]
) -> float:
    """Compute the probability of a sequence using bigram probabilities."""
    s = START_TOKEN
    tot = 0.0
    for x in sequence:
        d = probs[s]
        #print(d)
        if x not in d:
            return NEG_INF
        p = d[x] #p for probability
        s = x
        #print(p)
        tot += log(p)
    return tot


def trigram_sequence_prob(
    sequence: Sequence[str], probs: dict[tuple[str, str], dict[str, float]]
) -> float:
    """Compute the probability of a sequence using trigram probabilities."""
    s = (START_TOKEN, START_TOKEN)
    tot = 0.0
    for x in sequence:
        d = probs[s]
        #print(d)
        if x not in d:
            return NEG_INF
        p = d[x]
        s = (s[1],x)
        #print(s)
        tot += log(p)
    return tot
