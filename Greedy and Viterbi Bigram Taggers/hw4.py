from abc import abstractmethod, ABC
from collections import Counter
from collections import defaultdict
from math import log
from operator import itemgetter
from typing import Generator, Iterable, Sequence, TypeVar

############################################################
# The following constants, classes, and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
NEG_INF = float("-inf")

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"

T = TypeVar("T")

# DO NOT MODIFY
class TaggedToken:
    """Store the text and tag for a token."""

    # DO NOT MODIFY
    def __init__(self, text: str, tag: str):
        self.text: str = text
        self.tag: str = tag

    # DO NOT MODIFY
    def __str__(self) -> str:
        return f"{self.text}/{self.tag}"

    # DO NOT MODIFY
    def __repr__(self) -> str:
        return f"<TaggedToken {str(self)}>"

    # DO NOT MODIFY
    @classmethod
    def from_string(cls, s: str) -> "TaggedToken":
        """Create a TaggedToken from a string with the format "token/tag".

        While the tests use this, you do not need to.
        """
        splits = s.rsplit("/", 1)
        assert len(splits) == 2, f"Could not parse token: {repr(s)}"
        return cls(splits[0], splits[1])


# DO NOT MODIFY
class Tagger(ABC):
    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        """Train the part of speech tagger by collecting needed counts from sentences."""
        raise NotImplementedError

    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        """Tag a sentence with part of speech tags."""
        raise NotImplementedError

    # DO NOT MODIFY
    def tag_sentences(
        self, sentences: Iterable[Sequence[str]]
    ) -> Generator[list[str], None, None]:
        """Yield a list of tags for each sentence in the input."""
        for sentence in sentences:
            yield self.tag_sentence(sentence)

    # DO NOT MODIFY
    def test(
        self, tagged_sentences: Iterable[Sequence[TaggedToken]]
    ) -> tuple[list[str], list[str]]:
        """Return a tuple containing a list of predicted tags and a list of actual tags.

        Does not preserve sentence boundaries to make evaluation simpler.
        """
        predicted: list[str] = []
        actual: list[str] = []
        for sentence in tagged_sentences:
            predicted.extend(self.tag_sentence([tok.text for tok in sentence]))
            actual.extend([tok.tag for tok in sentence])
        return predicted, actual


# DO NOT MODIFY
def safe_log(n: float) -> float:
    """Return the log of a number or -inf if the number is zero."""
    return NEG_INF if n == 0.0 else log(n)


# DO NOT MODIFY
def max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return max(scores.items(), key=itemgetter(1))


# DO NOT MODIFY
def most_frequent_item(counts: Counter[str]) -> str:
    """Return the most frequent item in a Counter.

    In case of ties, the lexicographically first item is returned.
    """
    assert counts, "Counter is empty"
    return items_descending_value(counts)[0]


# DO NOT MODIFY
def items_descending_value(counts: Counter[str]) -> list[str]:
    """Return the keys in descending frequency, breaking ties lexicographically."""
    # Why can't we just use most_common? It sorts by descending frequency, but items
    # of the same frequency follow insertion order, which we can't depend on.
    # Why can't we just use sorted with reverse=True? It will give us descending
    # by count, but reverse lexicographic sorting, which is confusing.
    # So instead we used sorted() normally, but for the key provide a tuple of
    # the negative value and the key.
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return [key for key, value in sorted(counts.items(), key=_items_sort_key)]


# DO NOT MODIFY
def _items_sort_key(item: tuple[str, int]) -> tuple[int, str]:
    # This is used by items_descending_count, but you should never call it directly.
    return -item[1], item[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


class MostFrequentTagTagger(Tagger):
    def __init__(self) -> None:
        # Add an attribute to store the most frequent tag
        self.default_tag = None
        self.sent_count = Counter()
        self.tag_list = []


    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        for sent in sentences:
            for tok in sent:
                self.sent_count[tok.tag] += 1
        #print(most_frequent_item(self.sent_count))
        self.default_tag = most_frequent_item(self.sent_count)
        #print(type(self.default_tag))

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        self.tag_list = [self.default_tag] * len(sentence)
        return self.tag_list


class UnigramTagger(Tagger):
    def __init__(self) -> None:
        # Add data structures that you need here
        self.tok_tag_map = defaultdict(Counter)
        self.most_common_tag = None
        self.tag_count = Counter()
        self.uni_list = []
        self.uni_dic = {}

    def train(self, sentences: Iterable[Sequence[TaggedToken]]):
        for senten in sentences:
            for tt in senten:
                self.tag_count[tt.tag] +=1
                self.tok_tag_map[tt.text][tt.tag] +=1
        self.most_common_tag = most_frequent_item(self.tag_count)
        for k in self.tok_tag_map.keys():
            self.uni_dic[k] = most_frequent_item(self.tok_tag_map[k])

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        self.uni_list = [self.uni_dic.get(s,self.most_common_tag) for s in sentence]
        # for s in sentence:
        #     self.uni_list.append(self.uni_dic.get(s,self.most_common_tag))
        #print(self.uni_list)
        return self.uni_list

def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]: #Imported from previous HW and adapted for this HW
    """Return the bigrams contained in a sequence."""
    l = []
    l.append((START_TOKEN, sentence[0]))
    for x in range(len(sentence) - 1):
        l.append((sentence[x], sentence[x + 1]))
    #l.append((sentence[len(sentence) - 1], END_TOKEN))
    return l


class SentenceCounter:
    def __init__(self, k: float) -> None:
        self.k = k
        self.word_tag_dict = defaultdict(Counter)
        self.count_tag = Counter()
        self.V_count = defaultdict(Counter)
        self.tag_list = []
        self.bi_tot = {}
        self.sentence_list = []
        self.bi_sentence = []
        self.uniq_tag_list = []
        self.bi_count = defaultdict(Counter)
        # Add data structures that you need here

    def count_sentences(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        for sentence in sentences:
            # Fill in this loop
            #self.sentence_c += 1
            #self.sentence_list.append(sentence)
            self.tag_list = []
            for t in sentence:
                self.word_tag_dict[t.text][t.tag] += 1
                self.count_tag[t.tag] +=1
                self.V_count[t.tag][t.text] += 1
                self.tag_list.append(t.tag)
            self.sentence_list.append(self.tag_list)

        for sen in self.sentence_list:
            self.bi_sentence.extend(bigrams(sen))
        for b in self.bi_sentence:
            self.bi_count[b[0]][b[1]] +=1
        for k,v in self.bi_count.items():
            self.bi_tot[k] = v.total()
        #print(self.bi_tot)

        self.uniq_tag_list = items_descending_value(self.count_tag)


    def unique_tags(self) -> list[str]:
        return self.uniq_tag_list

    def emission_prob(self, tag: str, word: str) -> float:
        if (self.count_tag[tag] + (self.V_count[tag][word] * self.k)) == 0:
            return 0.0
        return (self.word_tag_dict[word][tag] + self.k) / (self.count_tag[tag] + (len(self.V_count[tag]) * self.k))

    def transition_prob(self, prev_tag: str, current_tag: str) -> float:
        if not self.bi_tot[prev_tag] > 0:
            return 0.0
        return self.bi_count[prev_tag][current_tag] / self.bi_tot[prev_tag]

    def initial_prob(self, tag: str) -> float:
        if tag not in self.bi_count:
            return 0.0
        return self.bi_count[START_TOKEN][tag] / self.bi_tot[START_TOKEN]


class BigramTagger(Tagger, ABC):
    # You can add additional methods to this class if you want to share anything
    # between the greedy and Viterbi taggers. However, do not modify any of the
    # implemented methods.

    def __init__(self, k) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter = SentenceCounter(k)

    def train(self, sents: Iterable[Sequence[TaggedToken]]) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter.count_sentences(sents)

    def sequence_probability(self, sentence: Sequence[str], tags: Sequence[str]) -> float:
        """Return the probability for a sequence of tags given tokens."""
        s = 0.0
        for st in range(len(sentence)-1):
            s += safe_log(self.counter.emission_prob(tags[st],sentence[st]))
            #s += safe_log(self.counter.initial_prob(tags[st]))
            s += safe_log(self.counter.transition_prob(tags[st],tags[st+1]))
        return s + safe_log(self.counter.emission_prob(tags[len(sentence)-1],sentence[len(sentence)-1])) + safe_log(self.counter.initial_prob(tags[0]))


class GreedyBigramTagger(BigramTagger):
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        greedy_list = []
        #t_dict = {}
        tag_word_dict ={}
        prev_t = None
        for u in self.counter.unique_tags():
            tag_word_dict[u] = safe_log(self.counter.initial_prob(u)) + safe_log(self.counter.emission_prob(u,sentence[0]))
            #word_dict[u] = safe_log(self.counter.emission_prob(u,sentence[0]))
        greedy_list.append(max_item(tag_word_dict)[0])
        prev_t = max_item(tag_word_dict)[0]
        for s in range(1,len(sentence)):
            for u2 in self.counter.unique_tags():
                tag_word_dict[u2] = safe_log(self.counter.emission_prob(u2,sentence[s])) + safe_log(self.counter.transition_prob(prev_t,u2))
            greedy_list.append(max_item(tag_word_dict)[0])
            prev_t = max_item(tag_word_dict)[0]
        #print(greedy_list)
        return greedy_list


class ViterbiBigramTagger(BigramTagger):
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        vit_list = []
        backp = [{}]
        ins = 0
        tag_word_dict2 = {}
        twd = None
        p = None
        word_dict2 = {}
        inner_backp = {}
        best_inner = {}
        for u in self.counter.unique_tags():
            tag_word_dict2[u] = safe_log(self.counter.initial_prob(u)) + safe_log(self.counter.emission_prob(u,sentence[0]))
        #vit_list.append(max_item(tag_word_dict2)[0])
        #previous_t = max_item(tag_word_dict2)[0]
        for se in range(1,len(sentence)):
            for u2 in self.counter.unique_tags():
                for u3 in self.counter.unique_tags():
                    twd = tag_word_dict2[u3]
                    best_inner[u3] = twd + safe_log(self.counter.emission_prob(u2,sentence[se])) + safe_log(self.counter.transition_prob(u3,u2))

                inner_backp[u2] = max_item(best_inner)[0]
                word_dict2[u2] = max_item(best_inner)[1]
                best_inner.clear()
                #print(best_inner)

            tag_word_dict2 = word_dict2
            backp.append(inner_backp)
            #print(word_dict2)
            word_dict2 = {}
            inner_backp = {}
        #print(len(backp))
        vit_list.append(max_item(tag_word_dict2)[0])
        for bp in range(len(backp)-1,0,-1):
            if bp == len(backp)-1:
                p = max_item(tag_word_dict2)[0]
            vit_list.insert(ins,backp[bp][p])
            p = backp[bp][p]
        #print(vit_list)
        return vit_list
