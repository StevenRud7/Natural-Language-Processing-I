import json
from collections import defaultdict, Counter
from math import log
from typing import (
    Iterable,
    Any,
    Sequence,
    Generator,
)

############################################################
# The following classes and methods are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


# DO NOT MODIFY
class AirlineSentimentInstance:
    """Represents a single instance from the airline sentiment dataset.

    Each instance contains the sentiment label, the name of the airline,
    and the sentences of text. The sentences are stored as a tuple of
    tuples of strings. The outer tuple represents sentences, and each
    sentences is a tuple of tokens."""

    def __init__(
        self, label: str, airline: str, sentences: Sequence[Sequence[str]]
    ) -> None:
        self.label: str = label
        self.airline: str = airline
        # These are converted to tuples so they cannot be modified
        self.sentences: tuple[tuple[str, ...], ...] = tuple(
            tuple(sentence) for sentence in sentences
        )

    def __repr__(self) -> str:
        return f"<AirlineSentimentInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; airline={self.airline}; sentences={self.sentences}"

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_dict(cls, json_dict: dict[str, Any]) -> "AirlineSentimentInstance":
        return AirlineSentimentInstance(
            json_dict["label"], json_dict["airline"], json_dict["sentences"]
        )


# DO NOT MODIFY
class SentenceSplitInstance:
    """Represents a potential sentence boundary in context.

    Each instance is labeled with whether it is ('y') or is not ('n') a sentence
    boundary, the characters to the left of the boundary token, the potential
    boundary token itself (punctuation that could be a sentence boundary), and
    the characters to the right of the boundary token."""

    def __init__(
        self, label: str, left_context: str, token: str, right_context: str
    ) -> None:
        self.label: str = label
        self.left_context: str = left_context
        self.token: str = token
        self.right_context: str = right_context

    def __repr__(self) -> str:
        return f"<SentenceSplitInstance: {str(self)}>"

    def __str__(self) -> str:
        return " ".join(
            [
                f"label={self.label};",
                f"left_context={repr(self.left_context)};",
                f"token={repr(self.token)};",
                f"right_context={repr(self.right_context)}",
            ]
        )

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_dict(cls, json_dict: dict[Any, Any]) -> "SentenceSplitInstance":
        return SentenceSplitInstance(
            json_dict["label"],
            json_dict["left"],
            json_dict["token"],
            json_dict["right"],
        )


# DO NOT MODIFY
def load_sentiment_instances(
    datapath: str,
) -> Generator[AirlineSentimentInstance, None, None]:
    """Load airline sentiment instances from a JSON file."""
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield AirlineSentimentInstance.from_dict(json_item)


# DO NOT MODIFY
def load_segmentation_instances(
    datapath: str,
) -> Generator[SentenceSplitInstance, None, None]:
    """Load sentence segmentation instances from a JSON file."""
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield SentenceSplitInstance.from_dict(json_item)


# DO NOT MODIFY
class ClassificationInstance:
    """Represents a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    """Compute the accuracy of the provided predictions."""
    correct = 0.0
    if len(predictions) != len(expected) or len(predictions)==0:
        raise ValueError()
    for x in range(len(predictions)):
        if predictions[x] == expected[x]:
            correct+=1
    return correct/len(predictions)


def recall(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the recall of the provided predictions."""
    tp = 0.0
    fn = 0.0
    if len(predictions) != len(expected) or len(predictions)==0:
        raise ValueError()
    for x in range(len(predictions)):
        if predictions[x] == label and expected[x] == label:
            tp+=1
        elif predictions[x] != label and expected[x] == label:
            fn+=1
    if tp + fn == 0.0:
        return 0.0
    return tp/(tp+fn)


def precision(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the precision of the provided predictions."""
    tp = 0.0
    fp = 0.0
    if len(predictions) != len(expected) or len(predictions)==0:
        raise ValueError()
    for x in range(len(predictions)):
        if predictions[x] == label and expected[x] == label:
            tp+=1
        elif predictions[x] == label and expected[x] != label:
            fp+=1
    if tp + fp == 0.0:
        return 0.0
    return tp/(tp+fp)


def f1(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the F1-score of the provided predictions."""
    p = precision(predictions,expected,label)
    r = recall(predictions,expected,label)
    if p + r == 0.0:
        return 0.0
    return 2 * ((p*r)/(p+r))

#####

def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]: #copied from previous HW and updated to fit this HW
    """Return the bigrams contained in a sequence."""
    s = set()
    s.add((START_TOKEN, sentence[0].lower()))
    for x in range(len(sentence) - 1):
        s.add((sentence[x].lower(), sentence[x + 1].lower()))
    s.add((sentence[len(sentence) - 1].lower(), END_TOKEN))
    l = list(s)
    return l

#####

class UnigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        """Extract unigram features from an instance."""
        label2 = instance.label
        lower_uni = set()
        for tups in instance.sentences:  # try list comp on this later
            for word in tups:
                lower_uni.add(word.lower())
        lower_uni2 = list(lower_uni)

        return ClassificationInstance(label2, lower_uni2)


class BigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        """Extract bigram features from an instance."""
        label2 = instance.label
        bigram = []
        bi_tups = []
        for tups in instance.sentences:
            bi_tups += bigrams(tups)
        for words in bi_tups:
            bigram.append(str(words))

        return ClassificationInstance(label2, bigram)


class BaselineSegmentationFeatureExtractor:
    @staticmethod
    def extract_features(instance: SentenceSplitInstance) -> ClassificationInstance:
        """Extract features for all three tokens from an instance."""
        label2 = instance.label
        feats =[]
        feats.append('left_tok='+str(instance.left_context))
        feats.append('split_tok='+str(instance.token))
        feats.append('right_tok='+str(instance.right_context))
        return ClassificationInstance(label2, feats)


class InstanceCounter:
    """Holds counts of the labels and features seen during training.

    See the assignment for an explanation of each method."""

    def __init__(self) -> None:
        self.label_counts = Counter()
        self.feature_counts = Counter()
        self.val_tot = 0
        self.uniq_label = set()
        self.uniq_label2 = []
        self.feat_set = set()
        self.dict_count = defaultdict(Counter)
        self.feat_for_lab = {}

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        # You should fill in this loop. Do not try to store the instances!
        for instance in instances:
            self.label_counts[instance.label] += 1
            self.feature_counts.update(instance.features)
            self.dict_count[instance.label].update(instance.features)
            self.val_tot+=1
            self.uniq_label.add(instance.label)
            #self.feat_set.add(instance.features)


        self.uniq_label2 = list(self.uniq_label)
        self.feat_set = set(self.feature_counts.keys())
        for la in self.uniq_label2:
            self.feat_for_lab[la] = self.dict_count[la].total()


    def label_count(self, label: str) -> int:
        #print(self.label_counts)
        return self.label_counts[label]

    def total_labels(self) -> int:
        #print(self.label_counts.values())
        return self.val_tot

    def feature_label_joint_count(self, feature: str, label: str) -> int:
        #print(self.dict_count)
        return self.dict_count[label][feature]

    def labels(self) -> list[str]:
        return self.uniq_label2

    def feature_vocab_size(self) -> int:
        #print(self.feat_set)
        return len(self.feature_counts.keys())

    def feature_set(self) -> set[str]:
        return self.feat_set

    def total_feature_count_for_label(self, label: str) -> int:
        return self.feat_for_lab[label]


class NaiveBayesClassifier:
    """Perform classification using naive Bayes.

    See the assignment for an explanation of each method."""

    # DO NOT MODIFY
    def __init__(self, k: float):
        self.k: float = k
        self.instance_counter: InstanceCounter = InstanceCounter()

    # DO NOT MODIFY
    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.instance_counter.count_instances(instances)

    def prior_prob(self, label: str) -> float:
        #print(self.instance_counter.label_count(label))
        return float(self.instance_counter.label_count(label) / self.instance_counter.total_labels())

    def likelihood_prob(self, feature: str, label) -> float:
        return (self.instance_counter.feature_label_joint_count(feature,label) + self.k) / (self.instance_counter.total_feature_count_for_label(label) + self.instance_counter.feature_vocab_size()*self.k)

    def log_posterior_prob(self, features: Sequence[str], label: str) -> float:
        s = 0.0
        #print(self.instance_counter)
        #print(features)
        for f in features:
            if f in self.instance_counter.feature_counts.keys():
                s += log(self.likelihood_prob(f,label))
        return s + log(self.prior_prob(label))

    def classify(self, features: Sequence[str]) -> str:
        p = []
        for l in self.instance_counter.labels():
            #print(self.prior_prob(f))
            p.append((self.log_posterior_prob(features,l),l))

        #print(p)
        return max(p)[1]

    def test(
        self, instances: Iterable[ClassificationInstance]
    ) -> tuple[list[str], list[str]]:
        pred = []
        tru = []
        for i in instances:
            # print(self.prior_prob(f))
            pred.append(self.classify(i.features))
            tru.append(i.label)
        tup = (pred,tru)
        #print(tup)
        return tup


# MODIFY THIS AND DO THE FOLLOWING:
# 1. Inherit from UnigramAirlineSentimentFeatureExtractor or BigramAirlineSentimentFeatureExtractor
#    (instead of object) to get an implementation for the extract_features method.
# 2. Set a value for self.k below based on your tuning experiments.
class TunedAirlineSentimentFeatureExtractor(UnigramAirlineSentimentFeatureExtractor):
    def __init__(self) -> None:
        self.k = float(0.5)
