
"""
Attention + LSH

=======

(A Strong Baseline for Query Efficient Attacks in a Black Box Setting)

"""
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InputColumnModification
)
from textattack.constraints.grammaticality.language_models import (
    LearningToWriteLanguageModel,
)
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import WordSwapHowNet
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack import Attack
from textattack.transformations import WordSwapWordNet
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.overlap import MaxWordsPerturbed

from .attack_recipe import AttackRecipe


class LSHWithAttentionWordNetPremise(AttackRecipe):
    @staticmethod
    def build(model, attention_model=None, product=False):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassification(model)

        search_method = GreedyWordSwapWIR("lsh_with_attention", attention_model_path=attention_model, product=product)
        return Attack(goal_function, constraints, transformation, search_method)


class LSHWithAttentionHowNetPremise(AttackRecipe):

    @staticmethod
    def build(model, attention_model=None, product=False):
        transformation = WordSwapHowNet()
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassification(model)
        search_method = GreedyWordSwapWIR(wir_method="lsh_with_attention", attention_model_path=attention_model, product=product)

        return Attack(goal_function, constraints, transformation, search_method)

class LSHWithAttentionEmbeddingPremise(AttackRecipe):

    @staticmethod
    def build(model, attention_model=None, product=False):
        transformation = WordSwapEmbedding(max_candidates=50)
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model)
        search_method = GreedyWordSwapWIR(wir_method="lsh_with_attention", attention_model_path=attention_model, product=product)

        return Attack(goal_function, constraints, transformation, search_method)


class LSHWithAttentionEmbeddingGenPremise(AttackRecipe):

    def build(model, attention_model=None, product=False):
        transformation = WordSwapEmbedding(max_candidates=8)
        constraints = []
        constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassification(model)
        search_method = GreedyWordSwapWIR(wir_method="lsh_with_attention", attention_model_path=attention_model, product=product)
        print(constraints)
        return Attack(goal_function, constraints, transformation, search_method)


class LSHWithAttentionWordNetHypothesis(AttackRecipe):
    @staticmethod
    def build(model, attention_model=None, product=False):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"sentence"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassification(model)

        search_method = GreedyWordSwapWIR("lsh_with_attention", attention_model_path=attention_model, product=product)
        return Attack(goal_function, constraints, transformation, search_method)


class LSHWithAttentionHowNetHypothesis(AttackRecipe):

    @staticmethod
    def build(model, attention_model=None, product=False):
        transformation = WordSwapHowNet()
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"sentence"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassification(model)
        search_method = GreedyWordSwapWIR(wir_method="lsh_with_attention", attention_model_path=attention_model, product=product)

        return Attack(goal_function, constraints, transformation, search_method)

class LSHWithAttentionEmbeddingHypothesis(AttackRecipe):

    @staticmethod
    def build(model, attention_model=None, product=False):
        transformation = WordSwapEmbedding(max_candidates=50)
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"sentence"}
        )
        constraints.append(input_column_modification)
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model)
        search_method = GreedyWordSwapWIR(wir_method="lsh_with_attention", attention_model_path=attention_model, product=product)

        return Attack(goal_function, constraints, transformation, search_method)


class LSHWithAttentionEmbeddingGenHypothesis(AttackRecipe):

    def build(model, attention_model=None, product=False):
        transformation = WordSwapEmbedding(max_candidates=8)
        constraints = []
        constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"sentence"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassification(model)
        search_method = GreedyWordSwapWIR(wir_method="lsh_with_attention", attention_model_path=attention_model, product=product)
        print(constraints)
        return Attack(goal_function, constraints, transformation, search_method)
