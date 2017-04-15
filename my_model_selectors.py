import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        """Initialize the variables to hold values"""
        best_model = None
        best_bic = float('inf')

        """Loop for optimal BIC value and model"""
        for n in range(self.min_n_components, self.max_n_components + 1):
            score = None
            model = self.base_model(n)
            if model is None:
                continue
            try:
                score = model.score(self.X, self.lengths)
            except:
                continue
            bic = -2 * score + (n ** 2 + (2 * len(self.X[0]) * n)) * np.log(len(self.X))
            if bic < best_bic:
                best_bic = bic
                best_model = model
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_dic = float('-inf')
        best_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            M = 1.0
            try:
                model = self.base_model(n)
                score_curr = model.score(self.X, self.lengths)
            except:
                continue

            score_all_not_curr = 0
            for word in self.hwords.keys():
                if word != self.this_word:
                    X, y = self.hwords[word]
                    try:
                        word_score = model.score(X, y)
                        M = M + 1.0
                    except:
                        word_score = 0
                    score_all_not_curr += word_score

            curr_dic = score_curr - (1 / (M - 1)) * score_all_not_curr * 1.0

            if curr_dic > best_dic:
                best_dic = curr_dic
                best_model = model
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = -float('Inf')
        best_model = None
        split_method = KFold(n_splits=min(len(self.lengths), 3), shuffle=True, random_state=self.random_state)

        for n in range(self.min_n_components, self.max_n_components + 1):
            score_sum = 0
            count = 1
            model = None
            for train_index, test_index in split_method.split(self.sequences):

                X_train, X_test = [], []
                for idx in train_index:
                    X_train += self.sequences[idx]
                for idx in test_index:
                    X_test += self.sequences[idx]

                X_train, X_test = np.array(X_train), np.array(X_test)
                y_train, y_test = np.array(self.lengths)[train_index], np.array(self.lengths)[test_index]

                try:
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, y_train)
                    score_curr = model.score(X_test, y_test)
                    count += 1
                except:
                    score_curr = 0
                score_sum += score_curr

            score_avg = score_sum / (count * 1.0)

            if score_avg > best_score:
                best_score = score_avg
                best_model = model

        return best_model
