# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from perm_equivariant_seq2seq.symmetry_groups import LanguageInvariance
from itertools import product, combinations
from typing import List
from functools import reduce
from operator import mul

# from perm_equivariant_seq2seq.g_utils import cyclic_group, cyclic_group_generator, cyclic_group_inv, get_group_product


class Language:
    """Object to keep track of languages to be translated.

    Args:
        name: (string) Name of language being used
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        """Process a sentence and add words to language vocabulary"""
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """Add a word to the language vocabulary"""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class InvariantLanguage(Language):
    """Object to track a language with a fixed set of invariances

    Args:
        name (string): Name of language being used
        invariances (list::invariances): A list of invariance objects representing language invariances.
    """
    def __init__(self, name, invariances):
        super(InvariantLanguage, self).__init__(name)
        self.invariances = invariances

    def add_word(self, word):
        """Add a word to language vocabulary"""
        word = self.map_word(word)
        super(InvariantLanguage, self).add_word(word)

    def map_word(self, word):
        """Map a word to its equivalence class"""
        for invariance in self.invariances:
            word = invariance.map_word(word)
        return word

    def map_sentence(self, sentence):
        """Process a sentence and map all words to their equivalence classes"""
        return ' '.join([self.map_word(word) for word in sentence.split(' ')])


class EquivariantLanguage(Language):
    """Object to track a language with a fixed (and known) set of equivariances

    Args:
        name (string): Name of language being used
        equivariant_words (list::strings): List of words in language that are equivariant
    """
    def __init__(self, name, equivariant_words):
        super(EquivariantLanguage, self).__init__(name)
        self.equivariant_words = equivariant_words

    def rearrange_indices(self):
        """Rearrange the language indexing such that the first N words after the

        Returns:
            None
        """
        num_fixed_words = 2
        other_words = [w for w in self.word2index if w not in self.equivariant_words]
        for idx, word in enumerate(self.equivariant_words):
            w_idx = idx + num_fixed_words
            self.word2index[word] = w_idx
            self.index2word[w_idx] = word
        for idx, word in enumerate(sorted(other_words)):
            w_idx = idx + num_fixed_words + self.num_equivariant_words
            self.word2index[word] = w_idx
            self.index2word[w_idx] = word

    @property
    def num_equivariant_words(self):
        return len(self.equivariant_words)

    @property
    def num_fixed_words(self):
        return 2

    @property
    def num_other_words(self):
        return len([w for w in self.word2index if w not in self.equivariant_words])


# Define SCAN language invariances
VERB_INVARIANCE = LanguageInvariance(['jump', 'run', 'walk', 'look'], 'verb')
DIRECTION_INVARIANCE = LanguageInvariance(['right', 'left'], 'direction')
CONJUNCTION_INVARIANCE = LanguageInvariance(['and', 'after'], 'conjunction')
ADVERB_INVARIANCE = LanguageInvariance(['once', 'twice', 'thrice'], 'adverb')
OTHER_INVARIANCE = LanguageInvariance(['around', 'opposite'], 'other')


def get_invariances(args):
    """Helper function to store some standard equivariances"""
    invariances = []
    if args.verb_invariance:
        invariances.append(VERB_INVARIANCE)
    if args.direction_invariance:
        invariances.append(DIRECTION_INVARIANCE)
    if args.conjunction_invariance:
        invariances.append(CONJUNCTION_INVARIANCE)
    if args.adverb_invariance:
        invariances.append(ADVERB_INVARIANCE)
    if args.other_invariance:
        invariances.append(OTHER_INVARIANCE)
    return invariances


class SCANGroup:

    def __init__(self, equivariance: list, commands: Language, actions: Language, canonical=False):

        self.eq_commands_indices = [[commands.word2index[word] for word in self.get_in_equivariant_words(eq)] for eq in equivariance]
        self.eq_word_indices = [[actions.word2index[word] for word in self.get_out_equivariant_words(eq)] for eq in equivariance]

        self.in_g = CyclicGroup(commands.n_words, self.eq_commands_indices, canonical)
        self.out_g = CyclicGroup(actions.n_words, self.eq_word_indices, canonical)

    def get_in_equivariant_words(self, eq):
        if eq == 'verb':
            return ['jump', 'run']
        elif eq == 'direction-rl':
            return ['right', 'left']
        elif eq == 'direction-ud':
            return ['up', 'down']
        else:
            return []

    def get_out_equivariant_words(self, eq):
        if eq == 'verb':
            return ['JUMP', 'RUN']
        elif eq == 'direction-rl':
            return ['TURN_RIGHT', 'TURN_LEFT']
        elif eq == 'direction-ud':
            return ['TURN_UP', 'TURN_DOWN']
        else:
            return []


class CyclicGroup:
    def __init__(self, n_words: int, indices: List[list], canonical=False):
        gen = self.generator(vocab_size=n_words, eq_indices=indices)
        self.g = self.group(generators=gen, group_sizes=list(map(len, indices)), vocab_size=n_words)
        self.cinv = self.group_canonical_inverse(indices=indices, vocab_size=n_words)
        if canonical:
            self.g = [self.group_canonical_inverse_product(n_words)]
    def group_canonical_inverse_product(self, vocab_size):
        cinv = {i: i for i in range(vocab_size)}
        for inv in self.cinv:
            cinv = {i: inv[cinv[i]] for i in range(vocab_size)}
        return cinv

    def generator(self, vocab_size, eq_indices):
        """
        :param vocab_size: size of the vocab
        :param eq_indices: indices of the equivariant words
        :return: a group generator of the required cyclic group consisting of all the equivariant words
        """
        generators = []
        for inds in eq_indices:
            g = {i: i for i in range(vocab_size)}  # generator initialized as id
            group_size = len(inds)
            for j in range(group_size):
                next_group_element = (j + 1) % group_size
                g[inds[j]] = inds[next_group_element]
            g['size'] = group_size  # add length of the group as a value
            generators.append(g)
        return generators

    def group_canonical_inverse(self, indices: list, vocab_size: int):
        invs = []
        for inds in indices:
            g = {i: i for i in range(vocab_size)}
            rep = inds[0]

            for i in inds:
                g[i] = rep
            invs.append(g)
        return invs

    def group(self, generators: list, group_sizes: list, vocab_size: int):
        """
        :param g: cyclic group generator
        :param group_size: size of the group
        :return: return a list of elements of a cyclic group
        """
        G = []
        for i in range(len(group_sizes) + 1):
            for gens in combinations(generators, i):
                g = {i: i for i in range(vocab_size)}
                for gen in gens:
                    g = {i: gen[g[i]] for i in range(vocab_size)}
                G.append(g)
        return G


class SCANMultiGroup:
    def __init__(self, equivariance, commands, actions, product=False, canonical=False):
        groups = [SCANGroup([eq], commands, actions) for eq in equivariance]
        self.in_g = [g.in_g for g in groups]
        self.out_g = [g.out_g for g in groups]
        self.eq_word_indices = [g.eq_word_indices for g in groups]