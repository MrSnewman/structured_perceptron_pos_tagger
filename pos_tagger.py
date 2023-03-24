# CS114B Spring 2023 Homework 4
# Part-of-speech Tagging with Structured Perceptrons

import os
import numpy as np
from collections import defaultdict
from random import Random

class POSTagger():

    def __init__(self):
        # for testing with the toy corpus from the lab 7 exercise
        self.tag_dict = {}
        self.word_dict = {}
        self.initial: np.array = None
        self.transition: np.array = None
        self.emission: np.array = None
        self.weights: np.array = None
        # should raise an IndexError; if you come across an unknown word, you
        # should treat the emission scores for that word as 0
        self.unk_index = np.inf

        self.vocab = set()
        self.tags = set()
        self.FULL_STOP = '.'

    '''
    Fills in self.tag_dict and self.word_dict, based on the training data.
    '''
    def make_dicts(self, train_set):
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                if name.startswith('c'):    # Started getting a codec error
                    with open(os.path.join(root, name)) as file:
                        for line in file:
                            clean_line = line.rstrip("\n")
                            tokens = clean_line.split()
                            for token in tokens:
                                split_token = token.rsplit('/', 1)
                                self.vocab.add(split_token[0])
                                self.tags.add(split_token[1])

        # Fill in word and tag dicts from vocab and tags collected in train_set walk
        for i, word in enumerate(self.vocab):
            self.word_dict[word] = i
        for i, tag in enumerate(self.tags):
            self.tag_dict[tag] = i

    '''
    Loads a dataset. Specifically, returns a list of sentence_ids, and
    dictionaries of tag_lists and word_lists such that:
    tag_lists[sentence_id] = list of part-of-speech tags in the sentence
    word_lists[sentence_id] = list of words in the sentence
    '''
    def load_data(self, data_set):
        sentence_ids: list[int] = []
        tag_lists: dict[int, list] = defaultdict(list)
        word_lists: dict[int, list] = defaultdict(list)
        # iterate over documents
        sentence_id: int = 0
        for root, dirs, files in os.walk(data_set):
            for name in files:
                if name.startswith('c'):    # Started getting a codec error
                    with open(os.path.join(root, name)) as file:
                        # be sure to split documents into sentences here
                        for line in file:
                            if line != "\n" and line != "":
                                sentence_id += 1
                                sentence_ids.append(sentence_id)
                                clean_line = line.strip("\n\t")
                                tokens = clean_line.split()
                                for token in tokens:
                                    split_token = token.rsplit('/', 1)
                                    word_lists[sentence_id].append(split_token[0])
                                    tag_lists[sentence_id].append(split_token[1])

        # I can't seem to avoid having some sentences of length 0
        for sentence_id in sentence_ids:
            if len(word_lists[sentence_id]) == 0:
                sentence_ids.remove(sentence_id)

        return sentence_ids, tag_lists, word_lists

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        # initialization step
        first_word = sentence[0]
        if first_word in self.vocab:
            z1 = self.initial + self.emission[self.word_dict[sentence[0]]]
        else:   # If word not in vocab, emission score is 0
            z1 = self.initial
        # Add to viterbi trellis
        v[:, 0] = z1
        backpointer[:, 0] = 0
        # recursion step
        for t in range(T - 1):
            word = sentence[t + 1]
            if word in self.vocab:
                z = self.transition + self.emission[self.word_dict[word]]
            else:   # If word not in vocab, emission score is 0
                z = self.transition
            # Add to viterbi trellis
            v[:, t + 1] = np.max(v[:, t:t+1] + z, axis=0)
            backpointer[:, t + 1] = np.argmax(v[:, t:t+1] + z, axis=0)
        # termination step
        best_path_pointer = np.argmax(v[:, T - 1])
        best_path = []
        for t in range(T - 1, -1, -1):
            best_path.append(best_path_pointer)
            best_path_pointer = backpointer[best_path_pointer, t]
        # Path traced backward so reverse before returning
        best_path.reverse()
        return best_path

    '''
    Trains a structured perceptron part-of-speech tagger on a training set.
    '''
    def train(self, train_set):
        self.make_dicts(train_set)
        sentence_ids, tag_lists, word_lists = self.load_data(train_set)
        Random(0).shuffle(sentence_ids)
        self.initial = np.zeros(len(self.tag_dict))
        self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
        self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        for i, sentence_id in enumerate(sentence_ids):
            sentence = word_lists[sentence_id]
            # Convert tags to indices to compare to prediction
            Y = [self.tag_dict[y] for y in tag_lists[sentence_id]]
            Y_hat = self.viterbi(sentence)
            # Convert words to indices to increment weights
            word_indices = [self.word_dict[word] for word in sentence]
            if Y != Y_hat:
                y_prev = None
                y_hat_prev = None
                for j, (y, y_hat) in enumerate(zip(Y, Y_hat)):
                    if j == 0:
                        # Update weights of PI matrix
                        self.initial[y_hat] -= 1
                        self.initial[y] += 1
                        # Store tags as previous for transition weights
                        y_prev = y
                        y_hat_prev = y_hat
                    else:
                        # Update weights of B matrix
                        self.emission[word_indices[j], y_hat] -= 1
                        self.emission[word_indices[j], y] += 1
                        # Update weights of A matrix
                        self.transition[y_hat_prev, y_hat] -= 1
                        self.transition[y_prev, y] += 1
                        # Store tags as previous for transition weights
                        y_prev = y
                        y_hat_prev = y_hat

            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of part-of-speech tags such that:
    results[sentence_id]['correct'] = correct sequence of tags
    results[sentence_id]['predicted'] = predicted sequence of tags
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        sentence_ids, tag_lists, word_lists = self.load_data(dev_set)
        for i, sentence_id in enumerate(sentence_ids):
            sentence = word_lists[sentence_id]
            tags = tag_lists[sentence_id]
            # Make sure tags in test set are in tag_dict to avoid key error during testing
            if all(tag in self.tag_dict for tag in tags):
                # Convert tags to indices to compare to prediction
                results[sentence_id]['correct'] = \
                    [self.tag_dict[y] if y in self.tag_dict else self.unk_index for y in tag_lists[sentence_id]]
                results[sentence_id]['predicted'] = self.viterbi(sentence)

            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        accuracy = 0.0
        acc = 0
        words = 0
        for tag_seq in results.values():
            for correct, predicted in zip(tag_seq['correct'], tag_seq['predicted']):
                if correct == predicted:
                    acc += 1
                words += 1
        accuracy = acc/words
        return accuracy


if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('brown/train')
    # pos.train('data_med/train')
    # pos.train('data_small/train')
    results = pos.test('brown/dev')
    # results = pos.test('data_med/dev')
    print('Accuracy:', pos.evaluate(results))
