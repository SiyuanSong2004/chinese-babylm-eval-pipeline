import os
import pandas as pd
from pyparsing import col
import numpy as np
from scipy.stats import spearmanr, pearsonr
from utills import find_valid_words, find_vocab_word, get_eye_features_matrix, merge_eye_matrix
import time
import json


DATA_PATH = "/mnt/backup/zhiheng2/chinese-babylm-2026/evaluation_data/cogbench/"
_SENTENCE_PATH = f"{DATA_PATH}/eye_tracking/Sentences.xlsx"
_ROI_PATH = f"{DATA_PATH}/eye_tracking/ROIs.xlsx"
_MAIN_MEASURES = f"{DATA_PATH}/eye_tracking/Main Measures.xlsx"
_SUPPLEMENTARY_MEASURES = f"{DATA_PATH}/eye_tracking/Supplementary Measures.xlsx"
_SENTENCE_LEVEL_DATA = f"{DATA_PATH}/eye_tracking/eye_features_sentence_level.json"

# _ZUCO_SENTENCE_DATA_PATH = "data/zuco-corpus/eye_features_sentence_level.json"
_ZUCO_SENTENCE_DATA_PATH = f"{DATA_PATH}/zuco-corpus/eye_features_sentence_level.zuco1_2.json"

class EyeTrackingFeatures:
    # word vocabulary
    vocab = None
    word2idx = None
    idx2word = None

    sentence_data = None
    # data for split sentences
    roi_data = None
    roi_data_unique = None
    
    # word-level word features
    mean_features = None

    # sentence_level info
    sent_level_data = None
    zuco_sent_level_data = None
    english_sentences = None
    
    def __init__(self) -> None:
        self.sentence_data = pd.read_excel(_SENTENCE_PATH, index_col=1, header=0, sheet_name= "Sheet 1")

        self.main_measures= pd.read_excel(_MAIN_MEASURES, header=0, sheet_name= "Sheet 1")
        self.supplementary_measures = pd.read_excel(_SUPPLEMENTARY_MEASURES, header=0, sheet_name= "Sheet 1")
        self.all_measures = self.main_measures.set_index("words").join(self.supplementary_measures.set_index("words"),lsuffix="_main",rsuffix="_supple")
        
        # 9 eye-tracking features
        self.mean_features = self.all_measures.filter(regex='^Mean',axis=1)

        # read roi_data for sentence split
        self.roi_data = pd.read_excel(_ROI_PATH, header=0, sheet_name= "Sheet 1")
        self.roi_data_unique = self.roi_data.drop(columns=["Experiment","ROI_Beginning"]).drop_duplicates()

    def get_sentence(self, idx:int):
        return self.sentence_data.at[idx,"Sentence"]

    def get_sentences(self):
        return self.sentence_data.values
    
    def get_sentence_level_info(self, idx:int):
        if self.sent_level_data is None:
            with open(_SENTENCE_LEVEL_DATA, "r") as f:
                self.sent_level_data = json.load(f)
        return self.sent_level_data[str(idx)]
    
    def get_english_sentences(self):
        if self.english_sentences is not None:
            return self.english_sentences

        if self.zuco_sent_level_data is None:
            with open(_ZUCO_SENTENCE_DATA_PATH, "r") as f:
                self.zuco_sent_level_data = json.load(f)
        
        self.english_sentences = []
        for k in self.zuco_sent_level_data.keys():
            # only retain normal reading condition (SR or NR)
            exp_valid = ["SR", "NR", "ZUCO2-NR"]
            if self.zuco_sent_level_data[k]["split_experiments"] in exp_valid:
                self.english_sentences.append((k, self.zuco_sent_level_data[k]['content']))
        return self.english_sentences
    
    def get_english_sentence_level_info(self, idx:int):
        if self.zuco_sent_level_data is None:
            with open(_ZUCO_SENTENCE_DATA_PATH, "r") as f:
                self.zuco_sent_level_data = json.load(f)
        
        return self.zuco_sent_level_data[str(idx)]
    
    def get_english_sentence_splited(self, idx:int):
        sent_info = self.get_english_sentence_level_info(idx)
        return sent_info['num'], sent_info['all_split']

    def get_sentence_level_mean_sd(self, vocab:dict, rm_3chars=False, valid_min=3):
        eye_matrixs = None
        sentence_dict = self.get_sentence_dict()

        for idx, sentence in sentence_dict.items():
            num_split, split_words_list = self.get_sentence_splited(idx)

            for j in range(num_split):
                valid_words = find_valid_words(split_words_list[j]) if rm_3chars else None
                valid_num, valid_index = find_vocab_word(split_words_list[j],vocab,valid_words)

                if valid_num < valid_min:
                    print(f"Calculate the mean and sd of eye_matrixs: Sentence-{idx}, split-{j} is ignored due to insufficent valid words({valid_num}<{valid_min})")
                    continue

                eye_matrix = get_eye_features_matrix(
                                split_feature=self.get_sentence_level_info(idx)['split_features'][str(j)],
                                valid_num=valid_num,
                                valid_index=valid_index)
                
                # skip sentence with all zero value eye-tracking features
                feature_sum = np.sum(eye_matrix, axis=0)
                feature_all_zero = False
                feature_zero_index = None
                for f_i in range(feature_sum.shape[0]):
                    if feature_sum[f_i] == 0:
                        feature_all_zero = True
                        feature_zero_index = f_i
                        break

                if feature_all_zero:
                    print(f"Calculate the mean and sd of eye_matrixs: Sentence-{idx}, split-{j} is ignored due to all zero in feature-{feature_zero_index}")
                    continue
                
                eye_matrixs = merge_eye_matrix(eye_matrix, eye_matrixs)

        return np.mean(eye_matrixs,axis=0), np.std(eye_matrixs, axis=0)

    def get_sentence_splited(self, idx:int):

        words = self.roi_data_unique[self.roi_data_unique.Sentence_ID == idx].sort_values(by='Word_Order').Words
        
        sentence_splited = []
        for word in words:
            sentence_splited.append(str(word))
            # sentence_splited.append(word)
        
        src_sentence = self.get_sentence(idx)
        split_sentence = "".join(sentence_splited)

        possible_num = 1
        sentence_splited = [sentence_splited]

        # if idx == 2887:
        #     print(src_sentence)
        #     print(split_sentence)

        # miss a dot at the end of sentence
        if split_sentence in src_sentence and len(src_sentence) == len(split_sentence) + 1 and src_sentence[-1] == '。':
            # print(f"index:{idx} append 。")
            sentence_splited[0].append('。')
        elif src_sentence != split_sentence:
            # multiple splits
            sentence_splited = []
            possible_num = 0

            all_experiments = self.roi_data[self.roi_data.Sentence_ID == idx]
            
            sentence_experiments = all_experiments.Experiment.drop_duplicates()

            for v in sentence_experiments:
                current_split = []
                words = all_experiments[all_experiments.Experiment == v].sort_values(by='Word_Order').Words
                for word in words:
                    current_split.append(word)
                
                split_sentence = "".join(current_split)

                if split_sentence in src_sentence and len(src_sentence) == len(split_sentence) + 1 and src_sentence[-1] == '。':
                    current_split.append('。')
                
                sentence_splited.append(current_split)

                possible_num += 1

                # print(f"src_sentence:{src_sentence},experiment:{v},split:{current_split}")

        return possible_num, sentence_splited
    
    def get_vocabulary(self):
        if self.vocab is not None:
            return self.vocab

        self.vocab = self.mean_features.index.values.tolist()
        # All mean values in '舟' and '舞厅' are 0, thus we remove it
        self.vocab.remove('舟')
        self.vocab.remove('舞厅')
        return self.vocab

    def get_features(self,word:str,features=None):
        if features is None:
            return self.mean_features.loc[word]

        # select features defined to return
        return self.mean_features.loc[word].filter(features)

    def get_feature_names(self):
        return self.mean_features.columns.values

    def get_word_dict(self) -> dict:
        if self.word2idx is not None and self.idx2word is not None:
            return self.word2idx, self.idx2word

        vocab = self.get_vocabulary()
        self.word2idx = {w:idx for idx,w in enumerate(vocab)}
        self.idx2word = {idx:w for idx,w in enumerate(vocab)}
        return self.word2idx, self.idx2word

    def get_sentence_dict(self) -> dict:
        # {1:{'Sentence':'月蓉和学诚喜欢吃剁椒鱼头，昨天聚餐月蓉点了这道菜。'},...,sentence_id:{'Sentence':'sentence_str'},...,}
        return self.sentence_data.to_dict('index')

    def get_rsm(self, vocab=None, features=None):
        vocab = vocab if vocab is not None else self.get_vocabulary()
        num_words = len(vocab)

        rsm = np.zeros((num_words,num_words))

        feature_dict = {}

        for i in range(num_words):
            w_i = vocab[i]

            feature_i = None
            if w_i in feature_dict:
                feature_i = feature_dict[w_i]
            else:
                feature_i = np.nan_to_num(self.get_features(w_i,features=features).values)
                # add feature for speed up and decrease the times of looking up
                feature_dict[w_i] = feature_i

            for j in range(i,num_words):
                w_j = vocab[j]

                feature_j = None
                if w_j in feature_dict:
                    feature_j = feature_dict[w_j]
                else:
                    feature_j = np.nan_to_num(self.get_features(w_j,features=features).values)
                    # cache feature for speed up and decrease the times of looking up
                    feature_dict[w_j] = feature_j

                pearson_similarity_ij = pearsonr(feature_i,feature_j)
                rsm[i][j] = pearson_similarity_ij[0]
                # reduce calculation due to the symmetric of pearson similarity
                rsm[j][i] = pearson_similarity_ij[0]
        
        return vocab, feature_dict, rsm


def test_split():
    eye_tracking_features = EyeTrackingFeatures()
    sentence_dict = eye_tracking_features.get_sentence_dict()

    for k,v in sentence_dict.items():
        src_sentence = v["Sentence"]

        num_split, split_words_list = eye_tracking_features.get_sentence_splited(k)

        for i in range(num_split):
            split_sentence = "".join(split_words_list[i])
            
            if split_sentence in src_sentence and len(src_sentence) == len(split_sentence) + 1 and src_sentence[-1] == '。':
                continue

            if src_sentence != split_sentence:
                print(f"idx:{k} exits difference: src_sentence:{src_sentence}, split_idx:{i}, split_sentence:{split_sentence}")

def test_word_features():
    eye_tracking_features = EyeTrackingFeatures()
    vocab = eye_tracking_features.get_vocabulary()
    print(vocab)

    word2idx,idx2word = eye_tracking_features.get_word_dict()

    print(word2idx)
    print(idx2word)

    word_feature = eye_tracking_features.get_features(word="喜欢")
    word_feature = np.array(word_feature.values)
    print(word_feature)


def convert2matrix(vocab, word_representation):
    w_num = len(vocab)
    feature_matrix = np.zeros((w_num, word_representation[vocab[0]].shape[0]))

    for w_i in range(w_num):
        feature_matrix[w_i] = word_representation[vocab[w_i]]
    
    return feature_matrix

def test_rsm():
    eye_tracking_features = EyeTrackingFeatures()

    start_time = time.time()

    vocab, word_representation, eye_tracking_features_rsm = eye_tracking_features.get_rsm()

    end_time = time.time()

    # Time cost 3147s
    print(f"Time cost {end_time-start_time}s in calculation of RSM of eye-tracking features.")
    print(f"vocab:{vocab}")
    print(f"eye_tracking_features_rsm:{eye_tracking_features_rsm}")

    eye_feature_matrix = convert2matrix(vocab, word_representation=word_representation)
    eye_matrix_path = f"../data/word_representation/word_eye_matrix.npy"
    with open(eye_matrix_path,"wb") as f:
        np.save(f,eye_feature_matrix)

    word_output_path = "../data/word_representation/word_eye_tracking.npy"
    with open(word_output_path,"wb") as f:
        np.save(f,word_representation)

    rsm_path = "../data/rsm/rsm_eye_tracking.npy"
    with open(rsm_path,"wb") as f:
        np.save(f,eye_tracking_features_rsm)