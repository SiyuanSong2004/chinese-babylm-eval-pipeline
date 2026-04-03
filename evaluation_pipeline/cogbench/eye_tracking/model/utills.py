import json
import joblib as jlb # parallelizing
import time
import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch
import multiprocess as mp
from gensim import models

def cos_sim(v_i,v_j):
    return np.dot(v_i,v_j)/(np.linalg.norm(v_i)*np.linalg.norm(v_j))

def spearman_sim(v_i,v_j):
    return spearmanr(v_i,v_j)[0]

def pearson_sim(v_i,v_j):
    return pearsonr(v_i,v_j)[0]

def standardize_matrix(feature_matrix, mean=None, std=None):
    mean = np.nanmean(feature_matrix, axis=0) if mean is None else mean
    std = np.nanstd(feature_matrix, axis=0) if std is None else std
    return (feature_matrix - mean)/ std

def merge_layer_output(sub_dict, total_dict=None):
    if total_dict is None:
        return sub_dict

    for k,v in total_dict.items():
        total_dict[k].extend(sub_dict[k])
    return total_dict

def merge_eye_matrix(sub_matrix, total_matrix=None):
    if total_matrix is None:
        return sub_matrix

    return np.concatenate([total_matrix,sub_matrix], axis=0)

def is_sign(c):
    sign_list = [',','.','!','?','%',':',';','-']
    return (c in sign_list)

def get_word2vec(vec_path:str, skip_first_line=True):
    words = []
    idx = 0
    word2idx = {}
    vectors = []

    with open(vec_path, 'rb') as f:
        for l in f:
            if skip_first_line:
                skip_first_line = False
                continue
            word, vec = l.decode().split(' ',1)
            words.append(word)
            word2idx[word] = idx
            idx += 1
            # print(idx)
            vect = np.fromstring(vec, sep=' ')
            vectors.append(vect)

    return vectors,words,word2idx

def get_cbow_vec(vec_path:str, vocab=None):
    word_vec_ret = {}
    w_vocab = set()

    word_vec = models.KeyedVectors.load_word2vec_format(vec_path, binary=True)
    if vocab is None:
        return word_vec, set(word_vec.index_to_key)

    for word in list(word_vec.index_to_key):
        if word in vocab:
            word_vec_ret[word] = word_vec[word]
            w_vocab.add(word)

    print(f'Found {len(word_vec_ret)} words with word vectors, out of {len(word2id)} words')
    return word_vec_ret, w_vocab

def refine_word_dict(vocab, word2idx, wordvecs):
    w_dict = {}
    for word in vocab:
        if word in w_dict:
            raise ValueError(f"Word:{word} appears two times in vocab.")
        else:
            w_dict[word] = wordvecs[word2idx[word]]
    return w_dict

def get_eye_features_matrix(split_feature, valid_num, valid_index, features=['FFD', 'GD', 'FPF', 'FN', 'RI', 'RO', 'LI_left', 'LI_right', 'TT']):
    feature_num = len(features)
    eye_matrix = np.zeros((valid_num, feature_num))
    current_row = 0
    for w_i in range(len(valid_index)):
        if valid_index[w_i]:
            for f_idx, f_n in enumerate(features):
                eye_matrix[current_row][f_idx] = split_feature[w_i][f_n]
            current_row += 1
    return eye_matrix

def get_zuco_eye_features_matrix(split_feature, valid_num, valid_index, features=['FFD', 'TRT', 'GD', 'GPT', 'SFD', 'nFixations']):
    feature_num = len(features)
    eye_matrix = np.zeros((valid_num, feature_num))
    current_row = 0
    feature_names = ['FFD', 'TRT', 'GD', 'GPT', 'SFD', 'nFixations', 'meanPupilSize']
    for w_i in range(len(valid_index)):
        if valid_index[w_i]:
            mean_features = np.nanmean(np.array(split_feature[w_i]), axis=0)
            for f_idx, f_n in enumerate(features):
                eye_matrix[current_row][f_idx] = mean_features[feature_names.index(f_n)]
            current_row += 1
    return eye_matrix

def find_valid_words(sentence_split):
    num_words = len(sentence_split)
    
    valid_index = [True for i in range(num_words)]

    left_ignore_char = 0
    idx = 0
    while left_ignore_char<3:
        valid_index[idx] = False
        left_ignore_char += len(sentence_split[idx])
        idx += 1
    
    right_ignore_char = 0
    idx = num_words-1
    while right_ignore_char < 3:
        # ignore the last dot
        if sentence_split[idx] == '。':
            valid_index[idx] = False
            idx -= 1

        valid_index[idx] = False
        right_ignore_char += len(sentence_split[idx])
        idx -= 1
    
    return valid_index

def find_no_nan_words(sentence_split, split_feature, valid_index=None, features=['FFD', 'TRT', 'GD', 'GPT', 'SFD', 'nFixations']):
    num_words = len(sentence_split)
    valid_index = [True for i in range(num_words)] if valid_index is None else valid_index
    valid_num = 0

    feature_names = ['FFD', 'TRT', 'GD', 'GPT', 'SFD', 'nFixations', 'meanPupilSize']
    feature_index = [feature_names.index(f) for f in features]
    for w_i in range(len(valid_index)):
        if valid_index[w_i]:
            # print(split_feature[w_i])
            mean_features = np.nanmean(np.array(split_feature[w_i]), axis=0)[feature_index]
            # set it is invalid if np.nan exists in mean value
            if len(np.argwhere(np.isnan(mean_features))) > 0:
                valid_index[w_i] = False
                continue
            valid_num += 1

    return valid_num, valid_index


def find_vocab_word(sentence_split,vocab=None,valid_index=None):
    num_words = len(sentence_split)
    valid_index = [True for i in range(num_words)] if valid_index is None else valid_index
    valid_num = 0
    if vocab is None:
        return sum(valid_index), valid_index

    for w_idx, w in enumerate(sentence_split):
        if valid_index[w_idx]:
            if (w in vocab) or (is_sign(w[-1]) and w[:-1] in vocab):
                valid_num += 1
                continue

        valid_index[w_idx] = False

    return valid_num, valid_index

def calculate_rsm_list(w_vectors, similarity_metric=pearson_sim):
    num_words = len(w_vectors)
    rsm_sent = np.zeros((num_words,num_words))

    for i in range(num_words):
        feature_i = w_vectors[i]

        for j in range(i,num_words):
            feature_j = w_vectors[j]

            # print(f"w_i:{w_i}({feature_i}) and w_j:{w_j}({feature_j})")
            try:
                similarity_ij = similarity_metric(feature_i,feature_j)
            except:
                print(f"w_i:{i}({feature_i}) and w_j:{j}({feature_j})")

            rsm_sent[i][j] = similarity_ij
            # reduce calculation due to the symmetric of pearson similarity
            rsm_sent[j][i] = similarity_ij

    return rsm_sent

def calculate_vec_similarity(idx):
    res = []
    for j in range(idx,parallel_len):
        res.append(parallel_similarity_metric(parallel_vecs[idx],parallel_vecs[j]))
    return (idx, res)

def deal_with_results(res):
    idx, row_res = res
    for j in range(idx, parallel_len):
        parallel_matrix[idx][j] = row_res[j-idx]
        parallel_matrix[j][idx] = row_res[j-idx]
    if idx % 5000 == 0:
        current_time = time.strftime("%Y/%m/%d, %H:%M:%S")
        print(f"{current_time}\tRow {idx} is done.")
    del idx
    del row_res
    del res

def calculate_rsm_list_parallel(w_vectors, similarity_metric=pearson_sim, num_thread=10):
    global parallel_similarity_metric
    parallel_similarity_metric = similarity_metric
    global parallel_vecs
    parallel_vecs = w_vectors
    global parallel_len
    num_words = len(w_vectors)
    parallel_len = num_words
    rsm_sent = np.zeros((num_words,num_words))
    global parallel_matrix
    parallel_matrix = rsm_sent
    p = mp.Pool(processes=num_thread)
    for i in range(num_words):
        p.apply_async(calculate_vec_similarity, args=(i,), callback=deal_with_results)
    p.close()
    p.join()

    del parallel_similarity_metric
    del parallel_vecs
    del parallel_len
    del parallel_matrix
    return rsm_sent


def calculate_similarity(model_rsm, feature_matrix, standardize=True, f_mean=None, f_std=None, similarity_metric=pearson_sim, similarity_axis="column"):
    w_num = model_rsm.shape[0]
    
    assert(model_rsm.shape[0] == feature_matrix.shape[0])

    if standardize:
        feature_matrix = standardize_matrix(feature_matrix, mean=f_mean, std=f_std)

    # weight_sum = np.sum((model_rsm - np.identity(w_num)), axis=1).reshape((w_num,1))
    # model_matrix = np.matmul((model_rsm - np.identity(w_num)), feature_matrix) / weight_sum

    # without normal
    model_matrix = np.matmul((model_rsm - np.identity(w_num)), feature_matrix)
    
    similarity_sum = 0
    similaritys = []
    
    if similarity_axis == "column":
        feature_num = feature_matrix.shape[1]
        for feature_i in range(feature_num):
            model_features = model_matrix[:,feature_i]
            eye_features = feature_matrix[:,feature_i]

            try:
                similarity_i = similarity_metric(model_features,eye_features)
            except:
                print(f"Error model_matrix:{model_matrix}")
                print(f"Error feature_matrix:{feature_matrix}")

            similarity_sum += similarity_i
            similaritys.append(similarity_i)
    else:
        # calculate similarity by row
        feature_num = feature_matrix.shape[0]
        for feature_i in range(feature_num):
            model_features = model_matrix[feature_i, :]
            eye_features = feature_matrix[feature_i, :]

            try:
                similarity_i = similarity_metric(model_features,eye_features)
            except:
                print(f"Error model_matrix:{model_matrix}")
                print(f"Error feature_matrix:{feature_matrix}")

            similarity_sum += similarity_i
            similaritys.append(similarity_i)
    
    average_similarity = similarity_sum/feature_num
    return average_similarity, similaritys


def calculate_word_output_sent(model_outputs:torch.tensor, split_words_list:list, output_index, valid_index):
    word_outputs = []
    num_words = 0

    for w_idx, w in enumerate(split_words_list):
        if not valid_index[w_idx]:
            continue

        output_shape = model_outputs.shape
        if len(output_shape) == 3 :
            word_average_output = torch.mean(model_outputs[0][output_index[w_idx]],dim=0).detach().cpu().numpy()
        elif len(output_shape) == 2:
            word_average_output = torch.mean(model_outputs[output_index[w_idx]],dim=0).detach().cpu().numpy()
        else:
            raise ValueError(f"Shape of output hidden vector is {output_shape}, which is not valid (2 or 3).")
        
        word_outputs.append(word_average_output)
        num_words += 1

    return num_words, word_outputs

def get_layer_similarity(word_vectors, eye_matrix, similarity_metric=pearson_sim, standardize=True, f_mean=None, f_std=None, similarity_axis="column"):
    layer_rsm = calculate_rsm_list(word_vectors,similarity_metric=similarity_metric)
    average_sim, similarities = calculate_similarity(layer_rsm, eye_matrix, standardize=standardize, f_mean=f_mean, f_std=f_std, similarity_axis=similarity_axis)
    return average_sim, similarities

def get_num_layers(model):
    # encoder model
    if not model.config.is_encoder_decoder:
        return model.config.num_hidden_layers

    # encoder_layers + decoder_layers
    num_layers = model.config.encoder_layers + model.config.decoder_layers
    return num_layers

# def test_split_valid():
#     # split = ["探险","队","这次","要","去","原始","沙漠","寻找","一种","远古","的","生物"]
#     # print(find_valid_words(split))

#     eye_tracking_features = EyeTrackingFeatures()
#     sentence_dict = eye_tracking_features.get_sentence_dict()

#     for k,v in sentence_dict.items():
#         num_split, split_words_list = eye_tracking_features.get_sentence_splited(k)

#         for i in range(num_split):
#             # split_sentence = "".join(split_words_list[i])
#             print(split_words_list[i])
#             print(find_valid_words(split_words_list[i]))

def check_split(s_1:list, s_2:list):
    if len(s_1) != len(s_2):
        return False

    for idx, w in enumerate(s_1):
        if s_2[idx] != w:
            return False
    
    return True

# def check_sent_level_split():
#     eye_tracking_features = EyeTrackingFeatures()
#     sentence_dict = eye_tracking_features.get_sentence_dict()
#     vocab = eye_tracking_features.get_vocabulary()

#     for k,v in sentence_dict.items():
#         num_split, split_words_list = eye_tracking_features.get_sentence_splited(k)
        
#         # check split
#         for i in range(num_split):
#             valid_words = find_valid_words(split_words_list[i])
#             valid_num = 0
#             for word_idx, word in enumerate(split_words_list[i]):
#                 # remove the first and last 3 characters and their words
#                 if not valid_words[word_idx]:
#                     continue
#                 if word in vocab:
#                     valid_num += 1
#             print(f"sentence-{k},split-{i}:valid word num is {valid_num}")
    

    # feature_dict_json = json.dumps(sent_level_info)
    # with open("eye_features_sentence_level_json","w") as f:
    #     f.write(feature_dict_json)

# for test the parallel function
def work_function(p1): 
    time.sleep(6-p1)   
    return p1

if __name__ == '__main__':
    # check_sent_level_split()
    s_time = time.time()

    res = jlb.Parallel(n_jobs=5)(jlb.delayed(work_function)(i) for i in range(5))

    e_time = time.time()
    print(res)
    print(f"Time cost:{e_time-s_time}")
