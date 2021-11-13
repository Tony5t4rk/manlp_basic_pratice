import os
import pickle
import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--feature_path', default='./feature', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.feature_path):
        os.mkdir(args.feature_path)

    _, speaker, label, text, audio, visual, _, train_vid, test_vid = pickle.load(open(f'{args.data_path}/IEMOCAP_features_raw.pkl', 'rb'), encoding='latin1')
    train_vid, test_vid = [list(_) for _ in [train_vid, test_vid]]

    train_text = [np.array(text[vid]) for vid in train_vid]
    train_audio = [np.array(audio[vid]) for vid in train_vid]
    train_visual = [np.array(visual[vid]) for vid in train_vid]
    train_speaker = [np.array([[1, 0] if _ == 'M' else [0, 1] for _ in speaker[vid]]) for vid in train_vid]
    train_mask = [np.array([1] * len(label[vid])) for vid in train_vid]
    train_label = [np.array(label[vid]) for vid in train_vid]
    train_data = (train_text, train_audio, train_visual, train_speaker, train_mask, train_label)

    test_text = [np.array(text[vid]) for vid in test_vid]
    test_audio = [np.array(audio[vid]) for vid in test_vid]
    test_visual = [np.array(visual[vid]) for vid in test_vid]
    test_speaker = [np.array([[1, 0] if _ == 'M' else [0, 1] for _ in speaker[vid]]) for vid in test_vid]
    test_mask = [np.array([1] * len(label[vid])) for vid in test_vid]
    test_label = [np.array(label[vid]) for vid in test_vid]
    test_data = (test_text, test_audio, test_visual, test_speaker, test_mask, test_label)

    pickle.dump(train_data, open(f'{args.feature_path}/train.data', 'wb'))
    pickle.dump(test_data, open(f'{args.feature_path}/test.data', 'wb'))

