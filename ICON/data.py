import os
import argparse

import pickle
import numpy as np
from sklearn import model_selection, metrics


def load_data(args):
    transcripts, labels, own_history_id, other_history_id, own_history_id_rank, other_history_id_rank = pickle.load(open(f'{args.data_path}/dataset.pkl','rb'), encoding='latin1')

    label_idx = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}

    train_id = pickle.load(open(f'{args.data_path}/trainID.pkl', 'rb'), encoding='latin1')
    test_id = pickle.load(open(f'{args.data_path}/testID.pkl', 'rb'), encoding='latin1')
    val_id, _ = model_selection.train_test_split(test_id, test_size=.4, random_state=args.seed)

    # Labels
    train_label = np.array([label_idx[labels[_]] for _ in train_id])
    val_label = np.array([label_idx[labels[_]] for _ in val_id])
    test_label = np.array([label_idx[labels[_]] for _ in test_id])

    # Text features
    text_transcripts_emb, text_own_history_emb, text_other_history_emb = pickle.load(open(f'{args.data_path}/text/IEMOCAP_text_embeddings.pickle', 'rb'), encoding='latin1')
    if args.context:
        text_emb_context = pickle.load(open(f'{args.data_path}/text/IEMOCAP_text_context.pickle', 'rb'), encoding='latin1')
        for k in text_transcripts_emb.keys():
            if k in text_emb_context.keys():
                text_transcripts_emb[k] = text_emb_context[k]
        for k in text_own_history_emb.keys():
            ids = own_history_id[k]
            for idx, _id in enumerate(ids):
                if _id in text_emb_context.keys():
                    text_own_history_emb[k][idx] = text_emb_context[_id]
        for k in text_other_history_emb.keys():
            ids = other_history_id[k]
            for idx, _id in enumerate(ids):
                if _id in text_emb_context.keys():
                    text_other_history_emb[k][idx] = text_emb_context[_id]
    
    # Audio features
    audio_emb = pickle.load(open(f'{args.data_path}/audio/IEMOCAP_audio_features.pickle', 'rb'), encoding='latin1')
    if args.context:
        audio_emb_context = pickle.load(open(f'{args.data_path}/audio/IEMOCAP_audio_context.pickle', 'rb'), encoding='latin1')
        for _id in audio_emb.keys():
            if _id in audio_emb_context.keys():
                audio_emb[_id] = audio_emb_context[_id]
    
    # Video features
    video_emb = pickle.load(open(f'{args.data_path}/video/IEMOCAP_video_features.pickle', 'rb'), encoding='latin1')
    if args.context:
        video_emb_context = pickle.load(open(f'{args.data_path}/video/IEMOCAP_video_context.pickle', 'rb'), encoding='latin1')
        for _id in video_emb.keys():
            if _id in audio_emb_context.keys():
                video_emb[_id] = video_emb_context[_id]
    
    text_train_query = np.array([text_transcripts_emb[_] for _ in train_id])  # (5810, 100)
    text_val_query = np.array([text_transcripts_emb[_] for _ in val_id])  # (973, 100)
    text_test_query = np.array([text_transcripts_emb[_] for _ in test_id])  # (1623, 100)

    audio_train_query = np.array([audio_emb[_] for _ in train_id])  # (5810, 100)
    audio_val_query = np.array([audio_emb[_] for _ in val_id])  # (973, 100)
    audio_test_query = np.array([audio_emb[_] for _ in test_id])  # (1623, 100)

    video_train_query = np.array([video_emb[_] for _ in train_id])  # (5810, 512)
    video_val_query = np.array([video_emb[_] for _ in val_id])  # (973, 512)
    video_test_query = np.array([video_emb[_] for _ in test_id])  # (1623, 512)
    
    if args.modal == 'text':
        train_query = text_train_query
        val_query = text_val_query
        test_query = text_test_query
    elif args.modal == 'audio':
        train_query = audio_train_query
        val_query = audio_val_query
        test_query = audio_test_query
    elif args.modal == 'video':
        train_query = video_train_query
        val_query = video_val_query
        test_query = video_test_query
    elif args.modal == 'text_audio':
        train_query = np.concatenate((text_train_query, audio_train_query), axis=-1)
        val_query = np.concatenate((text_val_query, audio_val_query), axis=-1)
        test_query = np.concatenate((text_test_query, audio_test_query), axis=-1)
    elif args.modal == 'text_video':
        train_query = np.concatenate((text_train_query, video_train_query), axis=-1)
        val_query = np.concatenate((text_val_query, video_val_query), axis=-1)
        test_query = np.concatenate((text_test_query, video_test_query), axis=-1)
    elif args.modal == 'audio_video':
        train_query = np.concatenate((audio_train_query, video_train_query), axis=-1)
        val_query = np.concatenate((audio_val_query, video_val_query), axis=-1)
        test_query = np.concatenate((audio_test_query, video_test_query), axis=-1)
    elif args.modal == 'all':
        train_query = np.concatenate((text_train_query, audio_train_query, video_train_query), axis=-1)
        val_query = np.concatenate((text_val_query, audio_val_query, video_val_query), axis=-1)
        test_query = np.concatenate((text_test_query, audio_test_query, video_test_query), axis=-1)
    
    # Pad the history
    train_own_history = np.zeros((len(train_id), args.time_step, train_query.shape[1]), dtype=np.float32)
    train_other_history = np.zeros((len(train_id), args.time_step, train_query.shape[1]), dtype=np.float32)
    train_own_history_mask = np.zeros((len(train_id), args.time_step), dtype=bool)
    train_other_history_mask = np.zeros((len(train_id), args.time_step), dtype=bool)

    for _idx, _id in enumerate(train_id):
        merge_history_id_rank = own_history_id_rank[_id] + other_history_id_rank[_id]
        if len(merge_history_id_rank) > 0:
            max_rank = np.max(merge_history_id_rank)
            own_history_rank = [max_rank - cur_rank for cur_rank in own_history_id_rank[_id]]
            other_history_rank = [max_rank - cur_rank for cur_rank in other_history_id_rank[_id]]

            _text_own_history_emb = np.array(text_own_history_emb[_id])
            _text_other_history_emb = np.array(text_other_history_emb[_id])

            _audio_own_history_emb = np.array([audio_emb[own_history_id[_id][_]] for _ in range(len(own_history_id[_id]))])
            _audio_other_history_emb = np.array([audio_emb[other_history_id[_id][_]] for _ in range(len(other_history_id[_id]))])

            _video_own_history_emb = np.array([video_emb[own_history_id[_id][_]] for _ in range(len(own_history_id[_id]))])
            _video_other_history_emb = np.array([video_emb[other_history_id[_id][_]] for _ in range(len(other_history_id[_id]))])

            for idx, rank in enumerate(own_history_rank):
                if rank >= args.time_step:
                    continue
                if args.modal == 'text':
                    train_own_history[_idx, rank] = _text_own_history_emb[idx]
                elif args.modal == 'audio':
                    train_own_history[_idx, rank] = _audio_own_history_emb[idx]
                elif args.modal == 'video':
                    train_own_history[_idx, rank] = _video_own_history_emb[idx]
                elif args.modal == 'text_audio':
                    train_own_history[_idx, rank] = np.concatenate((_text_own_history_emb[idx], _audio_own_history_emb[idx]))
                elif args.modal == 'text_video':
                    train_own_history[_idx, rank] = np.concatenate((_text_own_history_emb[idx], _video_own_history_emb[idx]))
                elif args.modal == 'audio_video':
                    train_own_history[_idx, rank] = np.concatenate((_audio_own_history_emb[idx], _video_own_history_emb[idx]))
                elif args.modal == 'all':
                    train_own_history[_idx, rank] = np.concatenate((_text_own_history_emb[idx], _audio_own_history_emb[idx], _video_own_history_emb[idx]))
                train_own_history_mask[_idx, rank] = True
            train_own_history[_idx] = train_own_history[_idx, ::-1, :]
            train_own_history_mask[_idx] = train_own_history_mask[_idx, ::-1]

            for idx, rank in enumerate(other_history_rank):
                if rank >= args.time_step:
                    continue
                if args.modal == 'text':
                    train_other_history[_idx, rank] = _text_other_history_emb[idx]
                elif args.modal == 'audio':
                    train_other_history[_idx, rank] = _audio_other_history_emb[idx]
                elif args.modal == 'video':
                    train_other_history[_idx, rank] = _video_other_history_emb[idx]
                elif args.modal == 'text_audio':
                    train_other_history[_idx, rank] = np.concatenate((_text_other_history_emb[idx], _audio_other_history_emb[idx]))
                elif args.modal == 'text_video':
                    train_other_history[_idx, rank] = np.concatenate((_text_other_history_emb[idx], _video_other_history_emb[idx]))
                elif args.modal == 'audio_video':
                    train_other_history[_idx, rank] = np.concatenate((_audio_other_history_emb[idx], _video_other_history_emb[idx]))
                elif args.modal == 'all':
                    train_other_history[_idx, rank] = np.concatenate((_text_other_history_emb[idx], _audio_other_history_emb[idx], _video_other_history_emb[idx]))
                train_other_history_mask[_idx, rank] = True
            train_other_history[_idx] = train_other_history[_idx, ::-1, :]
            train_other_history_mask[_idx] = train_other_history_mask[_idx, ::-1]

    val_own_history = np.zeros((len(val_id), args.time_step, val_query.shape[1]), dtype=np.float32)
    val_other_history = np.zeros((len(val_id), args.time_step, val_query.shape[1]), dtype=np.float32)
    val_own_history_mask = np.zeros((len(val_id), args.time_step), dtype=bool)
    val_other_history_mask = np.zeros((len(val_id), args.time_step), dtype=bool)

    for _idx, _id in enumerate(val_id):
        merge_history_id_rank = own_history_id_rank[_id] + other_history_id_rank[_id]
        if len(merge_history_id_rank) > 0:
            max_rank = np.max(merge_history_id_rank)
            own_history_rank = [max_rank - cur_rank for cur_rank in own_history_id_rank[_id]]
            other_history_rank = [max_rank - cur_rank for cur_rank in other_history_id_rank[_id]]

            _text_own_history_emb = np.array(text_own_history_emb[_id])
            _text_other_history_emb = np.array(text_other_history_emb[_id])

            _audio_own_history_emb = np.array([audio_emb[own_history_id[_id][_]] for _ in range(len(own_history_id[_id]))])
            _audio_other_history_emb = np.array([audio_emb[other_history_id[_id][_]] for _ in range(len(other_history_id[_id]))])

            _video_own_history_emb = np.array([video_emb[own_history_id[_id][_]] for _ in range(len(own_history_id[_id]))])
            _video_other_history_emb = np.array([video_emb[other_history_id[_id][_]] for _ in range(len(other_history_id[_id]))])

            for idx, rank in enumerate(own_history_rank):
                if rank >= args.time_step:
                    continue
                if args.modal == 'text':
                    val_own_history[_idx, rank] = _text_own_history_emb[idx]
                elif args.modal == 'audio':
                    val_own_history[_idx, rank] = _audio_own_history_emb[idx]
                elif args.modal == 'video':
                    val_own_history[_idx, rank] = _video_own_history_emb[idx]
                elif args.modal == 'text_audio':
                    val_own_history[_idx, rank] = np.concatenate((_text_own_history_emb[idx], _audio_own_history_emb[idx]))
                elif args.modal == 'text_video':
                    val_own_history[_idx, rank] = np.concatenate((_text_own_history_emb[idx], _video_own_history_emb[idx]))
                elif args.modal == 'audio_video':
                    val_own_history[_idx, rank] = np.concatenate((_audio_own_history_emb[idx], _video_own_history_emb[idx]))
                elif args.modal == 'all':
                    val_own_history[_idx, rank] = np.concatenate((_text_own_history_emb[idx], _audio_own_history_emb[idx], _video_own_history_emb[idx]))
                val_own_history_mask[_idx, rank] = True
            val_own_history[_idx] = val_own_history[_idx, ::-1, :]
            val_own_history_mask[_idx] = val_own_history_mask[_idx, ::-1]

            for idx, rank in enumerate(other_history_rank):
                if rank >= args.time_step:
                    continue
                if args.modal == 'text':
                    val_other_history[_idx, rank] = _text_other_history_emb[idx]
                elif args.modal == 'audio':
                    val_other_history[_idx, rank] = _audio_other_history_emb[idx]
                elif args.modal == 'video':
                    val_other_history[_idx, rank] = _video_other_history_emb[idx]
                elif args.modal == 'text_audio':
                    val_other_history[_idx, rank] = np.concatenate((_text_other_history_emb[idx], _audio_other_history_emb[idx]))
                elif args.modal == 'text_video':
                    val_other_history[_idx, rank] = np.concatenate((_text_other_history_emb[idx], _video_other_history_emb[idx]))
                elif args.modal == 'audio_video':
                    val_other_history[_idx, rank] = np.concatenate((_audio_other_history_emb[idx], _video_other_history_emb[idx]))
                elif args.modal == 'all':
                    val_other_history[_idx, rank] = np.concatenate((_text_other_history_emb[idx], _audio_other_history_emb[idx], _video_other_history_emb[idx]))
                val_other_history_mask[_idx, rank] = True
            val_other_history[_idx] = val_other_history[_idx, ::-1, :]
            val_other_history_mask[_idx] = val_other_history_mask[_idx, ::-1]

    test_own_history = np.zeros((len(test_id), args.time_step, test_query.shape[1]), dtype=np.float32)
    test_other_history = np.zeros((len(test_id), args.time_step, test_query.shape[1]), dtype=np.float32)
    test_own_history_mask = np.zeros((len(test_id), args.time_step), dtype=bool)
    test_other_history_mask = np.zeros((len(test_id), args.time_step), dtype=bool)

    for _idx, _id in enumerate(test_id):
        merge_history_id_rank = own_history_id_rank[_id] + other_history_id_rank[_id]
        if len(merge_history_id_rank) > 0:
            max_rank = np.max(merge_history_id_rank)
            own_history_rank = [max_rank - cur_rank for cur_rank in own_history_id_rank[_id]]
            other_history_rank = [max_rank - cur_rank for cur_rank in other_history_id_rank[_id]]

            _text_own_history_emb = np.array(text_own_history_emb[_id])
            _text_other_history_emb = np.array(text_other_history_emb[_id])

            _audio_own_history_emb = np.array([audio_emb[own_history_id[_id][_]] for _ in range(len(own_history_id[_id]))])
            _audio_other_history_emb = np.array([audio_emb[other_history_id[_id][_]] for _ in range(len(other_history_id[_id]))])

            _video_own_history_emb = np.array([video_emb[own_history_id[_id][_]] for _ in range(len(own_history_id[_id]))])
            _video_other_history_emb = np.array([video_emb[other_history_id[_id][_]] for _ in range(len(other_history_id[_id]))])

            for idx, rank in enumerate(own_history_rank):
                if rank >= args.time_step:
                    continue
                if args.modal == 'text':
                    test_own_history[_idx, rank] = _text_own_history_emb[idx]
                elif args.modal == 'audio':
                    test_own_history[_idx, rank] = _audio_own_history_emb[idx]
                elif args.modal == 'video':
                    test_own_history[_idx, rank] = _video_own_history_emb[idx]
                elif args.modal == 'text_audio':
                    test_own_history[_idx, rank] = np.concatenate((_text_own_history_emb[idx], _audio_own_history_emb[idx]))
                elif args.modal == 'text_video':
                    test_own_history[_idx, rank] = np.concatenate((_text_own_history_emb[idx], _video_own_history_emb[idx]))
                elif args.modal == 'audio_video':
                    test_own_history[_idx, rank] = np.concatenate((_audio_own_history_emb[idx], _video_own_history_emb[idx]))
                elif args.modal == 'all':
                    test_own_history[_idx, rank] = np.concatenate((_text_own_history_emb[idx], _audio_own_history_emb[idx], _video_own_history_emb[idx]))
                test_own_history_mask[_idx, rank] = True
            test_own_history[_idx] = test_own_history[_idx, ::-1, :]
            test_own_history_mask[_idx] = test_own_history_mask[_idx, ::-1]

            for idx, rank in enumerate(other_history_rank):
                if rank >= args.time_step:
                    continue
                if args.modal == 'text':
                    test_other_history[_idx, rank] = _text_other_history_emb[idx]
                elif args.modal == 'audio':
                    test_other_history[_idx, rank] = _audio_other_history_emb[idx]
                elif args.modal == 'video':
                    test_other_history[_idx, rank] = _video_other_history_emb[idx]
                elif args.modal == 'text_audio':
                    test_other_history[_idx, rank] = np.concatenate((_text_other_history_emb[idx], _audio_other_history_emb[idx]))
                elif args.modal == 'text_video':
                    test_other_history[_idx, rank] = np.concatenate((_text_other_history_emb[idx], _video_other_history_emb[idx]))
                elif args.modal == 'audio_video':
                    test_other_history[_idx, rank] = np.concatenate((_audio_other_history_emb[idx], _video_other_history_emb[idx]))
                elif args.modal == 'all':
                    test_other_history[_idx, rank] = np.concatenate((_text_other_history_emb[idx], _audio_other_history_emb[idx], _video_other_history_emb[idx]))
                test_other_history_mask[_idx, rank] = True
            test_other_history[_idx] = test_other_history[_idx, ::-1, :]
            test_other_history_mask[_idx] = test_other_history_mask[_idx, ::-1]
    
    # Save features
    train_data = [train_query, train_own_history, train_other_history, train_own_history_mask, train_other_history_mask, train_label]
    val_data = [val_query, val_own_history, val_other_history, val_own_history_mask, val_other_history_mask, val_label]
    test_data = [test_query, test_own_history, test_other_history, test_own_history_mask, test_other_history_mask, test_label]
    if not os.path.exists(f'{args.feature_path}/{args.modal}'):
        os.mkdir(f'{args.feature_path}/{args.modal}')
    pickle.dump(train_data, open(f'{args.feature_path}/{args.modal}/train.data', 'wb'))
    pickle.dump(val_data, open(f'{args.feature_path}/{args.modal}/val.data', 'wb'))
    pickle.dump(test_data, open(f'{args.feature_path}/{args.modal}/test.data', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--feature_path', default='./feature', type=str)
    parser.add_argument('--seed', default=1227, type=int)
    parser.add_argument('--context', default=True, type=bool)
    parser.add_argument('--time_step', default=40, type=int)
    parser.add_argument('--modal', default=None, type=str)
    args = parser.parse_args()
    if not os.path.exists(args.feature_path):
        os.mkdir(args.feature_path)

    if args.modal:
        load_data(args)
    else:
        for modal in ['text', 'audio', 'video', 'text_audio', 'text_video', 'audio_video', 'all']:
            args.modal = modal
            load_data(args)

