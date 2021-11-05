import os
import parser
import argparse
import itertools


# modal_list = ['text', 'audio', 'video', 'text_audio', 'text_video', 'audio_video', 'all']
batch_size_list = list(range(200, 6000, 200))

comb = itertools.product(batch_size_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store_path', default='static_dict', type=str)
    parser.add_argument('--record_path', default='tmp.txt', type=str)
    parser.add_argument('--device', default=7, type=str)
    args = parser.parse_args()
    if not os.path.exists(args.store_path):
        os.mkdir(args.store_path)
    record_file = open(args.record_path, 'w')
    record_file.close()
    for idx, params in enumerate(list(comb)):
        batch_size = params[0]
        store_path = f'{args.store_path}/version_{idx}'
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        os.system(f'python train.py --store_path {store_path} --record_path {args.record_path}' +\
        f' --device {args.device} --modal audio_video --batch_size {batch_size} > \'{store_path}/train.out\'')


if __name__ == '__main__':
    main()

