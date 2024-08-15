import os
import argparse

from solver import Solver
from data_loader import get_audio_loader

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

def main(config):
    assert config.mode in {'TRAIN', 'TEST', 'DEV', 'PREDICT'},\
        'invalid mode: "{}" not in ["TRAIN", "TEST", "DEV", "PREDICT"]'.format(config.mode)

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    if config.mode == 'TRAIN':
        data_loader = get_audio_loader(config.audio_path,
                                       config.subset,
                                        config.batch_size,
                                        tr_val='train',
                                        split=config.split)
        valid_loader = get_audio_loader(config.audio_path,
                                        config.subset,
                                        config.batch_size,   # 16 고정
                                        tr_val='validation',
                                        split=config.split)

        solver = Solver(data_loader, valid_loader, config)

        solver.train()

    elif config.mode == 'TEST':
        data_loader = get_audio_loader(config.audio_path,
                                       config.subset,
                                        config.batch_size,
                                        tr_val='test',
                                        split=config.split)

        solver = Solver(data_loader, None, config)

        solver.test()

    elif config.mode == 'PREDICT':
        pass

    elif config.mode == 'DEV':
        data_loader = get_audio_loader(config.audio_path,
                                       config.subset,
                                       config.batch_size,
                                       tr_val='train',
                                       split=config.split)

        solver = Solver(data_loader, None, config)
        solver.print_init()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mode', type=str, default='TEST')
    parser.add_argument('--model_save_path', type=str, default='./load_models')
    parser.add_argument('--audio_path', type=str, default='C:/Users/User/PycharmProjects/mtg-jamendo-cnn/data/melspecs_5')    # data path
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--subset', type=str, default='moodtheme')
    parser.add_argument('--model', type=str, default='Transformer')

    config = parser.parse_args()

    print(config)
    main(config)



