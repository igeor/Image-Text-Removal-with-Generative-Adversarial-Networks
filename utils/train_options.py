from argparse import ArgumentParser

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Inpainting Text Instances')

    # Data parameters
    parser.add_argument('--ckpt_save_path', help='checkpoint save path', default='/home/igeorvasilis/thesis_src/checkpoints')
    parser.add_argument('--train_imgs_save_path', help='checkpoint save path', default='/home/igeorvasilis/sdb/out_train')

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0002, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=200, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l1', type=str)


    return parser.parse_args()