from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--bs", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resolution", type=tuple, default=(224, 224))
parser.add_argument("--activation", type=str, default='selu')
parser.add_argument("--loss_type", type=str, default='DMON')
parser.add_argument("--process", type=str, default='DINO')
parser.add_argument("--dataset", type=str, default='CUB')

args = parser.parse_args()

if __name__ == '__main__':
    seg = Segmentation(args.process, args.bs, args.epochs, args.resolution, args.activation, args.loss_type)
    ds = Dataset(args.dataset)

    total_iou = 0
    total_samples = 0
    while ds.size > 0:
        for img, mask in ds.load_samples():
            try: 
                iou, _, _ = seg.segment(img, mask)
                total_iou += iou
                total_samples += 1
            except Exception as e:
                print(e)
                continue
    
    print(f'mIoU: {total_iou / total_samples}')

