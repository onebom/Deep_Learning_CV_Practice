from utils.config import parse_args

from dataset.data_loader import get_data_loader
from models.stylegan import StyleGAN

def main(args):
    model = None
    model = StyleGAN(args)

    train_loader, test_loader = get_data_loader(args)

    if args.is_train=='True':
        model.train(train_loader)
    
    else:
        model.evaluate(test_loader, args.load_model)
        for i in range(10):
            model.generate_latent_walk(i)

if __name__ == '__main__':
    args=parse_args()
    print(args.cuda)
    main(args)