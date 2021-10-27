"""
Generate Raw image bottleneck feature using resnet50 pretrained model

"""
import torch
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
from src.utils.utils import chunks
from torchvision import datasets, models, transforms


def resnet_infer(model, inputs):
    """
    resnet inference
    :param model: resnet torch mdoel
    :param inputs: torch tensor of the images inputs, already preprocessed
    :return: embedding features
    """
    with torch.no_grad():
        x = inputs.cuda()
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        out = torch.flatten(x, 1)

    out = out.cpu().numpy()
    return out


def generate_images_features(data_list, method='resnet'):
    """
    load and embed images
    :param data_list:     list of dictionary, each entry contains the key image_path
    :param method:        pretrained model to use, currently only support resnet
    :return: same dictionary with additional keyed value img_emb
    """

    # load model
    model = models.resnet50(pretrained=True)
    model = model.cuda().eval()

    # create resnet image processing pipeline
    input_size = 224
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    available_methods = {
        'resnet': [transform, resnet_infer]
    }
    transform, infer = available_methods[method]

    batches = list(chunks(data_list, 32))
    bad_inputs = []
    for batch in tqdm(batches):

        batch = [x for x in batch if x['image_path'] != 'NO_IMAGE']

        if len(batch) > 0:
            inputs = []
            good_batch = []
            for x in batch:
                try:
                    inputs.append(transform(Image.open(x['image_path']).convert('RGB')))
                    good_batch.append(x)
                except:
                    bad_inputs.append(x)
            if len(inputs) > 0:
                inputs = torch.stack(inputs)

                out = infer(model, inputs)
                for data, emb in zip(good_batch, out):
                    data['img_emb'] = emb

    return data_list


def generate_images_features_file(data_path, output_path, method='resnet'):
    """

    :param data_path: path of data file, data consists of list of dict, each entry contains key image_path
    :param output_path: output path
    :param method:  pretrained model type, only support resnet currently
    :return:
    """
    # load data
    data_list = pickle.load(open(data_path, 'rb'))

    # generate embedding
    data_list = generate_images_features(data_list, method)

    # save data
    pickle.dump(data_list, open(output_path, 'wb'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="input data path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="output data path"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    generate_images_features_file(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
