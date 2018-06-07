from __future__ import print_function
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

import sys
import time
from os import path
import skimage.measure
from os import listdir



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=2, help="super resolution upscale factor")

args = parser.parse_args()


TOTAL_BAR_LENGTH = 80
LAST_T = time.time()
BEGIN_T = LAST_T


def progress_bar(current, total, msg=None):
    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    current_len = int(TOTAL_BAR_LENGTH * (current + 1) / total)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    sys.stdout.write(' %d/%d' % (current + 1, total))
    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('-')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = '  Step: %s' % format_time(step_time)
    time_used += ' | Tot: %s' % format_time(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# return the formatted time
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds*1000)

    output = ''
    time_index = 1
    if days > 0:
        output += str(days) + 'D'
        time_index += 1
    if hours > 0 and time_index <= 2:
        output += str(hours) + 'h'
        time_index += 1
    if minutes > 0 and time_index <= 2:
        output += str(minutes) + 'm'
        time_index += 1
    if seconds_final > 0 and time_index <= 2:
        output += str(seconds_final) + 's'
        time_index += 1
    if millis > 0 and time_index <= 2:
        output += str(millis) + 'ms'
        time_index += 1
    if output == '':
        output = '0ms'
    return output



class Net(torch.nn.Module):

    def __init__(self, num_channels, upscale_factor, d=64, s=12, m=4):
        super(Net, self).__init__()

        # Feature extraction
        self.first_part = nn.Sequential(nn.Conv2d(in_channels=num_channels,
                                                  out_channels=d, kernel_size=5,
                                                  stride=1, padding=2), nn.PReLU())

        self.layers = []
        # Shrinking
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d,
                                                   out_channels=s,
                                                   kernel_size=1,
                                                   stride=1, padding=0), nn.PReLU()))
        # Non-linear Mapping
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s,
                                         kernel_size=3, stride=1, padding=1))

        self.layers.append(nn.PReLU())

        # Expanding
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s,
                                                   out_channels=d, kernel_size=1,
                                                   stride=1, padding=0), nn.PReLU()))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.last_part = nn.ConvTranspose2d(in_channels=d,
                                            out_channels=num_channels,
                                            kernel_size=9, stride=upscale_factor,
                                            padding=3, output_padding=1)


    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out


    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            # utils.weights_init_normal(m, mean=mean, std=std)
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()



class FSRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(FSRCNNTrainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.model = Net(num_channels=1, upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.2)
        self.criterion = nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)


    def train(self):
        """
        data: [torch.cuda.FloatTensor], 4 batches: [64, 64, 64, 8]
        """
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        """
        data: [torch.cuda.FloatTensor], 10 batches: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        """
        self.model.eval()
        avg_psnr = 0
        avg_ssim = 0
        contentPSNR = []
        contentSSIM = []

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                # mse = self.criterion(prediction, target)
                # PSNR = 10 * log10(1 / mse.item())
                # print("\n prediction: " + str(prediction))
                # print("\n target: " + str(target))

                # transform tensor to numpy array
                target_array = torch.Tensor.numpy(target)
                prediction_array = torch.Tensor.numpy(prediction)
                # print("\n prediction_array: " + str(prediction_array))
                # print(torch.is_tensor(prediction_array))
                # print(np.shape(prediction_array))

                PSNR = skimage.measure.compare_psnr(prediction_array[0, 0, :, :], target_array[0, 0, :, :],
                                                    data_range=1)
                SSIM = skimage.measure.compare_ssim(prediction_array[0, 0, :, :], target_array[0, 0, :, :])
                avg_psnr += PSNR
                avg_ssim += SSIM

                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f, SSIM: %.4f' % \
                             (PSNR, SSIM))
                # progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f, SSIM: %.4f' % \
                #              ((avg_psnr / (batch_num + 1)), (avg_ssim / (batch_num + 1))))

        print("\n    Average PSNR: {:.4f} dB\n".format(avg_psnr / len(self.testing_loader)))
        print("    Average SSIM: {:.4f}".format(avg_ssim / len(self.testing_loader)))

        content1 = avg_psnr / len(self.testing_loader)
        content2 = avg_ssim / len(self.testing_loader)

        contentPSNR.append(content1)
        contentSSIM.append(content2)

        return contentPSNR, contentSSIM

    def run(self):
        self.build_model()

        contentPSNRTotal = []; contentSSIMTotal = []

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            contentPSNR, contentSSIM = self.test()

            contentPSNRTotal.append(contentPSNR)
            contentSSIMTotal.append(contentSSIM)

            self.scheduler.step(epoch)
            # if epoch == self.nEpochs:
            #     self.save()

        return contentPSNRTotal, contentSSIMTotal



class DatasetFromFolder(data.Dataset):

    def __init__(self, image_dir, input_transform=None, target_transform=None):
        # image_dir: /Users/lihaotian/Desktop/DataSet/train
        # input_transform and target_transform both are Tensor(1, H, W)

        super(DatasetFromFolder, self).__init__()

        # image_fileNames is the whole path like: '/Users/.../ImageProcess/Test_F_01.tif'
        self.image_fileNames = [path.join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        # use image_fileNames[] load image one by one
        # input_image is original LR image without scale factor
        input_image = load_img(self.image_fileNames[index])

        target = input_image.copy()

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target



    def __len__(self):
        # get the numbers of image_fileNames, like: 600 counts for training set
        return len(self.image_fileNames)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # normal_(mean=0, std=1)
        m.weight.data.normal_(mean, std)
        # zero_() use 0 fill
        m.bias.data.zero_()



def is_image_file(filename):
    # check out if the fileName's extension is [".png", ".jpg", ".jpeg"]
    # return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])
    return any(filename.endswith(extension) for extension in [".tif"])



def load_img(filePath):
    # img = Image.open(filePath).convert('YCbCr')
    # y, _, _ = img.split()
    # return y

    img = Image.open(filePath).convert('L')
    return img



def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)



def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])



def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])



def get_training_set(upscale_factor):
    # set a root path
    root_dir = '/Users/lihaotian/Desktop/DataSet'

    # set the training sub path: .../train
    train_dir = path.join(root_dir, "train")

    # calculate the valid crop size, return the real crop size which can cut
    # such as  256 - (256%3) = 255, if scale_factor = 3
    crop_size = calculate_valid_crop_size(360, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))



def get_test_set(upscale_factor):
    # same with function: get_training_set()

    root_dir = '/Users/lihaotian/Desktop/DataSet'
    test_dir = path.join(root_dir, "test")
    crop_size = calculate_valid_crop_size(360, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))



def main():
    print('---> Loading datasets')
    train_set = get_training_set(args.upscale_factor)
    test_set = get_test_set(args.upscale_factor)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

    resultPSNR, resultSSIM = FSRCNNTrainer(args, training_data_loader, testing_data_loader).run()

    # np.savetxt('/Users/lihaotian/Desktop/dataPSNR.txt', np.array(resultPSNR), delimiter=',')
    # np.savetxt('/Users/lihaotian/Desktop/dataSSIM.txt', np.array(resultSSIM), delimiter=',')



if __name__ == '__main__':
    main()