import argparse
import os
import numpy as np
import time
import pathlib
from torch.utils.data import DataLoader
from data.dataset import DatasetFromFolder_train
from util.Nii_utils import NiiDataRead
from models.GAN_class import *
from util.util import *
from tensorboardX import SummaryWriter
import random
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

os.environ['PYTHONHASHSEED'] = '8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()

# set base_options
parser.add_argument("--image_dir", type=str, default='/home/DATASET', help="name of the dataset")
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--D_model', type=str, default='wave3DDiscriminator', help='specify discriminator architecture')
parser.add_argument('--G_model', type=str, default='MTTNET', help='specify generator architecture')
parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in the first conv layer')
parser.add_argument('--ndf', type=int, default=16, help='# of discrim filters in the first conv layer')
parser.add_argument('--n_layers_D', type=int, default=2, help='only used if netD==n_layers')
parser.add_argument('--G_norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--D_norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--depthSize', type=int, default=8, help='depth for 3d images')
parser.add_argument('--ImageSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--load_name', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--lambda_L1', type=float, default=20, help='weight for L1 loss')
parser.add_argument('--seed', type=int, default=15, help='random seed')
parser.add_argument('--VGG_loss', action='store_false', help='isVGG')
parser.add_argument('--isTrain', action='store_false', help='isTrain')
parser.add_argument('--Npatch', type=int, default=24, help='Npatch')
parser.add_argument('--print_freq_num', type=int, default=4, help='frequency of showing training results on console')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--max_epochs', type=int, default=200, help='# max_epoch')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr_max', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan ｜ wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--loss_pre_dir', type=str, default='perceive_loss/vgg19-dcbb9e9d.pth', help='resnet18_pretrain_path')
parser.add_argument('--Max_CT', type=int, default=2000, help='Max_CT')
parser.add_argument('--disx', type=int, default=10120, help='frequency of showing training results on console')  #10120,100000

opt = parser.parse_args()

opt.code_dir = os.getcwd()
opt.checkpoints_name = '%s_%s_results' % (opt.G_model,opt.D_model)
opt.checkpoints_dir = os.path.join(opt.code_dir, 'train_model', opt.checkpoints_name,'epoch_%d_Npatch_%d_maxCT_%d' % (opt.max_epochs,opt.Npatch,opt.Max_CT))
opt.model_results = os.path.join(opt.checkpoints_dir, 'model_results')
opt.file_name_txt = os.path.join(opt.checkpoints_dir, 'train_message.txt')
opt.prediction_results = os.path.join(opt.checkpoints_dir, 'prediction_results_')
opt.pretrain_model_path = os.path.join(opt.code_dir, opt.loss_pre_dir)

if not os.path.exists(opt.model_results):
    pathlib.Path(opt.model_results).mkdir(parents=True, exist_ok=True)

print_options(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
opt.gpu_ids = [0]

opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

# 设置随机数种子

np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

headneck_train_set = DatasetFromFolder_train(opt, region='Headandneck')
headneck_train_dataloader = DataLoader(dataset=headneck_train_set, num_workers=opt.num_threads,batch_size=opt.batch_size, shuffle=True)
all_train_set = DatasetFromFolder_train(opt, region='All')
all_train_dataloader = DataLoader(dataset=all_train_set, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True)

model = GANclass(opt)

if not opt.isTrain or opt.continue_train:
    load_networks(opt, model)

train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/train'))
val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/val'), flush_secs=2)

best_MAE = 1000
total_iters = 0
early_stop_num = 0
print('training')

for epoch in range(opt.epoch_count, opt.max_epochs + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration

    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
    train_iter_acc = []
    iter_num = 0

    if epoch <= 50:
        train_dataloader = headneck_train_dataloader
    else:
        train_dataloader = all_train_dataloader

    train_size = len(train_dataloader)
    opt.print_freq = int(train_size / opt.print_freq_num)

    for i, data in enumerate(train_dataloader):
        iter_start_time = time.time()
        total_iters += 1
        epoch_iter += 1
        iter_num += 1
        model.set_input(data)
        model.forward(epoch)

        labels = data['label']

        predicted = torch.argmax(model.pre_label, dim=1, keepdim=False)
        predicted = predicted.cpu()
        acc = (labels == predicted).sum().float().item() / labels.size(0)
        train_iter_acc.append(acc)

        if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
            losses = get_current_losses(model)
            lr = model.optimizer_G.param_groups[0]['lr']
            iter_acc = np.sum(train_iter_acc) / iter_num
            if epoch <= 50:
                iter_acc = 'null'
                display_message = '(epoch: %d, iters: %d/%d, lr: %.6f  iter_acc:%s)' % (epoch, epoch_iter, train_size, lr,iter_acc)
            else:
                display_message = '(epoch: %d, iters: %d/%d, lr: %.6f  iter_acc:%.3f)' % (epoch, epoch_iter, train_size, lr, iter_acc)
            for k, v in losses.items():
                display_message += '%s: %.3f ' % (k, v)
            print(display_message)  # print the message
            train_writer.add_scalar('learning_rate', lr, total_iters)
            for k, v in losses.items():
                train_writer.add_scalar('%s' % k, v, total_iters)

            train_iter_acc = []
            iter_num = 0

        iter_data_time = time.time()

    update_learning_rate(model, opt.max_epochs, epoch, opt.lr_max)


    '''***********************验证集*************************'''
    epoch_val_MAE = []
    epoch_val_SSIM = []
    epoch_val_PSNR = []
    head_neck_val_MAE = []
    head_neck_val_SSIM = []
    head_neck_val_PSNR = []
    abdominal_val_MAE = []
    abdominal_val_SSIM = []
    abdominal_val_PSNR = []

    if epoch <= 50:
        val_txt = os.path.join(opt.code_dir, 'data', 'headneck_val.txt')
    else:
        val_txt = os.path.join(opt.code_dir, 'data', 'all_val.txt')

    with open(val_txt, 'r') as f:
        name_list = f.readlines()
    image_filenames = [n.strip('\n') for n in name_list]

    patch_size = opt.ImageSize
    patch_deep = opt.depthSize

    Val_run = False

    if epoch < 10:
        if epoch % 2 == 0:
            Val_run = True
    elif (epoch > 10) & (epoch < int(opt.max_epochs*0.5)):
        if epoch % 10 == 0:
            Val_run = True
    elif (epoch > int(opt.max_epochs*0.5)) & (epoch < int(opt.max_epochs*0.8)):
        if epoch % 5 == 0:
            Val_run = True
    else:
        Val_run = True

    if Val_run:
        with torch.no_grad():
            for sub_index, sub in enumerate(image_filenames):
                if 'Abdomen' in sub:
                    MR, spacing, origin, direction = NiiDataRead(
                        os.path.join(opt.image_dir, sub, 'MR.nii.gz'))
                    CT, spacing1, origin1, direction1 = NiiDataRead(
                        os.path.join(opt.image_dir, sub, 'CT.nii.gz'))
                    MASK, spacing, origin, direction = NiiDataRead(
                        os.path.join(opt.image_dir, sub, 'mask.nii.gz'))
                    true_label = 1
                else:
                    MR, spacing, origin, direction = NiiDataRead(
                        os.path.join(opt.image_dir, sub, 'MR.nii.gz'))
                    CT, spacing1, origin1, direction1 = NiiDataRead(
                        os.path.join(opt.image_dir, sub, 'CT.nii.gz'))
                    MASK, spacing, origin, direction = NiiDataRead(
                        os.path.join(opt.image_dir, sub, 'mask.nii.gz'))
                    true_label = 0

                MR = normalization(MR, 0, 255)

                z, y, x = np.where((MASK > 0))
                x_min = np.min(x)
                x_max = np.max(x)
                y_min = np.min(y)
                y_max = np.max(y)
                z_min = np.min(z)
                z_max = np.max(z)

                z_edge1 = np.where((z + patch_deep / 2) > MR.shape[0])
                z[z_edge1] = MR.shape[0] - patch_deep / 2

                z_edge2 = np.where((z - patch_deep / 2) < 0)
                z[z_edge2] = patch_deep / 2

                y_edge1 = np.where((y + patch_size / 2) > MR.shape[1])
                y[y_edge1] = MR.shape[1] - patch_size / 2

                y_edge2 = np.where((y - patch_size / 2) < 0)
                y[y_edge2] = patch_size / 2

                x_edge1 = np.where((x + patch_size / 2) > MR.shape[2])
                x[x_edge1] = MR.shape[2] - patch_size / 2

                x_edge2 = np.where((x - patch_size / 2) < 0)
                x[x_edge2] = patch_size / 2

                MR = MR[None, :, :, :]

                output = np.zeros((MASK.shape[0], MASK.shape[1], MASK.shape[2])).astype('float')
                count_used = np.zeros((MASK.shape[0], MASK.shape[1], MASK.shape[2])).astype('float') + 0.0001
                dis = opt.disx
                total_label = []
                for num in range(len(x)):
                    if num % dis == 0:
                        deep = z[num]
                        height = y[num]
                        width = x[num]
                        X_MR = MR[:, int(deep - patch_deep / 2):int(deep + patch_deep / 2),
                               int(height - patch_size / 2):int(height + patch_size / 2),
                               int(width - patch_size / 2):int(width + patch_size / 2)]
                        X_MR = torch.tensor(X_MR).unsqueeze(0).float().to(opt.device)
                        CT_pred, pre_label = model.netG(X_MR)
                        pre_label = pre_label.cpu().numpy()
                        total_label.append(np.argmax(pre_label))
                        CT_pred = np.squeeze(CT_pred.cpu().numpy())
                        CT_pred[CT_pred < -1] = -1
                        CT_pred[CT_pred > 1] = 1
                        output[int(deep - patch_deep / 2):int(deep + patch_deep / 2),
                        int(height - patch_size / 2):int(height + patch_size / 2),
                        int(width - patch_size / 2):int(width + patch_size / 2)] += CT_pred
                        count_used[int(deep - patch_deep / 2):int(deep + patch_deep / 2),
                        int(height - patch_size / 2):int(height + patch_size / 2),
                        int(width - patch_size / 2):int(width + patch_size / 2)] += 1
                output = output / count_used
                output[MASK == 0] = -1

                pre_this_label = max(total_label, key=total_label.count)
                true_num = total_label.count(true_label)
                true_acc = true_num/len(total_label)

                output = inverser_norm_ct(output, opt.Max_CT, -1000)

                MAE = opt.Max_CT + 1000
                data_range = max(output[MASK > 0].max() - output[MASK > 0].min(),CT[MASK > 0].max() - CT[MASK > 0].min())
                SSIM = structural_similarity(output[MASK > 0], CT[MASK > 0], data_range=data_range)
                PSNR = peak_signal_noise_ratio(output[MASK > 0], CT[MASK > 0], data_range=data_range)

                message = 'epoch[%d/%d] val[%d/%d]: The MAE,SSIM,PSNR of %s is %.3f, %3f, %.3f, pre_label is %d, true_label is %d  true_acc is %.5f' % (
                        epoch, opt.max_epochs, sub_index + 1, len(image_filenames), sub, MAE, SSIM, PSNR,
                        pre_this_label, true_label, true_acc)

                print(message)
                with open(opt.file_name_txt, 'a') as opt_file:
                    opt_file.write(message)
                    opt_file.write('\n')

                epoch_val_MAE.append(MAE)
                epoch_val_SSIM.append(SSIM)
                epoch_val_PSNR.append(PSNR)

                if 'Abdomen' not in sub:
                    head_neck_val_MAE.append(MAE)
                    head_neck_val_PSNR.append(PSNR)
                    head_neck_val_SSIM.append(SSIM)
                    message = 'The head_neck MAE, SSIM, PSNR is %.3f,%.3f,%.3f, the total MAE, SSIM, PSNR is %.3f,%.3f,%.3f' % (
                        np.mean(head_neck_val_MAE),np.mean(head_neck_val_SSIM),np.mean(head_neck_val_PSNR),
                        np.mean(epoch_val_MAE),np.mean(epoch_val_SSIM),np.mean(epoch_val_PSNR))
                else:
                    abdominal_val_MAE.append(MAE)
                    abdominal_val_PSNR.append(PSNR)
                    abdominal_val_SSIM.append(SSIM)
                    message = 'The abdominal, SSIM, PSNR is %.3f,%.3f,%.3f, the total MAE, SSIM, PSNR is %.3f,%.3f,%.3f' % (
                    np.mean(abdominal_val_MAE), np.mean(abdominal_val_SSIM), np.mean(abdominal_val_PSNR),
                    np.mean(epoch_val_MAE), np.mean(epoch_val_SSIM), np.mean(epoch_val_PSNR))

                print(message)
                with open(opt.file_name_txt, 'a') as opt_file:
                    opt_file.write(message)
                    opt_file.write('\n')

        epoch_val_MAE = np.mean(epoch_val_MAE)
        val_writer.add_scalar('val_MAE', epoch_val_MAE, epoch)

        if epoch_val_MAE < best_MAE:
            best_MAE = epoch_val_MAE
            save_networks(opt, 'best_MAE', model, epoch)


        print('saving the model')
        save_networks(opt, 'latest', model, epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.max_epochs, time.time() - epoch_start_time))

train_writer.close()


