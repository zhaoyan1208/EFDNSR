import os.path
import logging
import time
from collections import OrderedDict
import torch
import os
import cv2


from utils import utils_logger
from utils import utils_image as util
from EFDNSR_K2 import EFDNSR

def main():

    utils_logger.logger_info('AIM-track', log_path='AIM-track.log')
    logger = logging.getLogger('AIM-track')


    testsets = ''
    testset_L = ''


    torch.cuda.current_device()
    torch.cuda.empty_cache()
    #torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join('', '')
    model = RLFN()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # --------------------------------
    # read image
    # --------------------------------
    L_folder = os.path.join(testsets, testset_L)
    E_folder = os.path.join(testsets, testset_L+'_results')
    util.mkdir(E_folder)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(L_folder)
    logger.info(E_folder)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    img_SR = []
    for img in util.get_image_paths(L_folder):




        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        path, name =os.path.split(img)
        print(path)
        p = path.split('/')[-1]
        print(path.split('/')[-1])



        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        start.record()
        img_E = model(img_L)
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds


        img_E = util.tensor2uint(img_E)
        img_SR.append(img_E)



        isExists = os.path.exists(E_folder + '/' + p)
        if not isExists:
            os.makedirs(E_folder + '/' + p)
        util.imsave(img_E, os.path.join(E_folder + '/' + p, img_name+ext))
        print(os.path.join(E_folder + '/' + p, img_name+ext))

    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))



    psnr = []
    psnr_rgb = []

    ssim = []
    idx = 0
    H_folder = ''
    for img in util.get_image_paths(H_folder):
        img_H = util.imread_uint(img, n_channels=3)

        imgHY = util.bgr2ycbcr(img_H)
        img_SSR = util.bgr2ycbcr(img_SR[idx])
        psnr_rgb.append(util.calculate_psnr(img_SSR, imgHY))

        ssim.append(util.calculate_ssim(img_SSR, imgHY))
        idx += 1
    # logger.info('------> Average psnr of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr)/len(psnr)))
    logger.info('------> Average psnr_Y of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr_rgb) / len(psnr_rgb)))
    logger.info('------> Average ssim of ({}) is : {:.6f} dB'.format(L_folder, sum(ssim) / len(ssim)))


    # 单独测PSNR
    # psnr_rgb = []
    #
    # ssim = []
    # idx = 0
    # H_folder = ''
    # L_folder = ''
    # for img in util.get_image_paths(H_folder):
    #     for img1 in util.get_image_paths(L_folder):
    #         img_H = util.imread_uint(img, n_channels=3)
    #         # psnr.append(util.calculate_psnr(img_SR[idx], img_H))
    #         img_L = util.imread_uint(img1, n_channels=3)
    #
    #         imgHY = util.bgr2ycbcr(img_H)
    #         imgLY = util.bgr2ycbcr(img_L)
    #         psnr_rgb.append(util.calculate_psnr(imgLY, imgHY))
    #
    #         ssim.append(util.calculate_ssim(imgLY, imgHY))
    #         idx += 1
    # # logger.info('------> Average psnr of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr)/len(psnr)))
    # logger.info('------> Average psnr_Y of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr_rgb) / len(psnr_rgb)))
    # logger.info('------> Average ssim of ({}) is : {:.6f} dB'.format(L_folder, sum(ssim) / len(ssim)))


if __name__ == '__main__':

    main()
