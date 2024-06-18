import os
import re
import time
import logging
import argparse
import cv2
from custom import Dasac

def infer_images(args, input_images, output_dir=None, save_output=False):
    dasac = Dasac(args)
    start_time = time.time()
    for input_path in input_images:
        logging.info("Processing {}".format(input_path))
        image_raw = cv2.imread(input_path)
        image = dasac.preprocess_image(image_raw)
        mask_pred = dasac.infer(image)
        image_overlay = dasac.get_image_overlay(image, mask_pred)
        if save_output:
            if not output_dir:
                output_path = input_path.replace(".png", "_output.png")
            else:
                output_path = os.path.join(
                    output_dir,
                    os.path.basename(input_path).replace(".png", "_output.png"),
                )
            cv2.imwrite(output_path, image_overlay)
    end_time = time.time()
    total_time = end_time - start_time
    return total_time


def get_images_list(images_dir):
    images_list = []
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith(".png"):
                images_list.append(os.path.join(root, file))
    images_list.sort()
    logging.info("Found {} images".format(len(images_list)))
    return images_list


def single_test(save_output=False):
    args = {
        "cfg_file": "configs/deeplabv2_resnet101.yaml",
        "set_cfgs": [],
        "resume": "snapshots/cityscapes/baselines/resnet101_gta/final_e136.pth",
        "dataloader": "cityscapes",
        "infer_list": "val_cityscapes",
    }
    input_path = os.path.join("input", "test_input.png")
    if not os.path.exists(input_path):
        logging.error("Input image not found")
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_images = [os.path.join("input", "test_input.png")]
    total_time = infer_images(args, input_images, output_dir, save_output)
    logging.info("Inference time: {:.2f}s".format(total_time))


def batch_test(save_output=False):
    args = {
        "cfg_file": "configs/deeplabv2_resnet101.yaml",
        "set_cfgs": [],
        "resume": "snapshots/cityscapes/baselines/resnet101_gta/final_e136.pth",
        "dataloader": "cityscapes",
        "infer_list": "val_cityscapes",
    }
    images_dir = os.path.join("input", "image")
    input_images = get_images_list(images_dir)
    output_dir = os.path.join("output", "image")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    total_time = infer_images(args, input_images, output_dir, save_output)
    logging.info("Inference time: {:.2f}s".format(total_time))
    avg_time = total_time / len(input_images)
    logging.info("Average inference time: {:.2f}s".format(avg_time))


def process_logs(gpu_info_log_path, test_log_path):
    # 处理 gpu_info.log
    with open(gpu_info_log_path, "r") as f:
        lines = f.readlines()
    temps, powers, mems, utils = [], [], [], []
    for line in lines:
        temp, power, mem, util = map(int, re.findall(r"\d+", line))
        if util != 0:  # 去除占用率为0 %的行
            temps.append(temp)
            powers.append(power)
            mems.append(mem)
            utils.append(util)
    avg_temp = sum(temps) / len(temps)
    avg_power = sum(powers) / len(powers)
    avg_mem = sum(mems) / len(mems)
    avg_util = sum(utils) / len(utils)

    # 处理 test.log
    with open(test_log_path, "r") as f:
        lines = f.readlines()
    total_images = int(re.search(r"\d+", [line for line in lines if "Found" in line][0]).group())
    total_time = float(re.search(r"\d+\.\d+", [line for line in lines if "Inference time" in line][0]).group())
    fps = total_images / total_time

    return avg_temp, avg_power, avg_mem, avg_util, fps


def main(args):
    if args.mode == "single":
        single_test(save_output=args.save_output)
    elif args.mode == "batch":
        batch_test(save_output=args.save_output)
    elif args.mode == "parse_logs":
        avg_temp, avg_power, avg_mem, avg_util, fps = process_logs(args.gpu_info_log_path, args.test_log_path)
        print("Average temperature: {}°C".format(avg_temp))
        print("Average power: {}W".format(avg_power))
        print("Average memory usage: {}MB".format(avg_mem))
        print("Average GPU utilization: {}%".format(avg_util))
        print("FPS: {:.2f}".format(fps))
    else:
        raise ValueError("Invalid mode. Choose 'single', 'batch' or 'parse_logs'")

def parse_args():
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument(
        "--mode",
        type=str,
        default="batch",
        help="Test mode: single, batch or parse_logs",
    )
    parser.add_argument("--save_output", action="store_true", help="Save output or not")
    parser.add_argument(
        "--log_name",
        type=str,
        default="test.log",
        help="Name of the log file",
    )
    parser.add_argument(
        "--gpu_info_log_path",
        type=str,
        default="gpu_info.log",
        help="Path to gpu_info.log",
    )
    parser.add_argument("--test_log_path", type=str, default="test.log", help="Path to test.log")
    return parser.parse_args()

def init_logger(log_name):
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)

if __name__ == "__main__":
    args = parse_args()
    init_logger(args.log_name)
    main(args)
