import os
import glob
import cv2
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import pyshark
from pyshark.capture.capture import TSharkCrashException
from diffusers.models.unets.unet_2d import UNet2DModel


# Parameters

IMAGE_SIZE = 16  # image size
BYTE_SIZE = IMAGE_SIZE * IMAGE_SIZE
PCAP_DIR = "E://1/data/USTC-TFC2016/Benign/SMB"  # Address          E://1/data/isot_app_and_botnet_dataset/application_data      E://1/data/USTC-TFC2016/Benign
OUTPUT_DIR = "C://Users/whf80/Desktop/SMB"          # D:/Data/isot_Benign_images/    D://Data/Benign_images_16/
TSHARK_PATH = "D:/Wireshark/tshark.exe"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_pcap(pcap_file):
    try:
        with pyshark.FileCapture(
                pcap_file,
                display_filter="tcp or udp or http or dns or ftp or mqtt",
                tshark_path=TSHARK_PATH
        ) as capture:
            header_only_data_list, l5_data_list, l7_data_list = [], [], []
            for packet in capture:
                try:
                    packet_str = str(packet)

                    # Network, data link, transport layer
                    session_info = ""
                    if hasattr(packet, "ip") or hasattr(packet, "ipv6"):
                        if hasattr(packet, "eth"):
                            src_mac = getattr(packet.eth, "src", "N/A")
                            dst_mac = getattr(packet.eth, "dst", "N/A")
                        else:
                            src_mac, dst_mac = "N/A", "N/A"

                        src = getattr(packet, "ip", getattr(packet, "ipv6", "N/A"))
                        dst = getattr(packet, "ipv6", getattr(packet, "ip", "N/A"))
                        if hasattr(packet, "transport_layer"):
                            transport_layer = packet.transport_layer
                        else:
                            transport_layer = "Unknown"
                        try:
                            src_port = getattr(packet[transport_layer], "srcport", "0")
                            dst_port = getattr(packet[transport_layer], "dstport", "0")
                        except AttributeError:
                            src_port, dst_port = "0", "0"
                        session_info = f"{src_mac} {dst_mac} {src} {dst} {transport_layer} {src_port} {dst_port}"
                        l5_data_list.append(session_info)
                    else:
                        l5_data_list.append("")

                    # Application layer
                    app_info = ""
                    if hasattr(packet, "http"):
                        app_info = f"HTTP: {getattr(packet.http, 'host', 'N/A')} {getattr(packet.http, 'request_uri', 'N/A')}"
                    elif hasattr(packet, "dns"):
                        app_info = f"DNS Query: {getattr(packet.dns, 'qry_name', 'N/A')}"
                    elif hasattr(packet, "ftp"):
                        app_info = f"FTP: {getattr(packet.ftp, 'request_command', 'N/A')}"
                    elif hasattr(packet, "mqtt"):
                        app_info = f"MQTT: {getattr(packet.mqtt, 'topic', 'N/A')}"
                    else:
                        app_info = ""
                    l7_data_list.append(app_info)

                    # Header-only Data
                    remaining_info = packet_str
                    if session_info and session_info in remaining_info:
                        remaining_info = remaining_info.replace(session_info, "")
                    if app_info and app_info in remaining_info:
                        remaining_info = remaining_info.replace(app_info, "")
                    header_only_data_list.append(remaining_info[:BYTE_SIZE])

                except Exception as e:
                    print(f"Error parsing packet in {pcap_file}: {e}")
                    continue
    except TSharkCrashException as e:
        print(f"TShark crashed for {pcap_file}: {e}")
        return "", "", ""
    except Exception as e:
        print(f"Failed to open {pcap_file}: {e}")
        return "", "", ""

    header_data = "".join(header_only_data_list) if header_only_data_list else ""
    l5_data = "".join(l5_data_list) if l5_data_list else ""
    l7_data = "".join(l7_data_list) if l7_data_list else ""

    return header_data, l5_data, l7_data



def split_string_into_chunks(data, chunk_size=BYTE_SIZE):
    if len(data) % chunk_size != 0:
        data = data.ljust(((len(data) // chunk_size) + 1) * chunk_size, "\x00")
    num_chunks = len(data) // chunk_size
    return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]


def generate_hsv_images(all_layer_data, l5_data, l7_data):
    def split_bytes_into_chunks(data_str, chunk_size=BYTE_SIZE):
        data_bytes = data_str.encode('latin1', errors="ignore")
        if len(data_bytes) % chunk_size != 0:
            data_bytes = data_bytes.ljust(((len(data_bytes) // chunk_size) + 1) * chunk_size, b'\x00')
        num_chunks = len(data_bytes) // chunk_size
        return [data_bytes[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    H_chunks = split_bytes_into_chunks(all_layer_data, BYTE_SIZE)
    S_chunks = split_bytes_into_chunks(l5_data, BYTE_SIZE)
    V_chunks = split_bytes_into_chunks(l7_data, BYTE_SIZE)

    num_images = min(len(H_chunks), len(S_chunks), len(V_chunks))
    hsv_images = []
    rgb_images = []

    for i in range(num_images):
        H_bytes = H_chunks[i]
        S_bytes = S_chunks[i]
        V_bytes = V_chunks[i]

        H = np.frombuffer(H_bytes, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
        S = np.frombuffer(S_bytes, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
        V = np.frombuffer(V_bytes, dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)

        hsv_img = np.stack([H, S, V], axis=-1)
        rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        hsv_images.append(hsv_img)
        rgb_images.append(rgb_img)

    return hsv_images, rgb_images


def process_pcap_file(pcap_path, output_subdir):
    filename = os.path.basename(pcap_path).replace(".pcap", "")
    all_layer_data, l5_data, l7_data = parse_pcap(pcap_path)
    if not all_layer_data.strip():
        print(f"{filename} has no valid data, skipping.")
        return

    hsv_images, rgb_images = generate_hsv_images(all_layer_data, l5_data, l7_data)
    os.makedirs(output_subdir, exist_ok=True)

    for idx, (hsv_img, rgb_img) in enumerate(zip(hsv_images, rgb_images)):
        npy_path = os.path.join(output_subdir, f"{filename}_{idx}.npy")
        png_path = os.path.join(output_subdir, f"{filename}_{idx}.png")
        np.save(npy_path, hsv_img)
        cv2.imwrite(png_path, rgb_img)
        print(f"{filename} image {idx} saved: {png_path}")


def process_all_pcaps():
    pcap_files = []

    for root, _, files in os.walk(PCAP_DIR):
        for file in files:
            if file.endswith(".pcap"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, PCAP_DIR)
                output_subdir = os.path.join(OUTPUT_DIR, relative_path)
                pcap_files.append((full_path, output_subdir))

    for pcap_path, output_subdir in tqdm(pcap_files, desc="Processing PCAP files", unit="file"):
        process_pcap_file(pcap_path, output_subdir)

    print("Processing complete!")


if __name__ == "__main__":
    process_all_pcaps()

