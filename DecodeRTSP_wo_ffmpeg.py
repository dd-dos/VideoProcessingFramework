# import os
# import sys
# if os.name == 'nt':
#     # Add CUDA_PATH env variable
#     cuda_path = os.environ["CUDA_PATH"]
#     if cuda_path:
#         os.add_dll_directory(cuda_path)
#     else:
#         print("CUDA_PATH environment variable is not set.", file = sys.stderr)
#         print("Can't set CUDA DLLs search path.", file = sys.stderr)
#         exit(1)

#     # Add PATH as well for minor CUDA releases
#     sys_path = os.environ["PATH"]
#     if sys_path:
#         paths = sys_path.split(';')
#         for path in paths:
#             if os.path.isdir(path):
#                 os.add_dll_directory(path)
#     else:
#         print("PATH environment variable is not set.", file = sys.stderr)
#         exit(1)

from concurrent.futures import thread
from re import I
import time
from collections import deque
from threading import Thread
from multiprocessing import Process

import torch
from loguru import logger

import PyNvCodec as nvc
from streaming import JanusClient


def main(gpuID, encFilePath, camID):
    st_main = time.time()
    nvDec = nvc.PyNvDecoder(encFilePath, gpuID)

    width = nvDec.Width()
    height = nvDec.Height()

    # Colorspace conversion contexts. 
    color_space, color_range = nvDec.ColorSpace(), nvDec.ColorRange()
    if nvc.ColorSpace.UNSPEC == color_space:
        color_space = nvc.ColorSpace.BT_601
    if nvc.ColorRange.UDEF == color_range:
        color_range = nvc.ColorRange.MPEG

    cc_ctx = nvc.ColorspaceConversionContext(color_space, color_range)

    # Initialize colorspace conversion chain
    if color_space != nvc.ColorSpace.BT_709:
        nvYuv = nvc.PySurfaceConverter(width, height, nvDec.Format(), nvc.PixelFormat.YUV420, gpuID)
        nvCvt = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB, gpuID)
    else:
        nvYuv = None
        nvCvt = nvc.PySurfaceConverter(width, height, nvDec.Format(), nvc.PixelFormat.RGB, gpuID)

    # PyTorch tensor the VPF Surfaces will be exported to

    # Initialize Janus client
    logger.info(f"Decoder init time: {(time.time()-st_main)*1000} ms")
    frame_deque = deque(maxlen=2)
    st_janus = time.time()
    janus_client = JanusClient(
        janus_server_url="http://192.168.40.3:8088/janus",
        # ice_server_url="turn:janus.truongkyle.tech:3478?transport=udp",
        # ice_server_username="horus",
        # ice_server_password="horus123@!",
        frame_dequeue=frame_deque,
        cam_id=camID,
    )
    # logger.info(f"Janus client init time: {(time.time()-st_janus)*1000} ms")

    decoded_frame = 0
    while True:
        try:
            try:
                logger.debug(f"#"*30)
                st = time.time()
                nv12_surface = nvDec.DecodeSingleSurface()
                import ipdb; ipdb.set_trace(context=10)
                logger.debug(f"Decode single surface took: {(time.time()-st)*1000} ms")
            except nvc.HwResetException:
                logger.warning("Hardware reset detected. Trying to recover...")
                continue

            if nv12_surface.Empty():
                logger.warning('Decoding finished')
                break
            
            decoded_frame += 1
            st = time.time()
            if nvYuv:
                yuvSurface = nvYuv.Execute(nv12_surface, cc_ctx)
                cvtSurface = nvCvt.Execute(yuvSurface, cc_ctx)
            else:
                cvtSurface = nvCvt.Execute(nv12_surface, cc_ctx)
            logger.debug(f"Conversion took: {(time.time()-st)*1000} ms")

            st = time.time()
            surface_tensor = torch.zeros(height, width, 3, dtype=torch.uint8,
                                        device=torch.device(f'cuda:{gpuID}'))
            logger.debug(f"Tensor init took: {(time.time()-st)*1000} ms")

            st = time.time()
            cvtSurface.PlanePtr().Export(surface_tensor.data_ptr(), width * 3, gpuID)
            # logger.debug(f"Framerate: {decoded_frame / (time.time() - st)}")
            logger.debug(f"Export took: {(time.time()-st)*1000} ms")

            # This should be a typical rgb image but idk why it's a bgr one??
            st = time.time()
            bgr_img = surface_tensor.cpu().numpy()
            rgb_img = bgr_img[..., ::-1]
            logger.debug(f"Numpy conversion took: {(time.time()-st)*1000} ms")

            frame_deque.appendleft(rgb_img)
        except KeyboardInterrupt:
            janus_client.stop()
            return

if __name__ == "__main__":
    gpuID = 0
    encFilePath = "rtsp://admin:Techainer123@techainer-hikvision-office-2:554/media/video1"
    
    thread_list = []
    for idx in range(4):
        main_thread = Thread(target=main, args=(gpuID, encFilePath, idx))
        # main_thread = Process(target=main, args=(gpuID, encFilePath, idx))
        main_thread.start()
        thread_list.append(main_thread)
        logger.info(f"Init camera {idx}")
    # main(gpuID, encFilePath, 0)

    for _thread in thread_list:
        _thread.join()
    logger.info(f"All clean")
