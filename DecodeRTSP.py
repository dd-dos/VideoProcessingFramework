import traceback
from threading import Thread

import cv2
import torch

import PyNvCodec as nvc


class Worker(Thread):
    def __init__(self, gpuID, encFile):
        Thread.__init__(self)
        self.gpuID = gpuID
        self.nvDec = nvc.PyNvDecoder(encFile, gpuID, {'rtsp_transport': 'tcp', 'max_delay': '5000000', 'bufsize': '30000k'})
        
        self.width, self.height = self.nvDec.Width(), self.nvDec.Height()

        # Initialize colorspace conversion chain
        if self.nvDec.ColorSpace() != nvc.ColorSpace.BT_709:
            self.nvYuv = nvc.PySurfaceConverter(self.width, self.height, self.nvDec.Format(), nvc.PixelFormat.YUV420, gpuID)
        else:
            self.nvYuv = None

        if self.nvYuv:
            self.nvCvt = nvc.PySurfaceConverter(self.width, self.height, self.nvYuv.Format(), nvc.PixelFormat.RGB, gpuID)
        else:
            self.nvCvt = nvc.PySurfaceConverter(self.width, self.height, self.nvDec.Format(), nvc.PixelFormat.RGB, gpuID)
        
        self.num_frame = 0
        self.surface_tensor = torch.zeros(self.height, self.width, 3, dtype=torch.uint8,
                                    device=torch.device(f'cuda:{gpuID}'))

        cspace, crange = self.nvDec.ColorSpace(), self.nvDec.ColorRange()
        if nvc.ColorSpace.UNSPEC == cspace:
            cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == crange:
            crange = nvc.ColorRange.MPEG
        self.cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)

    def run(self):
        try:
            while True:
                try:
                    rawSurface = self.nvDec.DecodeSingleSurface()
                    if (rawSurface.Empty()):
                        print('No more video frames')
                        break
                except nvc.HwResetException:
                    print('Continue after HW decoder was reset')
                    continue
 
                if self.nvYuv:
                    yuvSurface = self.nvYuv.Execute(rawSurface, self.cc_ctx)
                    cvtSurface = self.nvCvt.Execute(yuvSurface, self.cc_ctx)
                else:
                    cvtSurface = self.nvCvt.Execute(rawSurface, self.cc_ctx)

                cvtSurface.PlanePtr().Export(self.surface_tensor.data_ptr(), self.width*3, self.gpuID)
                cv2.imwrite(f"frames/{self.num_frame}.jpg", self.surface_tensor.cpu().numpy())
                self.num_frame += 1
 
        except Exception as e:
            print(traceback.format_exc())

def create_threads(gpu_id1, input_file1):
    th1  = Worker(gpu_id1, input_file1)
    th1.start()
    th1.join()
 
if __name__ == "__main__":
    gpu_id = 0
    input_file = "rtsp://admin:Techainer123@techainer-hikvision-office-3:554/media/video1"
    input_file = "rtsp://rtsp.stream/pattern"
    input_file = "rtsp://192.168.40.4:8554/live.sdp"
    input_file = "rtsp://admin:Techainer123@192.168.50.4/Streaming/Channels/101"

    create_threads(gpu_id, input_file)
