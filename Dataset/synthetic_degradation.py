import random
import cv2
import subprocess
import numpy as np

class SynthDeg:

    def __init__(self) -> None:
        self.blurs = {'Isotropic': Blur.isotropic_blur, 'Anisotropic': Blur.anisotropic_blur, 'LinearMotion': Blur.linear_motion_blur}
        self.scaling = {'Bicubic': Downscale.bicubic_downscale, 'Bilinear': Downscale.bilinear_downscale, 'AreaSampling': Downscale.areasampling_downscale}
        self.noise = {'Gaussian': Noise.gaussian_noise, 'SaltnPepper': Noise.saltpepper_noise, 'Speckle': Noise.speckle_noise}
        self.compression = {'JPEG': Compression.jpeg_compression, 'H264': Compression.h264_compression}

    def degrade_image(self, img: np.ndarray, blur: str = None, scaling: str = None, noise: str = None) -> np.ndarray:
        if blur is None:
            blur = random.choice(list(self.blurs))
            # print(blur)
        if scaling is None:
            scaling = random.choice(list(self.scaling))
            # print(scaling)
        if noise is None:
            noise = random.choice(list(self.noise))
            # print(noise)
        
        #First Pass
        blurred_img = self.blurs[blur](img)
        downscaled_img = self.scaling[scaling](blurred_img)
        noised_img = self.noise[noise](downscaled_img)
        compressed_img = self.compression['H264'](noised_img)

        #Second Pass
        scndblurred_img = self.blurs['Isotropic'](compressed_img, random.uniform(0.2, 0.8))
        scndnoised_img = self.noise['Gaussian'](scndblurred_img, random.uniform(1, 8))
        final_img = self.compression['JPEG'](scndnoised_img, random.randint(70, 85))

        return final_img

class Blur:
    @staticmethod
    def isotropic_blur(img: np.ndarray, sig: float = 0) -> np.ndarray:
        if sig <= 0:
            sig = random.uniform(0.2, 3)
        
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sig, sigmaY=sig)
    
    @staticmethod
    def anisotropic_blur(img: np.ndarray, sigX: float = 0.0, sigY: float = 0.0) -> np.ndarray:
        if sigX <= 0:
            sigX = random.uniform(0.5, 8)
        if sigY <= 0:
            sigY = random.uniform(0.5, 8)
        
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigX, sigmaY=sigY)
    
    @staticmethod
    def linear_motion_blur(img: np.ndarray, rad: float = None, length: int = None) -> np.ndarray:
        if rad is None:
            rad = random.uniform(0, np.pi)
        if length is None:
            length = random.randint(1, 20)

        kernel = np.zeros((length, length), dtype=np.float32)
        center = length // 2
        dx = int(round(np.cos(rad) * center))
        dy = int(round(np.sin(rad) * center))
        u = (center - dx, center - dy)
        v = (center + dx, center + dy)
        cv2.line(kernel, u, v, 1.0, thickness=1)
        kernel /= kernel.sum()

        return cv2.filter2D(img, -1, kernel)

class Downscale:
    @staticmethod
    def bicubic_downscale(img: np.ndarray, scale: float = 0.25) -> np.ndarray:  
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def bilinear_downscale(img: np.ndarray, scale: float = 0.25) -> np.ndarray:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def areasampling_downscale(img: np.ndarray, scale: float = 0.25) -> np.ndarray:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
class Noise:
    @staticmethod
    def gaussian_noise(img: np.ndarray, mean: int = 0, std: float = None) -> np.ndarray:
        if std is None:
            std = random.uniform(1, 25)

        noise = np.random.normal(mean, std, img.shape)
        res = img.astype(np.float32) + noise
        
        return np.clip(res, 0, 255).astype(np.uint8)
    
    @staticmethod
    def saltpepper_noise(img: np.ndarray, amount: float = None, s_vs_p: float = None) -> np.ndarray:
        if amount is None:
            amount = random.uniform(0.0, 0.005)
        if s_vs_p is None:
            s_vs_p = random.uniform(0.3, 0.7)

        h, w = img.shape[:2]
        pixels = h * w

        num_salt = int(np.ceil(amount * pixels * s_vs_p))
        salt_r = np.random.randint(0, h, num_salt)
        salt_c = np.random.randint(0, w, num_salt)

        num_pepper = int(np.ceil(amount * pixels * (1 - s_vs_p)))
        pepper_r = np.random.randint(0, h, num_pepper)
        pepper_c = np.random.randint(0, w, num_pepper)

        res = img.astype(np.float32)
        res[salt_r, salt_c] = 255
        res[pepper_r, pepper_c] = 0

        return np.clip(res, 0, 255).astype(np.uint8)
    
    @staticmethod
    def speckle_noise(img: np.ndarray, std: float = None) -> np.ndarray:
        if std is None:
            std = random.uniform(0.02, 0.15)

        noise = np.random.randn(*img.shape) * std
        res = img.astype(np.float32) + img.astype(np.float32) * noise

        return np.clip(res, 0, 255).astype(np.uint8)

class Compression:
    @staticmethod
    def jpeg_compression(img: np.ndarray, quality: int = None) -> np.ndarray:
        if quality is None:
            quality = random.randint(30, 95)
        
        encode = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpeg', img, encode)
        compressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        return compressed

    #Credit to: Frobot StackOverflow
    @staticmethod
    def h264_compression(img: np.ndarray, amount: int = None) -> np.ndarray:
        _, png_data = cv2.imencode('.png', img)
        if amount is None:
            amount = random.randint(25, 50)

        ffmpeg_command = [
            'ffmpeg',
            '-y',                        # Overwrite output files without asking
            '-i', 'pipe:0',              # Input from stdin
            '-vcodec', 'libx264',        # Use H.264 codec
            '-qp', str(amount),          # Quality parameter
            '-pix_fmt', 'yuv420p',       # Pixel format
            '-f', 'matroska',            # Use MKV container
            'pipe:1'                     # Output to stdout
        ]

        result = subprocess.run(
            ffmpeg_command,
            input=png_data.tobytes(),    # Pass PNG data to stdin
            stdout=subprocess.PIPE,      # Capture stdout
            stderr=subprocess.PIPE       # Capture stderr for debugging
        )

        if result.returncode != 0:
            print("FFmpeg error during compression:", result.stderr.decode())
            raise RuntimeError("FFmpeg compression failed")

        compressed_data = result.stdout

        #Decompression
        ffmpeg_command = [
            'ffmpeg',
            '-i', 'pipe:0',              # Input from stdin
            '-f', 'rawvideo',            # Output raw video format
            '-pix_fmt', 'bgr24',         # Pixel format
            'pipe:1'                     # Output to stdout
        ]

        result = subprocess.run(
            ffmpeg_command,
            input=compressed_data,       # Pass compressed data to stdin
            stdout=subprocess.PIPE,      # Capture stdout
            stderr=subprocess.PIPE       # Capture stderr for debugging
        )

        if result.returncode != 0:
            print("FFmpeg error during decompression:", result.stderr.decode())
            raise RuntimeError("FFmpeg decompression failed")

        raw_img_data = result.stdout

        expected_size = img.shape[1] * img.shape[0] * 3

        if len(raw_img_data) != expected_size:
            print("Unexpected raw img data size:", len(raw_img_data))
            raise ValueError(f"Cannot reshape array of size {len(raw_img_data)} into shape ({img.shape[0]},{img.shape[1]},3)")

        # Convert the raw data to a numpy array
        frame = np.frombuffer(raw_img_data, dtype=np.uint8).reshape((img.shape[0], img.shape[1], 3))

        return frame

if __name__ == "__main__":
    img = cv2.imread('initial_test.jpg')
    
    deg = SynthDeg()
    deg_img = deg.degrade_image(img)
    cv2.imwrite('initial_test_output.jpeg', deg_img)