import base64
import json
import io
import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

# App modules
from modules.noise import add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise
from modules.filters import (
    apply_gaussian_blur,
    apply_median_filter,
    apply_bilateral_filter,
    apply_non_local_means_denoising,
)
from modules.enhancement import (
    apply_hist_equalization,
    apply_clahe,
    apply_contrast_stretching,
    apply_sharpening,
)
from modules.metrics import calculate_mse, calculate_psnr

def cv2_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if len(image_bgr.shape) == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def cv2_to_base64(image: np.ndarray) -> str:
    """Encode an image as a base64 string to be sent over JSON."""
    if image is None: return ""
    encode_ext = '.jpg'
    # Use JPG for speed or PNG if preferred
    image_bgr = image
    if len(image.shape) == 3 and image.shape[2] == 3:
        # cv2 imencode expects BGR. If we've maintained BGR through pipeline, then just encode.
        pass
    success, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success: return ""
    return base64.b64encode(encoded.tobytes()).decode("utf-8")

def auto_resize(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    new_width = int(width * scale)
    new_height = int(height * scale)
    if scale < 1.0:
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    return image

def add_noise_pipeline(image: np.ndarray, params: dict) -> np.ndarray:
    ntype = params.get("noise_type", "None")
    if ntype == "Gaussian":
        return add_gaussian_noise(image, mean=params.get("gaussian_mean", 0), std=params.get("gaussian_std", 25))
    elif ntype == "Salt & Pepper":
        return add_salt_pepper_noise(image, amount=params.get("sp_amount", 0.02), salt_vs_pepper=params.get("sp_salt_vs_pepper", 0.5))
    elif ntype == "Speckle":
        return add_speckle_noise(image, var=params.get("speckle_var", 0.05))
    return image.copy()

def denoise_pipeline(image: np.ndarray, params: dict) -> np.ndarray:
    ftype = params.get("filter_name", "None")
    if ftype == "Gaussian Blur":
        return apply_gaussian_blur(image, ksize=params.get("gaussian_ksize", 5), sigma=params.get("gaussian_sigma", 1.0))
    elif ftype == "Median Filter":
        return apply_median_filter(image, ksize=params.get("median_ksize", 5))
    elif ftype == "Bilateral Filter":
        return apply_bilateral_filter(image, d=params.get("bilateral_d", 9), sigma_color=params.get("bilateral_sigma_color", 75), sigma_space=params.get("bilateral_sigma_space", 75))
    elif ftype == "Non-local Means (NLM)":
        return apply_non_local_means_denoising(image, h=params.get("nlm_h", 10.0), template_window_size=params.get("nlm_template_window", 7), search_window_size=params.get("nlm_search_window", 21))
    return image.copy()

def enhancement_pipeline(image: np.ndarray, params: dict) -> np.ndarray:
    etype = params.get("enhancement_name", "None")
    if etype == "Histogram Equalization":
        return apply_hist_equalization(image)
    if etype == "CLAHE":
        return apply_clahe(image, clip_limit=params.get("clahe_clip_limit", 2.0), tile_grid_size=params.get("clahe_tile_grid_size", 8))
    if etype == "Contrast Stretching":
        return apply_contrast_stretching(image, low_perc=params.get("contrast_low_perc", 2.0), high_perc=params.get("contrast_high_perc", 98.0))
    if etype == "Sharpening":
        return apply_sharpening(image, amount=params.get("sharpening_amount", 1.0))
    return image.copy()

app = FastAPI(title="DIP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse('static/index.html')


@app.post("/api/process")
async def process_image(file: UploadFile = File(...), data: str = Form(...)):
    try:
        params = json.loads(data)
        
        file_bytes = await file.read()
        file_array = np.frombuffer(file_bytes, np.uint8)
        image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})
            
        image_bgr = auto_resize(image_bgr)
        image_rgb = cv2_to_rgb(image_bgr)
        
        # Pipelines operate on BGR internally for traditional CV functions
        noisy_bgr = add_noise_pipeline(image_bgr, params)
        denoised_bgr = denoise_pipeline(noisy_bgr, params)
        enhanced_bgr = enhancement_pipeline(denoised_bgr, params)
        
        # Convert to RGB for metrics
        noisy_rgb = cv2_to_rgb(noisy_bgr)
        denoised_rgb = cv2_to_rgb(denoised_bgr)
        enhanced_rgb = cv2_to_rgb(enhanced_bgr)
        
        mse_noisy = calculate_mse(image_rgb, noisy_rgb)
        psnr_noisy = calculate_psnr(image_rgb, noisy_rgb)
        mse_denoised = calculate_mse(image_rgb, denoised_rgb)
        psnr_denoised = calculate_psnr(image_rgb, denoised_rgb)
        
        response = {
            "images": {
                "original": f"data:image/jpeg;base64,{cv2_to_base64(image_bgr)}",
                "noisy": f"data:image/jpeg;base64,{cv2_to_base64(noisy_bgr)}",
                "denoised": f"data:image/jpeg;base64,{cv2_to_base64(denoised_bgr)}",
                "enhanced": f"data:image/jpeg;base64,{cv2_to_base64(enhanced_bgr)}"
            },
            "metrics": {
                "mse_noisy" : float(mse_noisy) if mse_noisy != float("inf") else 0,
                "psnr_noisy": float(psnr_noisy) if psnr_noisy != float("inf") else 0,
                "mse_denoised": float(mse_denoised) if mse_denoised != float("inf") else 0,
                "psnr_denoised": float(psnr_denoised) if psnr_denoised != float("inf") else 0
            }
        }
        return JSONResponse(content=response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
