import cv2
import numpy as np
import os
import glob

def add_gaussian_noise(img: np.ndarray, mean: float = 0, var: float = 0.005) -> np.ndarray:
    """ 
    Thêm nhiễu hạt (Gaussian Noise) - Phù hợp để kiểm thử thuật toán Gaussian Blur 
    """
    image_normalized = img / 255.0
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image_normalized.shape)
    noisy = image_normalized + gauss
    noisy = np.clip(noisy, 0, 1) # Giữ giá trị trong khoảng [0, 1]
    return (noisy * 255).astype(np.uint8)

def add_scratches_and_stains_with_mask(img: np.ndarray, num_scratches: int = 5, num_stains: int = 2):
    """ 
    Thêm vết xước và vết ố, đồng thời lưu lại MASK chuẩn 
    - Phù hợp để kiểm thử thuật toán Inpainting 
    """
    corrupted = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8) # Mask nền đen (0) chứa vị trí cần khôi phục (255)

    # 1. Vẽ vết xước (Scratches)
    for _ in range(num_scratches):
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        thickness = np.random.randint(1, 3)
        color = (255, 255, 255) # Vết xước trắng
        
        cv2.line(corrupted, pt1, pt2, color, thickness)
        cv2.line(mask, pt1, pt2, 255, thickness) # Lưu lại vị trí xước lên mask

    # 2. Tạo vết ố (Stains)
    for _ in range(num_stains):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        axes = (np.random.randint(10, 30), np.random.randint(10, 30))
        angle = np.random.randint(0, 360)
        stain_color = (120, 160, 180) # Màu ố mốc
        
        cv2.ellipse(corrupted, center, axes, angle, 0, 360, stain_color, -1)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1) # Đưa vết ố vào mask để Inpaint

    return corrupted, mask

def create_corrupted_dataset(input_dir: str, output_dir: str, mask_dir: str):
    """
    Tạo bộ dữ liệu thử nghiệm chuẩn cho Gaussian Blur và Inpainting
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    extensions = ('*.jpg', '*.png', '*.jpeg')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if len(image_paths) == 0:
        print(f"❌ Không tìm thấy ảnh nào trong thư mục '{input_dir}'!")
        return

    print(f"🚀 Đang khởi tạo dữ liệu lỗi phù hợp cho Gaussian Filter & Inpainting ({len(image_paths)} ảnh)...")

    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None: 
            continue

        # Bước 1: Thêm vết xước & vết ố + Xuất Mask (để chữa bằng Inpainting)
        corrupted_img, mask = add_scratches_and_stains_with_mask(img, num_scratches=6, num_stains=2)
        
        # Bước 2: Thêm nhiễu Gaussian hạt (để chữa bằng Gaussian Blur)
        corrupted_img = add_gaussian_noise(corrupted_img, var=0.003)

        # Lưu kết quả
        cv2.imwrite(os.path.join(output_dir, filename), corrupted_img)
        cv2.imwrite(os.path.join(mask_dir, filename), mask)

    print(f"✅ Hoàn thành! Dataset ảnh hỏng nằm tại '{output_dir}', Mask tương ứng nằm tại '{mask_dir}'.")

if __name__ == "__main__":
    create_corrupted_dataset(
        input_dir="dataset/images_clean",
        output_dir="dataset/images_corrupted",
        mask_dir="dataset/images_masks"
    )