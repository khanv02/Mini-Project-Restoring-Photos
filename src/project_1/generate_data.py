import cv2
import numpy as np
import os
import glob

def add_salt_and_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    """ Thêm nhiễu muốm tiêu (Salt & Pepper) """
    noisy = img.copy()
    # Salt (Điểm trắng)
    num_salt = int(salt_prob * img.size / 3)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = [255, 255, 255]

    # Pepper (Điểm đen)
    num_pepper = int(pepper_prob * img.size / 3)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = [0, 0, 0]
    return noisy

def add_gaussian_noise(img, mean=0, var=0.005):
    """ Thêm nhiễu hạt (Gaussian Noise) """
    image_normalized = img / 255.0
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image_normalized.shape)
    noisy = image_normalized + gauss
    noisy = np.clip(noisy, 0, 1) # Giữ giá trị [0, 1]
    return (noisy * 255).astype(np.uint8)

def add_random_scratches_and_stains(img, num_scratches=5, num_stains=2):
    """ Thêm vết xước và vết ố vàng/mốc của ảnh cũ """
    corrupted = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8) # Mask đen chứa vị trí xước

    # 1. Vẽ vết xước (Scratches)
    for _ in range(num_scratches):
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        thickness = np.random.randint(1, 3) # Độ rộng đường xước
        color = (np.random.randint(220, 255), np.random.randint(220, 255), np.random.randint(220, 255))
        
        cv2.line(corrupted, pt1, pt2, color, thickness)
        cv2.line(mask, pt1, pt2, 255, thickness) # Lưu lại vị trí xước trên mask

    # 2. Tạo vết ố vàng/mốc nhẹ (Stains)
    for _ in range(num_stains):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        axes = (np.random.randint(10, 40), np.random.randint(10, 40))
        angle = np.random.randint(0, 360)
        # Màu ố ngả vàng/mốc nhẹ
        stain_color = (np.random.randint(100, 150), np.random.randint(140, 190), np.random.randint(160, 210))
        
        overlay = corrupted.copy()
        cv2.ellipse(overlay, center, axes, angle, 0, 360, stain_color, -1)
        # Hòa trộn mờ nhẹ vết ố
        cv2.addWeighted(overlay, 0.25, corrupted, 0.75, 0, corrupted)

    return corrupted, mask

def create_corrupted_dataset(input_dir, output_dir, mask_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    extensions = ('*.jpg', '*.png', '*.jpeg')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if len(image_paths) == 0:
        print(f"❌ Không tìm thấy ảnh nào trong thư mục '{input_dir}'!")
        return

    print(f"🚀 Đang tạo ảnh lỗi cho {len(image_paths)} ảnh...")

    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None: continue

        # Bước 1: Thêm xước & vết ố ảnh cũ
        corrupted_img, mask = add_random_scratches_and_stains(img, num_scratches=6, num_stains=2)
        
        # Bước 2: Thêm nhiễu muốm tiêu (Salt & Pepper)
        corrupted_img = add_salt_and_pepper_noise(corrupted_img, salt_prob=0.008, pepper_prob=0.008)

        # Bước 3: Thêm nhiễu hạt (Gaussian Noise)
        corrupted_img = add_gaussian_noise(corrupted_img, var=0.003)

        # Bước 4: Thêm nhiễu mờ (Gaussian Blur) nhẹ
        corrupted_img = cv2.GaussianBlur(corrupted_img, (3, 3), 0.5)

        # Lưu file
        cv2.imwrite(os.path.join(output_dir, filename), corrupted_img)
        cv2.imwrite(os.path.join(mask_dir, filename), mask)

    print(f"✅ Hoàn thành! 100 ảnh lỗi nằm ở '{output_dir}', mask xước ở '{mask_dir}'.")

# Chạy tạo dataset
if __name__ == "__main__":
    create_corrupted_dataset(
        input_dir="dataset/images_clean",
        output_dir="dataset/images_corrupted",
        mask_dir="dataset/images_masks"
    )