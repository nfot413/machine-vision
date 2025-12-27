import cv2
import numpy as np
from PIL import Image
import os

# ==========================================
#               配置区域
# ==========================================
CONFIG = {
    "input_path": "photo3.jpg",
    "save_dir": "final_mnist_dataset_fixed",  # 修改保存路径以示区别
    "rows": 13,
    "cols": 9,
    "canvas_size": 28,
    "padding": 4,  # [修改] 增加留白，让字符不要顶格
    "binary_thresh": 140
}


# ==========================================
#        第一部分：透视变换 (提取方格)
# ==========================================
# ... (这部分代码保持不变，为了节省篇幅，折叠) ...
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def extract_grid_image(image_path):
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"图片读取失败: {image_path}")
    img_original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise Exception("未检测到任何轮廓")
    max_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    if len(approx) == 4:
        print("[Step 1] 检测到四边形表格，正在拉平...")
        return four_point_transform(img_original, approx.reshape(4, 2))
    else:
        print("[Step 1] 未检测到完美四边形，执行降级裁剪...")
        x, y, w, h = cv2.boundingRect(max_contour)
        return img_original[y:y + h, x:x + w]


# ==========================================
#        第二部分：去除表格线
# ==========================================
def remove_grid_lines(image_bgr):
    print("[Step 2] 正在去除表格边框线...")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 稍微加大一点核，保证线条去得更干净
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    grid_lines = cv2.add(horizontal_lines, vertical_lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_lines_dilated = cv2.dilate(grid_lines, kernel, iterations=1)

    binary_clean = binary.copy()
    binary_clean[grid_lines_dilated > 0] = 0
    return binary_clean


# ==========================================
#        第三部分：切割并转MNIST格式 (重点修改)
# ==========================================

def slice_and_save_mnist(binary_img, output_dir):
    print(f"[Step 3] 正在切割并生成MNIST图片，目标文件夹: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    rows = CONFIG["rows"]
    cols = CONFIG["cols"]
    target_size = CONFIG["canvas_size"] - 2 * CONFIG["padding"]

    img_h, img_w = binary_img.shape
    cell_w = img_w // cols
    cell_h = img_h // rows

    valid_count = 0

    for row_idx in range(rows):
        y_start = row_idx * cell_h
        y_end = (row_idx + 1) * cell_h if row_idx < rows - 1 else img_h

        for col_idx in range(cols):
            x_start = col_idx * cell_w
            x_end = (col_idx + 1) * cell_w if col_idx < cols - 1 else img_w

            # 【改进点1】物理内缩：
            # 切割 ROI 时，四周各切掉 3-5 像素，物理去除边缘残留的黑线
            # 这能解决 90% 的“首字符变小”问题
            margin = 4
            if (y_end - y_start) > 2 * margin and (x_end - x_start) > 2 * margin:
                cell_roi = binary_img[y_start + margin: y_end - margin, x_start + margin: x_end - margin]
            else:
                cell_roi = binary_img[y_start:y_end, x_start:x_end]

            if cv2.countNonZero(cell_roi) < 10:
                continue

            # 【改进点2】智能轮廓筛选：
            # 不再粗暴地把所有白点当成字符，而是找到最大的那个连通块
            cnts, _ = cv2.findContours(cell_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue

            # 过滤掉面积太小的噪点（比如面积小于15像素的灰尘）
            valid_cnts = [c for c in cnts if cv2.contourArea(c) > 15]

            if not valid_cnts:
                continue

            # 将所有有效轮廓合并，计算总的包围盒 (应对断裂字符如 'i', '5')
            all_points = np.vstack(valid_cnts)
            x, y, w, h = cv2.boundingRect(all_points)

            # 提取纯净的字符区域
            char_crop = cell_roi[y:y + h, x:x + w]

            # --- 标准化为 MNIST ---
            char_h, char_w = char_crop.shape
            if char_h == 0 or char_w == 0: continue

            # 计算缩放比例
            scale = target_size / max(char_h, char_w)
            new_h = int(char_h * scale)
            new_w = int(char_w * scale)

            # INTER_AREA 插值效果更平滑
            resized_char = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            mnist_canvas = np.zeros((CONFIG["canvas_size"], CONFIG["canvas_size"]), dtype=np.uint8)
            y_offset = (CONFIG["canvas_size"] - new_h) // 2
            x_offset = (CONFIG["canvas_size"] - new_w) // 2

            # 只有在尺寸匹配时才复制，防止边缘溢出报错
            try:
                mnist_canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_char
            except:
                continue

            save_name = f"row{row_idx + 1}_col{col_idx + 1}_mnist_{valid_count}.png"
            Image.fromarray(mnist_canvas).save(os.path.join(output_dir, save_name))
            valid_count += 1

    print(f"处理完毕！共生成 {valid_count} 张图片。")


# ==========================================
#               主程序入口
# ==========================================
if __name__ == "__main__":
    try:
        # 1. 提取并拉平方格
        warped_grid = extract_grid_image(CONFIG["input_path"])
        # 2. 去除表格线
        cleaned_binary = remove_grid_lines(warped_grid)
        # 3. 切割并保存 (修复版)
        slice_and_save_mnist(cleaned_binary, CONFIG["save_dir"])

    except Exception as e:
        print(f"发生错误: {e}")