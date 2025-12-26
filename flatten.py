import cv2
import numpy as np


# --------------- 工具函数：排序轮廓的四个顶点（左上、右上、右下、左下）---------------
def order_points(pts):
    # 初始化坐标顺序：左上、右上、右下、左下
    rect = np.zeros((4, 2), dtype="float32")

    # 左上点：x+y最小；右下点：x+y最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上点：x-y最小；左下点：x-y最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# --------------- 工具函数：透视变换（拉平核心）---------------
def four_point_transform(image, pts):
    # 获取排序后的顶点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算目标矩形的宽度（取水平两边的最大长度，保证拉平后无变形）
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算目标矩形的高度（取垂直两边的最大长度）
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建目标矩形的四个顶点（正长方形）
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 计算透视变换矩阵并执行变换（拉平）
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


# --------------- 主流程：提取方格+去干扰+拉平 ---------------
if __name__ == "__main__":
    # 1. 读取图片（替换为你的图片路径）
    img = cv2.imread("photo3.jpg")
    if img is None:
        print("图片读取失败，请检查路径！")
        exit()
    # 备份原图用于显示
    img_original = img.copy()

    # 2. 预处理：提升轮廓检测效果（去噪+二值化）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊去噪（减少干扰轮廓）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 自适应二值化（适配不同光照的图片，比固定阈值更鲁棒）
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 3. 查找轮廓（只找外部轮廓，减少内部干扰）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 筛选面积最大的轮廓（对应方格区域）
    max_contour = max(contours, key=cv2.contourArea)

    # 4. 多边形逼近：获取方格的四个顶点（核心：从轮廓中提取四边形）
    # epsilon为轮廓周长的百分比，越小逼近越精确
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # 检查是否提取到四个顶点（方格应为四边形）
    if len(approx) == 4:
        # 转换顶点格式
        pts = approx.reshape(4, 2)
        # 5. 透视变换拉平方格
        warped = four_point_transform(img_original, pts)
        # 6. 保存结果
        cv2.imwrite("cropped_warped_grid.jpg", warped)
        print("方格已提取、拉平并保存为 cropped_warped_grid.jpg")

        # （可选）显示过程结果
        # 绘制轮廓和顶点
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        for (x, y) in pts:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

    else:
        print("未检测到方格的四个顶点，可能是图片清晰度不足或轮廓提取失败！")
        # 降级方案：使用包围矩形裁剪（无拉平）
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped = img_original[y:y + h, x:x + w]
        cv2.imwrite("cropped_grid.jpg", cropped)
        print("已执行降级裁剪，保存为 cropped_grid.jpg")