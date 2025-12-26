import cv2
import numpy as np


def remove_grid_lines(image_path, output_path=None, show=False):
    """
    去除表格边框线，只保留文字内容
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径（可选，如果为None则不保存）
    :param show: 是否显示处理结果
    :return: 处理后的图像（numpy array）
    """
    # 1. 读取图像并转为灰度图
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 二值化（黑字白底）
    # 手写字较浅，使用反转阈值更稳健
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 分别检测水平线和垂直线
    # 水平线：使用长横向内核
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (46, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # 垂直线：使用长纵向内核
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 46))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # 4. 合并检测到的表格线
    grid_lines = cv2.add(horizontal_lines, vertical_lines)

    # 5. 将检测到的线区域稍微膨胀，使去除更彻底
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_lines_dilated = cv2.dilate(grid_lines, kernel, iterations=1)

    # 6. 从原二值图中去除表格线（置为背景）
    binary_clean = binary.copy()
    binary_clean[grid_lines_dilated > 0] = 0

    # 7. 反转回正常黑字白底
    result = cv2.bitwise_not(binary_clean)

    # 8. 可选：轻微去噪
    result = cv2.medianBlur(result, 3)

    # 9. 如果需要彩色背景，可以把结果贴回原图
    # 这里直接返回灰度图（更干净），如果想保留原背景色可取消注释下面代码
    # result_color = img.copy()
    # result_color[result == 0] = 255  # 把文字区域外设为白色

    if output_path:
        cv2.imwrite(output_path, result)


    return result


# ==================== 使用示例 ====================
if __name__ == "__main__":
    input_image = "cropped_warped_grid.jpg"  # 替换成你的图片文件名
    output_image = "clean_text.jpg"  # 处理后保存的文件名

    cleaned_img = remove_grid_lines(input_image, output_path=output_image, show=True)
    print("表格线已去除，干净文字图像已保存为:", output_image)