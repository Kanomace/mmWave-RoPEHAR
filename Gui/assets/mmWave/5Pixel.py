import os
from PIL import Image
from openpyxl import Workbook

# 图像文件夹路径
folder_path = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\traindata\3tumble\1\pHistBytes_clustered_voxel_XOY'

# 获取文件夹中的图像文件列表
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
# 图像文件排序
image_files.sort(key=lambda x: int(x.split('_')[-2]))

# 创建Excel工作簿
workbook = Workbook()
sheet = workbook.active

# 遍历图像文件并将数据写入Excel文件
for i, image_file in enumerate(image_files):
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path).convert('L')  # 打开并转换为灰度图像
    pixel_values = list(image.getdata())

    # 在第一个图像时调整表头为图像的像素数量
    if i == 0:
        header = ['pixel' + str(i+1) for i in range(len(pixel_values))]
        sheet.append(header)

    # 根据图像的像素数量截取前相应数量的像素值
    pixel_values = pixel_values[:len(header)]
    sheet.append(pixel_values)

# 保存Excel文件
excel_file_path = os.path.join(folder_path, 'image_data.xlsx')
workbook.save(excel_file_path)

print('Excel文件已保存：', excel_file_path)