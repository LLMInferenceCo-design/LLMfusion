import sys
import os
import copy
import pandas as pd
# 获取当前工作目录
# if __name__ == "__main__":
#     current_dir = os.getcwd()
#     print("当前目录:", current_dir)

#     # 更改到上两级目录
#     parent_dir = os.path.dirname(os.path.dirname(current_dir))
#     os.chdir(parent_dir)
#     sys.path.append(parent_dir)

#     # 验证当前工作目录
#     new_dir = os.getcwd()
#     print("更改后的目录:", new_dir)
sys.path.append("/root/paper/LLMfusion")
os.chdir("/root/paper/LLMfusion")
def save_to_xlsx(csv_file_path, sheet_name, results):
    """将数据保存到Excel文件"""
    filtered_data = [(index, values) for index, _, values in results]
    # 创建一个空的 DataFrame
    df = pd.DataFrame()

    # 遍历过滤后的数据
    for index, values in filtered_data:
        # 把每个子字典转换为 DataFrame 的一行，并设置索引
        row = pd.DataFrame(values, index=[index])
        # 将行合并到主 DataFrame 中
        df = pd.concat([df, row], axis=0)

    # 检查文件是否存在
    if os.path.exists(csv_file_path):
        # 读取 Excel 文件
        excel_file = pd.ExcelFile(csv_file_path)
        # 获取所有表名
        sheet_names = excel_file.sheet_names
        with pd.ExcelWriter(csv_file_path, mode='a', if_sheet_exists='replace') as writer:
            # 如果工作表存在，先删除
            if sheet_name in sheet_names:
                df.to_excel(writer, sheet_name=sheet_name)
            else:
                df.to_excel(writer, sheet_name=sheet_name)
    else:
        # 文件不存在，直接写入
        df.to_excel(csv_file_path, sheet_name=sheet_name)


results = (("A", 100, {"latency": 10, "test":20}), ("B", 100, {"latency": 20, "test":30}))
save_to_xlsx("../ae/test.xlsx", "Sheet1", results)