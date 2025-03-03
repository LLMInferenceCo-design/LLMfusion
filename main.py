# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import os

config = "./tem/data.cfg"
os.makedirs(os.path.dirname(config), exist_ok=True)
with open(config, "w") as f:
    f.writelines("[general]\n")
    f.writelines("run_name = systolic_array\n\n")
    f.writelines("[architecture_presets]\n")
    f.writelines("ArrayHeight:    " + str(16) + "\n")
    f.writelines("ArrayWidth:     " + str(16) + "\n")
    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
    f.writelines("IfmapOffset:    0\n")
    f.writelines("FilterOffset:   10000000\n")
    f.writelines("OfmapOffset:    20000000\n")
    f.writelines("Dataflow : " + "os" + "\n")
    f.writelines("Bandwidth : " + "100" + "\n")
    f.writelines("MemoryBanks: 1\n\n")
    f.writelines("[run_presets]\n")
    f.writelines("InterfaceBandwidth: CALC\n")