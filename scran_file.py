import os
from pathlib import Path


def find_files(skip_dirs=None):
    """
    查找当前目录及其子目录下的所有文件路径，跳过指定目录

    :param skip_dirs: 需要跳过的目录名列表，例如 ['.git', 'node_modules']
    :return: 包含所有文件路径的列表
    """
    if skip_dirs is None:
        skip_dirs = []

    file_list = []
    for root, dirs, files in os.walk(Path.cwd()):  # 使用pathlib获取规范路径
        # 实时过滤要遍历的目录（原地修改dirs列表）
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        file_list.extend(str(Path(root) / file) for file in files)

    return file_list


if __name__ == "__main__":
    # 示例用法：跳过.git和temp目录
    skipped_directories = ['logs', 'outputs','docs']
    all_files = find_files(skip_dirs=skipped_directories)

    # 优雅打印结果
    print(f"Found {len(all_files)} files (skip dirs: {skipped_directories}):")
    print("\n".join(all_files))
