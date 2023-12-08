# 打开1.txt文件以读取模式
with open('1.txt', 'r') as file1:
    # 读取1.txt文件的内容
    content = file1.readlines()

# 打开2.txt文件以写入模式
with open('2.txt', 'w') as file2:
    # 遍历1.txt中的每一行
    for line in content:
        # 使用逗号分割每一行，并选择第1和第2列
        columns = line.strip().split(',')
        column1 = columns[0]
        column2 = columns[1]

        # 将第1和第2列写入2.txt，并用逗号隔开
        file2.write(f"{column1},{column2}\n")

print("处理完成")
