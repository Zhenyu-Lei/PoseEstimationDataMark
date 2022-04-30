import tkinter as tk
from PIL import ImageTk
import tkinter.filedialog
import os
import pandas
import re
import json
from tkinter import messagebox
import numpy

photo = None
photoH = 0
photoW = 0
fileMark = 0
FileList = []
rootPath = ""
Flag = False
PaintH = 600
PaintW = 800
csvData = []
df = pandas.core.frame.DataFrame
circleSize = 4
color = ["#FF8C00", "#6495ED", "#FFF0F5", "#00FF7F", "#8B008B", "#00BFFF", "#00BFFF", "#778899", "#4B0082", "#40E0D0",
         "#FFE4B5", "#FFE4B5", "#C71585"]
nowNum = -1
PartList = ["鼻子", "左肩", "右肩", "左肘", "右肘", "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"]
PointList = []
circleId = {}
hasEmphasize = []
clickX = 0
clickY = 0
rePaintId = -1


def load_csv_file(root_file):
    global df
    df = pandas.read_csv(root_file, index_col='FileName')
    # pandas.core.frame.DataFrame 用 loc[索引].values访问


def myPaint(num, a, b, X, Y, outline, width, cv):
    if a == -1 and b == -1:
        # 不予显示
        return
    circleId[num] = cv.create_oval(X - circleSize, Y - circleSize, X + circleSize,
                                   Y + circleSize,
                                   fill=color[num], outline=outline, width=width)


# 加载文件
def load_folder(llb, cv):
    global FileList, fileMark, rootPath, Flag
    # 从本地选择一个文件，并返回文件的目录
    filename = tk.filedialog.askdirectory()
    fileMark = 0
    Flag = True
    for root, dirs, files in os.walk(filename, topdown=False):
        rootPath = root
        FileList = []
        for file in files:
            if re.match("(.*).jpg", file):
                FileList.append(file)
            elif re.match("(.*).csv", file):
                load_csv_file(root + "/" + file)
    do_list(0, cv, llb)
    if filename == '':
        llb.config(text='您没有选择任何文件')


def rePaint(num, cv, label):
    global rePaintId
    # 如果circleId包括这个num
    if circleId.__contains__(num):
        cv.delete(circleId[num])
        del [circleId[num]]
        # 删除这个需要repaint的点
        PointList[num * 2] = -1
        PointList[num * 2 + 1] = -1
    # 并且设置repaint的值
    rePaintId = num
    label.set("重绘:%s" % PartList[num])
    if hasEmphasize.__contains__(num):
        hasEmphasize.clear()


def emphasize(num, a, cv, labelText):
    global nowNum, rePaintId
    rePaintId = -1
    if num + a < 0 or num + a > 12:
        return
    # 先清除掉上一次加粗的点
    for i in hasEmphasize:
        if circleId.__contains__(i):
            cv.delete(circleId[i])
            del [circleId[i]]
            # if PointList[i * 2] == 0 and PointList[i * 2 + 1] == 0:
            #     continue
            PointX = PointList[i * 2] + (PaintW - photoW) / 2
            PointY = PointList[i * 2 + 1] + (PaintH - photoH) / 2
            myPaint(i, PointList[i * 2], PointList[i * 2 + 1], PointX, PointY, 'black', 2, cv)
            # circleId[i] = cv.create_oval(PointX - circleSize, PointY - circleSize, PointX + circleSize,
            #                              PointY + circleSize,
            #                              fill=color[i], outline='black', width=2)
    hasEmphasize.clear()
    # 如果a为0，直接尝试显示，否则向前或者向后计数直到越界或找到点
    if a == 0:
        if circleId.__contains__(num):
            cv.delete(circleId[num])
            del [circleId[num]]
            # 删除后重新进行绘制
            PointX = PointList[num * 2] + (PaintW - photoW) / 2
            PointY = PointList[num * 2 + 1] + (PaintH - photoH) / 2
            myPaint(num, PointList[num * 2], PointList[num * 2 + 1], PointX, PointY, 'red', 2, cv)
            # circleId[num] = cv.create_oval(PointX - 5, PointY - 5, PointX + 5, PointY + 5, fill=color[num],
            #                                outline='red',
            #                                width=2)
            nowNum = num
            labelText.set(PartList[num])
            hasEmphasize.append(num)
    else:
        tmp = a
        while not circleId.__contains__(num + tmp):
            if num + tmp < 0 or num + tmp > 12:
                return
            tmp += a
        num = num + tmp
        cv.delete(circleId[num])
        del [circleId[num]]
        # 删除后重新进行绘制
        PointX = PointList[num * 2] + (PaintW - photoW) / 2
        PointY = PointList[num * 2 + 1] + (PaintH - photoH) / 2
        myPaint(num, PointList[num * 2], PointList[num * 2 + 1], PointX, PointY, 'red', 2, cv)
        # circleId[num] = cv.create_oval(PointX - 5, PointY - 5, PointX + 5, PointY + 5, fill=color[num],
        #                                outline='red',
        #                                width=2)
        nowNum = num
        labelText.set(PartList[num])
        hasEmphasize.append(num)


def do_list(a, cv, lb):
    global fileMark, photo, photoW, photoH, PointList
    if 0 <= fileMark + a < len(FileList):
        # 删除当前的所有图形，重新绘制
        cv.delete("all")
        fileMark += a
        photo = ImageTk.PhotoImage(file=rootPath + '/' + FileList[fileMark])  # file：t图片路径
        lb.config(text=rootPath + '/' + FileList[fileMark])
        cv.create_image((PaintW - photo.width()) / 2, (PaintH - photo.height()) / 2, image=photo, anchor="nw")
        # 画所有的点
        PointList = df.loc[FileList[fileMark]].values
        photoH = PointList[27]
        photoW = PointList[26]
        for i in range(0, 26, 2):
            PointX = PointList[i] + (PaintW - photoW) / 2
            PointY = PointList[i + 1] + (PaintH - photoH) / 2
            #  print(PointX, PointY)
            myPaint(int(i / 2), PointList[i], PointList[i + 1], PointX, PointY, 'black', 2, cv)
            # circleId[int(i / 2)] = cv.create_oval(PointX - circleSize, PointY - circleSize, PointX + circleSize,
            #                                       PointY + circleSize,
            #                                       fill=color[int(i / 2)], outline='black', width=2)
        return True
    return False


# 上一张按钮绑定的回调事件
def callback1(lb, cv, label):
    global nowNum
    if not Flag:
        print("需要选择文件夹")
    else:
        if do_list(-1, cv, lb):
            nowNum = -1
            label.set("未选择")
            print("前一张图片")


# 下一张按钮绑定的回调事件
def callback2(lb, cv, label):
    global nowNum
    if not Flag:
        print("需要选择文件夹")
    else:
        if do_list(1, cv, lb):
            nowNum = -1
            label.set("未选择")
            print("后一张图片")


# 重绘按钮绑定的回调事件
def callback3(event, cv, label):
    global rePaintId
    # 如果有点需要修改
    if rePaintId != -1:
        # 修改pointList里面的点
        PointList[rePaintId * 2] = (event.x - (PaintW - photoW) / 2)
        PointList[rePaintId * 2 + 1] = (event.y - (PaintH - photoH) / 2)
        # print(PointList)
        # 重绘该点
        PointX = event.x
        PointY = event.y
        myPaint(rePaintId, event.x, event.y, PointX, PointY, 'red', 2, cv)
        # circleId[rePaintId] = cv.create_oval(PointX - 5, PointY - 5, PointX + 5, PointY + 5, fill=color[rePaintId],
        #                                      outline='red', width=2)
        for i in hasEmphasize:
            if circleId.__contains__(i):
                cv.delete(circleId[i])
                del [circleId[i]]
                PointX = PointList[i * 2] + (PaintW - photoW) / 2
                PointY = PointList[i * 2 + 1] + (PaintH - photoH) / 2
                myPaint(i, PointList[i * 2], PointList[i * 2 + 1], PointX, PointY, 'black', 2, cv)
                # circleId[i] = cv.create_oval(PointX - circleSize, PointY - circleSize, PointX + circleSize,
                #                              PointY + circleSize, fill=color[i], outline='black', width=2)
        hasEmphasize.clear()
        hasEmphasize.append(rePaintId)
        label.set(PartList[rePaintId])
        rePaintId = -1


def insert(cv, path):
    global photo
    photo = ImageTk.PhotoImage(file=path)  # file：t图片路径
    # anchor是前面(0,0)点所在的位置，也就是定在(0,0 )这个点
    cv.create_image(10, 10, image=photo, anchor="nw")


def saveData():
    df.to_csv(rootPath + "/dataPreMark.csv")
    save_json()
    messagebox.showinfo("提示", "保存csv文件成功")


def calculateMargin(thePointList):
    ListX = []
    ListY = []
    for i in range(0, 26, 2):
        ListX.append(thePointList[i])
        ListY.append(thePointList[i + 1])
    return min(ListX), max(ListX), min(ListY), max(ListY)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):  # add this line
            return obj.tolist()  # add this line
        return json.JSONEncoder.default(self, obj)


def save_json():
    json_dict = {"images": [], "annotations": [], "categories": []}
    # 计算窗口中心位置
    fillList = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    factor = 1.20
    for file in FileList:
        ThePointList = df.loc[file].values
        MiX, MaX, MiY, MaY = calculateMargin(ThePointList)
        centerX = (MiX + MaX) / 2
        centerY = (MiY + MaY) / 2
        keyPointLast = []
        fileId = int(file[6:][:-4])
        if ThePointList[0] == -1 and ThePointList[1] == -1:
            fillList[0] = 0
        for i in range(2, 26, 2):
            keyPointLast += ThePointList[i:i + 2].tolist()
            if ThePointList[i] == -1 and ThePointList[i + 1] == -1:
                keyPointLast += [0]
            else:
                keyPointLast += [1]
        json_dict["images"].append({
            "id": fileId,
            "file_name": file,
            "width": ThePointList[26],
            "height": ThePointList[27]
        })
        json_dict["annotations"].append({
            "num_keypoints": 17,
            "iscrowd": 0,
            "image_id": fileId,
            "keypoints": ThePointList[0:2].tolist() + fillList + keyPointLast,
            "category_id": 1,
            "bbox": [max(0, centerX - factor * (centerX - MiX)), max(0, centerY - factor * (centerY - MiY)),
                     (MaX - MiX) * factor, (MaY - MiY) * factor],
            "id": 1,
            "area": ThePointList[26] * ThePointList[27]
        })
    json_dict["categories"].append({
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                      "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "skeleton": [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7]
        ]
    })
    with open(rootPath + "/my_keypoint_annotation.json", 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, cls=NpEncoder)


def main():
    # 创建一个主窗口对象
    window = tk.Tk()
    window.title("人体关键点标注工具")
    window.geometry('1000x700')
    label = tk.StringVar()
    # 将画布设为白色
    cv = tk.Canvas(window, bg='white', height=PaintH, width=PaintW)
    cv.bind("<Button-1>", lambda event: callback3(event, cv, label))
    # insert(cv, "D:/深度学习的人体姿态估计算法研究/HRNet/OpenSitUp-main/logo.png")

    lb = tk.Label(window, text='', bg='#87CEEB')
    lb.pack(fill="x")
    frm_1 = tkinter.Frame(window)
    # 当成功创建标签（文本）对象后，必须使用 pack 方法将其放置在主窗口内（pack 方法又称窗口布局管理器）
    preButton = tk.Button(frm_1, text="前一张图片", command=lambda: callback1(lb, cv, label), padx=10).pack()
    nextButton = tk.Button(frm_1, text="后一张图片", command=lambda: callback2(lb, cv, label), padx=10).pack()
    frm_1.pack(side="left")
    # 上一张图片
    window.bind(sequence="<Up>", func=lambda event: callback1(lb, cv, label))
    window.bind(sequence="<Down>", func=lambda event: callback2(lb, cv, label))
    # 移动关键点
    window.bind(sequence="<Left>", func=lambda event: emphasize(nowNum, - 1, cv, label))
    window.bind(sequence="<Right>", func=lambda event: emphasize(nowNum, + 1, cv, label))

    # 当你创建一个Label对象后，马上调用了pack方法，然后pack()返回None，所以接下来再执行config()会报错

    # lb.grid(row=0, column=1)
    # 如果command有参数，则需要加lambda，否则会先执行一遍回调函数
    btn = tk.Button(window, text='选择文件夹', command=lambda: load_folder(lb, cv))
    btn.place(x=0, y=0)
    # btn.grid(row=0, column=0)
    buttonSave = tk.Button(window, text='保存所有', command=lambda: saveData())
    buttonSave.place(x=75, y=0)
    label.set("未选择")
    tk.Label(window, textvariable=label, fg='red', height=4, width=16, font="font=('微软雅黑',45,'italic')",
             relief="sunken") \
        .place(x=435, y=565)

    frm_2 = tkinter.Frame(window)
    noesButton = tk.Button(frm_2, text="鼻子", command=lambda: emphasize(0, 0, cv, label)).grid(row=0, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(0, cv, label)).grid(row=0, column=1)
    leftShoulderButton = tk.Button(frm_2, text="左肩", command=lambda: emphasize(1, 0, cv, label)).grid(row=1, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(1, cv, label)).grid(row=1, column=1)
    rightShoulderButton = tk.Button(frm_2, text="右肩", command=lambda: emphasize(2, 0, cv, label)).grid(row=2, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(2, cv, label)).grid(row=2, column=1)
    leftElbowButton = tk.Button(frm_2, text="左肘", command=lambda: emphasize(3, 0, cv, label)).grid(row=3, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(3, cv, label)).grid(row=3, column=1)
    rightElbowButton = tk.Button(frm_2, text="右肘", command=lambda: emphasize(4, 0, cv, label)).grid(row=4, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(4, cv, label)).grid(row=4, column=1)
    leftWristButton = tk.Button(frm_2, text="左腕", command=lambda: emphasize(5, 0, cv, label)).grid(row=5, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(5, cv, label)).grid(row=5, column=1)
    rightWristButton = tk.Button(frm_2, text="右腕", command=lambda: emphasize(6, 0, cv, label)).grid(row=6, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(6, cv, label)).grid(row=6, column=1)
    leftHipButton = tk.Button(frm_2, text="左髋", command=lambda: emphasize(7, 0, cv, label)).grid(row=7, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(7, cv, label)).grid(row=7, column=1)
    rightHipButton = tk.Button(frm_2, text="右髋", command=lambda: emphasize(8, 0, cv, label)).grid(row=8, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(8, cv, label)).grid(row=8, column=1)
    leftKneeButton = tk.Button(frm_2, text="左膝", command=lambda: emphasize(9, 0, cv, label)).grid(row=9, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(9, cv, label)).grid(row=9, column=1)
    rightKneeButton = tk.Button(frm_2, text="右膝", command=lambda: emphasize(10, 0, cv, label)).grid(row=10, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(10, cv, label)).grid(row=10, column=1)
    leftAnkleButton = tk.Button(frm_2, text="左踝", command=lambda: emphasize(11, 0, cv, label)).grid(row=11, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(11, cv, label)).grid(row=11, column=1)
    rightAnkleButton = tk.Button(frm_2, text="右踝", command=lambda: emphasize(12, 0, cv, label)).grid(row=12, column=0)
    tk.Button(frm_2, text="重绘", command=lambda: rePaint(12, cv, label)).grid(row=12, column=1)
    frm_2.pack(side="right")

    cv.pack()
    # 调用mainloop()显示主窗口
    window.mainloop()


if __name__ == '__main__':
    main()
