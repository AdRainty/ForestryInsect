# -*- coding: utf-8 -*-
  
import base64
import os
import json
import re
import xlrd, xlwt


def toBase64():
    insect = {
        'Drosicha corpulenta': "草履蚧 (Drosicha corpulenta (Kuwana))",
        'Erthesina fullo': "麻皮蝽 (Erthesina fullo (Thunberg))",
        'Anoplophora chinensis': "星天牛 (Anoplophora chinensis Forster,1771)",
        'Chalcophora japonica': "日本脊吉丁 (Chalcophora japonica(Gory))",
        'Apriona germar': "桑天牛 (Apriona germari(Hope))",
        'plagiodera versicolora': "柳蓝叶甲 (plagiodera versicolora )",
        'Monochamus alternatus Hope': "松墨天牛 (Monochamus alternatus Hope, 1842)",
        'Cnidocampa flavescens': "黄刺蛾 (Cnidocampa flavescens（Walker）)",
        'Latoria consocia Walker': "褐边绿刺蛾 (Latoria consocia Walker)",
        'Psilogramma menephron': "霜天蛾 (Psilogramma menephron(Gramer.))",
        'Hyphantria cunea': "美国白蛾 (Hyphantria cunea (Drury))",
        'Sericinus montelus Grey': "丝带凤蝶 (Sericinus montelus Grey)",
        'Spilarctia subcarnea': "人纹污灯蛾 (Spilarctia subcarnea (Walker))",
        'Micromelalopha troglodyta': "杨小舟蛾 (Micromelalopha troglodyta (Graeser))",
        'Clostera anachoreta': "杨扇舟蛾 (Clostera anachoreta (Denis et Schiffermüller, 1775))"
    }
    dir_path = os.path.dirname(os.path.abspath(__file__))
    json_path = dir_path + "\\detect_img\\output\\"
    with open(json_path + 'result.json', 'r') as f:
        load_dict = json.load(f)
        item_dir = load_dict[0]
        category = item_dir["name"]
        mesLs = getMessage(insect[category])
        allName = mesLs[0]
        chineseName = re.match(r'(.*?)\((.*?)\)(.*?)', allName).group(1).strip()
        englishName = allName.replace(chineseName, '')[1: -1]
        item_dir["name"] = chineseName
        item_dir["english"] = englishName
        item_dir["category"] = mesLs[1]
        item_dir["introduce"] = mesLs[2]

    print(item_dir)
    return item_dir


"""
def writeJson():

    insect = {
        'Drosicha corpulenta': "草履蚧 (Drosicha corpulenta (Kuwana))",
        'Erthesina fullo': "麻皮蝽 (Erthesina fullo (Thunberg))",
        'Anoplophora chinensis': "星天牛 (Anoplophora chinensis Forster,1771)",
        'Chalcophora japonica': "日本脊吉丁 (Chalcophora japonica(Gory))",
        'Apriona germar': "桑天牛 (Apriona germari(Hope))",
        'plagiodera versicolora': "柳蓝叶甲 (plagiodera versicolora )",
        'Monochamus alternatus Hope': "松墨天牛 (Monochamus alternatus Hope, 1842)",
        'Cnidocampa flavescens': "黄刺蛾 (Cnidocampa flavescens（Walker）)",
        'Latoria consocia Walker': "褐边绿刺蛾 (Latoria consocia Walker)",
        'Psilogramma menephron': "霜天蛾 (Psilogramma menephron(Gramer.))",
        'Hyphantria cunea': "美国白蛾 (Hyphantria cunea (Drury))",
        'Sericinus montelus Grey': "丝带凤蝶 (Sericinus montelus Grey)",
        'Spilarctia subcarnea': "人纹污灯蛾 (Spilarctia subcarnea (Walker))",
        'Micromelalopha troglodyta': "杨小舟蛾 (Micromelalopha troglodyta (Graeser))",
        'Clostera anachoreta': "杨扇舟蛾 (Clostera anachoreta (Denis et Schiffermüller, 1775))"
    }
    dir_path = os.path.dirname(os.path.abspath(__file__))
    json_path = dir_path + "\\detect_img\\output\\"
    with open(json_path + 'result.json', 'r') as f:
        load_dict = json.load(f)
        item_dir = load_dict[0]
        category = item_dir["name"]
        mesLs = getMessage(insect[category])
        item_dir["name"] = mesLs[0]
        item_dir["category"] = mesLs[1]
        item_dir["introduce"] = mesLs[2]

        print(mesLs)

    print("Return Success!")
    return item_dir
"""


def deBase64(base64_data):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    img_path = dir_path + "\\detect_img\\input\\"
    print("Load img...")
    imgData = base64.b64decode(base64_data)
    file = open(img_path + 'result.jpg', 'wb')
    file.write(imgData)
    file.close()
    print("Img has been loaded!")


# 从CSV中提取数据
def getMessage(data):
    dataXlsx = xlrd.open_workbook('pest.xls')
    table = dataXlsx.sheet_by_name(u'Sheet1')  # 通过名称获取
    for i in range(table.nrows):
        lsMs = table.row_values(i)
        if data == lsMs[0]:
            return table.row_values(i)


if __name__ == '__main__':
    toBase64()
