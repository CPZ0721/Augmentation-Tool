from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import glob
import os
import random
import time
import cv2
from pascal_voc_writer import Writer
import imgaug as ia
from imgaug import augmenters as iaa

import annotation as an
import random_rotation as rota
import resize as resize
import augment_window as ui
from xml.etree.ElementTree import ParseError
import xml.etree.ElementTree as ET
import numpy as np

class Main(QWidget, ui.Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_dirA.clicked.connect(self.DirA)
        self.btn_dirB.clicked.connect(self.DirB)
        self.btn_save.clicked.connect(self.Save)
        self.btn_augment.clicked.connect(self.Augment)
        self.checkBoxB_1.stateChanged.connect(self.changeset)
        self.imageDirA = None
        self.imageDirB = None
        self.savepath = None

    def DirA(self):
        # 選擇圖片A位置
        self.imageDirA = QFileDialog.getExistingDirectory(self, "選取文件夾")
        self.imageListA = glob.glob(os.path.join(self.imageDirA, '*.JPG'))
        self.xmlListA = glob.glob(os.path.join(self.imageDirA, '*.xml'))

        if len(self.imageListA) == 0:
            print('No .JPG images found in the specified dir!')
            return

    def DirB(self):
        # 選擇圖片B位置
        self.imageDirB = QFileDialog.getExistingDirectory(self, "選取文件夾")
        self.imageListB = glob.glob(os.path.join(self.imageDirB, '*.JPG'))
        self.xmlListB = glob.glob(os.path.join(self.imageDirB, '*.xml'))

        if len(self.imageListB) == 0:
            print('No .JPG images found in the specified dir!')
            return

    def Save(self):
        # 選擇儲存路徑資料夾
        self.savepath = QFileDialog.getExistingDirectory(self, "選取儲存資料夾")

    #設定B設定，需勾選改變影像大小，才可進一步勾選維持比例
    def changeset(self):
        if self.checkBoxB_1.checkState() == Qt.Checked:
            self.checkBoxB_1_1.setCheckable(True)
        elif self.checkBoxB_1.checkState() == Qt.Unchecked:
            self.checkBoxB_1_1.setCheckable(False)

    def process_B(self):
        # 改變影像大小
        if self.checkBoxB_1.checkState() == 2:
            # 保持比例
            if self.checkBoxB_1_1.checkState() == 2:
                ratio = random.uniform(0.6, 1.4)
                seq = iaa.Sequential(
                    [iaa.Resize({"height": int(self.h_B * ratio), "width": int(self.w_B * ratio)})])
                imageB_aug = seq.augment_images([self.image_B])[0]
                bbs_aug = seq.augment_bounding_boxes(
                    [self.bbs])[0].remove_out_of_image().cut_out_of_image()
            # 不保持比例
            elif self.checkBoxB_1_1.checkState() == 0:
                list_height = [100, 200, 300, 400, 500]
                list_width = [100, 200, 300, 400, 500]
                h = random.choice(list_height)
                w = random.choice(list_width)
                seq = iaa.Sequential([iaa.Resize({"height": h, "width": w})])
                imageB_aug = seq.augment_images([self.image_B])[0]
                bbs_aug = seq.augment_bounding_boxes(
                    [self.bbs])[0].remove_out_of_image().cut_out_of_image()

        # 不改變影像大小
        elif self.checkBoxB_1.checkState() == 0:
            imageB_aug = self.image_B
            bbs_aug = self.bbs

        # B的augment被勾選
        if self.checkBoxB_2.checkState() == 2:
            seq_det = self.seq_arg.to_deterministic()
            imageB_aug = seq_det.augment_images([imageB_aug])[0]
            bbs_aug = seq_det.augment_bounding_boxes(
                [bbs_aug])[0].remove_out_of_image().cut_out_of_image()

        return imageB_aug, bbs_aug

    #檢查B影像是否與A影像的所有label重疊70%以上
    def coincide(self, Label_A, global_y, global_x, img, bbs):
        #設定閾值為70%
        crop_threshold = 0.5

        self.img = img
        self.b_bbs = bbs
        self.y = global_y
        self.x = global_x

        # B照片的四點座標
        x01 = global_x
        y01 = global_y
        x02 = global_x + self.img.shape[1]
        y02 = global_y + self.img.shape[0]

        self.Error = 0

        for aa in Label_A:
            #A的label的四點座標
            x11 = int(aa.x1)
            y11 = int(aa.y1)
            x12 = int(aa.x2)
            y12 = int(aa.y2)
            col = min(x02, x12) - max(x01, x11)
            row = min(y02, y12) - max(y01, y11)
            #判斷重疊率
            intersection = col * row
            area = (x12 - x11) * (y12 - y11)
            self.coincide_prop = intersection / area
            #如果col 或row為負值，代表不重疊
            if col < 0 or row < 0:
                self.coincide_prop = 0

            #超過設定比率，就重新設定座標
            if self.coincide_prop > crop_threshold :
                self.Error = 1
                break

        #如果重新設定座標超過20次，就縮小圖片
        if self.test > 20:
            self.change += 1
            self.test = 0
            ratio = random.uniform(0.6, 0.9)

            #設定圖片縮小不可縮至太小，預設最小像素值為40
            pixel_threshold = 40
            if self.img.shape[0] < self.img.shape[1] :
                value = pixel_threshold / self.img.shape[0]
                ratio = random.uniform(value, 1)

            if self.img.shape[0]  > self.img.shape[1] :
                value = pixel_threshold / self.img.shape[1]
                ratio = random.uniform(value, 1)

            seq = iaa.Sequential(
                [iaa.Resize({"height": int(self.img.shape[0] * ratio), "width": int(self.img.shape[1] * ratio)})])
            self.img = seq.augment_images([self.img])[0]
            self.b_bbs = seq.augment_bounding_boxes(
                [self.b_bbs])[0].remove_out_of_image().cut_out_of_image()
            #如果改變照片超過3次，就跳過這張圖片
            if self.change < 3 :
                self.coincide(self.bbs_aug_A.bounding_boxes, self.y, self.x, self.img, self.b_bbs)
            else :
                self.Error = 0

        #重疊大於閾值，就重新隨機座標
        if (self.Error == 1):
            self.again()

        return self.img, self.b_bbs ,self.y ,self.x

    def again(self):
        while (self.Error == 1):
            # 隨機取得影像疊加之座標
            self.y = random.randint(0, self.imageA_aug.shape[0] - self.img.shape[0])
            self.x = random.randint(0, self.imageA_aug.shape[1] - self.img.shape[1])
            self.test += 1
            self.coincide(self.bbs_aug_A.bounding_boxes, self.y, self.x, self.img, self.b_bbs)

    # 判斷影像重疊，回傳值為1表示有重疊，回傳值為0表示未重疊
    def bb_overlap(self,x1, y1, w1, h1, x2, y2, w2, h2):

        if (x1 > x2 + w2):
            return 0
        if (y1 > y2 + h2):
            return 0
        if (x1 + w1 < x2):
            return 0
        if (y1 + h1 < y2):
            return 0
        # 表示有重疊
        return 1


    def Augment(self):
        if self.imageDirA is None or self.imageDirB is None or self.savepath is None:
            QMessageBox.warning(self, '警告', '尚未完成設定資料夾路徑')
        else:
            for j in range(len(self.imageListB)):
                for i in range(int(self.times.text())):
                    try:
                        self.seq_arg = rota.get()

                        # 確認開啟的xml與影像為同一檔案
                        new_name = self.imageListB[j].split('.')[0] + '.xml'
                        if new_name in self.xmlListB:
                            annotation = an.parse_xml(new_name)
                        else:
                            continue

                        filename = annotation['filename']

                        sp = filename.split('.')
                        string = time.time()

                        outfile = '%s/%s-%02d-%s.%s' % (self.savepath, sp[0], i, str(string)[-6:-1], sp[-1])

                        #讀取影像B的label資訊
                        _bbs = []
                        for obj in annotation['objects']:
                            bb = ia.BoundingBox(x1=int(obj['xmin']),
                                                y1=int(obj['ymin']),
                                                x2=int(obj['xmax']),
                                                y2=int(obj['ymax']),
                                                label=obj['name'])
                            _bbs.append(bb)


                        self.image_B = cv2.imread(self.imageListB[j])
                        print(self.imageListB[j])

                        self.h_B = self.image_B.shape[0]
                        self.w_B = self.image_B.shape[1]

                        self.bbs = ia.BoundingBoxesOnImage(_bbs, shape=self.image_B.shape)

                        # 得到augment後的B影像
                        self.imageB_aug, self.bbs_aug = self.process_B()
                        self.height_B = self.imageB_aug.shape[0]
                        self.width_B = self.imageB_aug.shape[1]

                        #生成兩張影像被勾選
                        if self.checkBoxB_3.checkState() == 2:
                            double = True
                            # 得到第二張augment後的B影像
                            self.imageB_aug2, self.bbs_aug2 = self.process_B()
                            h = self.imageB_aug2.shape[0]
                            w = self.imageB_aug2.shape[1]
                            self.height_B = self.height_B + h
                            self.width_B = self.width_B + w

                        elif self.checkBoxB_3.checkState() == 0:
                            double = False

                        # 隨機在A資料夾選取一張圖片
                        ran = random.randint(0, len(self.imageListA) - 1)

                        image_A = cv2.imread(self.imageListA[ran])
                        height_A = image_A.shape[0]
                        width_A = image_A.shape[1]

                        # 確認開啟的xml與影像為同一檔案
                        new_name = self.imageListA[ran].split('.')[0] + '.xml'

                        if new_name in self.xmlListA:
                            annotation_A = an.parse_xml(new_name)
                        else:
                            continue

                        #讀取A的label資訊
                        _bbs_A = []
                        for obj in annotation_A['objects']:
                            bb_A = ia.BoundingBox(x1=int(obj['xmin']),
                                                  y1=int(obj['ymin']),
                                                  x2=int(obj['xmax']),
                                                  y2=int(obj['ymax']),
                                                  label=obj['name'])
                            _bbs_A.append(bb_A)

                        self.bbs_A = ia.BoundingBoxesOnImage(_bbs_A, shape=image_A.shape)

                        # 影像A改變大小，輸入B影像大小確保A影像大於B影像
                        seqA = resize.get(self.height_B, self.width_B, height_A, width_A, self.checkBoxA_1.checkState())
                        self.imageA_aug = seqA.augment_images([image_A])[0]
                        self.bbs_aug_A = seqA.augment_bounding_boxes(
                            [self.bbs_A])[0].remove_out_of_image().cut_out_of_image()

                        # A的augment被勾選
                        if self.checkBoxA_2.checkState() == 2:
                            seq_det = self.seq_arg.to_deterministic()
                            self.imageA_aug = seq_det.augment_images([self.imageA_aug])[0]

                        writer = Writer(outfile,
                                        annotation['size']['width'],
                                        annotation['size']['height'])

                        if double == False:
                            # 影像疊加
                            new_h_A = self.imageA_aug.shape[0]
                            new_w_A = self.imageA_aug.shape[1]
                            new_h_B = self.imageB_aug.shape[0]
                            new_w_B = self.imageB_aug.shape[1]

                            # 隨機取得影像疊加之座標
                            global_y0 = random.randint(0, new_h_A - new_h_B)
                            global_x0 = random.randint(0, new_w_A - new_w_B)

                            #確認B與A的Label重疊不低於70%
                            self.test = 0
                            self.change = 0
                            self.imageB_aug, self.bbs_aug , global_y0 ,global_x0 = self.coincide(self.bbs_aug_A.bounding_boxes,global_y0,global_x0,self.imageB_aug, self.bbs_aug)

                            #超過3次，代表縮小圖片3次也找不到適當的位置，就跳過本次augment
                            if self.change >= 3 :
                                continue

                            # 更改xml檔之座標
                            for bb in self.bbs_aug.bounding_boxes:
                                writer.addObject(bb.label,
                                                 int(bb.x1) + global_x0,
                                                 int(bb.y1) + global_y0,
                                                 int(bb.x2) + global_x0,
                                                 int(bb.y2) + global_y0)

                            #self.white = np.zeros_like(self.imageB_aug)
                            #將影像B放入影像A中
                            self.imageA_aug[global_y0:self.imageB_aug.shape[0] + global_y0, global_x0:self.imageB_aug.shape[1] + global_x0] = self.imageB_aug

                        elif double == True:

                            # 影像疊加
                            new_h_A = self.imageA_aug.shape[0]
                            new_w_A = self.imageA_aug.shape[1]
                            new_h_B = self.imageB_aug.shape[0]
                            new_w_B = self.imageB_aug.shape[1]
                            new_h_B2 = self.imageB_aug2.shape[0]
                            new_w_B2 = self.imageB_aug2.shape[1]

                            # 隨機取得兩張影像疊加之座標
                            global_y0 = random.randint(0, new_h_A - new_h_B)
                            global_x0 = random.randint(0, new_w_A - new_w_B)
                            global_y0_2 = random.randint(0, new_h_A - new_h_B2)
                            global_x0_2 = random.randint(0, new_w_A - new_w_B2)

                            # 確認B與A的Label重疊不低於70%
                            self.test = 0
                            self.change = 0
                            self.imageB_aug, self.bbs_aug, global_y0, global_x0 = self.coincide(
                                self.bbs_aug_A.bounding_boxes, global_y0, global_x0, self.imageB_aug, self.bbs_aug)
                            if self.change >= 3 :
                                continue

                            self.test = 0
                            self.change = 0
                            self.imageB_aug2, self.bbs_aug2, global_y0_2, global_x0_2 = self.coincide(
                                self.bbs_aug_A.bounding_boxes, global_y0_2, global_x0_2, self.imageB_aug2, self.bbs_aug2)
                            if self.change >= 3 :
                                continue

                            # 判斷兩張b是否有重疊(a=1 為重疊,a=0則為否)
                            a = self.bb_overlap(global_x0, global_y0, new_w_B, new_h_B,
                                              global_x0_2, global_y0_2, new_w_B2, new_h_B2)

                            times = 0
                            #設定重疊測試次數
                            overlap_threshold_times = 5

                            while (a == 1):
                                # 如果第一次測試結果為重疊，就重新隨機取值，再重新判斷，如果超過overlap_threshold_times，則跳出while迴圈，輸出為一張疊加照片
                                global_y0 = random.randint(0, new_h_A - new_h_B)
                                global_x0 = random.randint(0, new_w_A - new_w_B)
                                global_y0_2 = random.randint(0, new_h_A - new_h_B2)
                                global_x0_2 = random.randint(0, new_w_A - new_w_B2)
                                a = self.bb_overlap(global_x0, global_y0, new_w_B, new_h_B,
                                                  global_x0_2, global_y0_2, new_w_B2, new_h_B2)
                                times += 1
                                if times > overlap_threshold_times:
                                    break

                            #超過預設測試次數，就輸出一張照片即可
                            if times > overlap_threshold_times:
                                #self.white = np.zeros_like(self.imageB_aug)
                                # 將影像B放入影像A中
                                self.imageA_aug[global_y0:self.imageB_aug.shape[0] + global_y0,
                                                global_x0:self.imageB_aug.shape[1] + global_x0] = self.imageB_aug

                                #將影像B資訊寫入xml檔中
                                for bb in self.bbs_aug.bounding_boxes:
                                    writer.addObject(bb.label,
                                                     int(bb.x1) + global_x0,
                                                     int(bb.y1) + global_y0,
                                                     int(bb.x2) + global_x0,
                                                     int(bb.y2) + global_y0)

                            else:
                                # 將兩張影像B放入影像A中
                                #self.white = np.zeros_like(self.imageB_aug)
                                #self.white2 = np.zeros_like(self.imageB_aug2)
                                self.imageA_aug[global_y0:self.imageB_aug.shape[0] + global_y0,
                                                global_x0:self.imageB_aug.shape[1] + global_x0] = self.imageB_aug
                                self.imageA_aug[global_y0_2:self.imageB_aug2.shape[0] + global_y0_2,
                                                global_x0_2:self.imageB_aug2.shape[1] + global_x0_2] = self.imageB_aug2

                                # 將影像B資訊寫入xml檔中
                                for bb in self.bbs_aug.bounding_boxes:
                                    writer.addObject(bb.label,
                                                     int(bb.x1) + global_x0,
                                                     int(bb.y1) + global_y0,
                                                     int(bb.x2) + global_x0,
                                                     int(bb.y2) + global_y0)
                                for bb in self.bbs_aug2.bounding_boxes:
                                    writer.addObject(bb.label,
                                                     int(bb.x1) + global_x0_2,
                                                     int(bb.y1) + global_y0_2,
                                                     int(bb.x2) + global_x0_2,
                                                     int(bb.y2) + global_y0_2)

                        # 加入A的label進xml檔
                        for bb in self.bbs_aug_A.bounding_boxes:
                            writer.addObject(bb.label,
                                             int(bb.x1),
                                             int(bb.y1),
                                             int(bb.x2),
                                             int(bb.y2))

                        cv2.imwrite(outfile, self.imageA_aug)
                        writer.save('%s.xml' % outfile.split('.')[0])

                    # 避免xml為空檔案的報錯
                    except ParseError:
                        continue
                    self.xml = '%s.xml' % outfile.split('.')[0]

                    xml = ET.parse(self.xml)
                    root = xml.getroot()

                    node_size = root[4]
                    h = self.imageA_aug.shape[0]
                    w = self.imageA_aug.shape[1]
                    node_size[0].text = str(w)
                    node_size[1].text = str(h)

                    xml = ET.ElementTree(root)
                    xml.write(self.xml)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.setWindowTitle('Augment Window')
    window.show()
    sys.exit(app.exec_())