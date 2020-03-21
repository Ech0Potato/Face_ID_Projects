import cv2
import numpy as np
import math
import dlib
import time

class face_cam_api():

    # 类封装的常量 ： 人脸追踪返回常量
    TRACK_FLAG_IMG = 0
    TRACK_FLAG_IMG_TIME = 1

    def __init__(self,file,width,height,face_detect_mode ):
        # 文件名称:
        self.CNN_DATA_FILENAME = "mmod_human_face_detector.dat"
        self.FACE_PREDICTOR_FILENAME = "68Marks.dat"

        # 基础信息
        self.file = file
        self.width = width
        self.height = height

        # 初始化Video捕获句柄
        self.cap = cv2.VideoCapture(self.file)
        self.cap.set(3,self.width)
        self.cap.set(4,self.height)

        # 判断输入流种类
        self.STREAM_VIDEO = 0
        self.STREAM_WEBCAM = 1
        self.stream_type = None
        if type(file) == str :
            self.stream_type = self.STREAM_VIDEO
        elif type(file) == int :
            self.stream_type = self.STREAM_WEBCAM
        else :
            raise TypeError("视频流格式输入错误，请检查声明视频采集实例时第一个参数")

        # 根据输入视频流种类分析信息
        self.video_fps = None
        if self.stream_type == self.STREAM_VIDEO :
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 确认人脸检测方式: 使用CNN网络 : 0  , 使用梯度检测算法 : 1
        self.face_detector = None
        self.face_detect_mode = face_detect_mode
        if self.face_detect_mode == 0:
            self.face_detector = dlib.cnn_face_detection_model_v1(
                                 self.CNN_DATA_FILENAME)
        elif self.face_detect_mode == 1 :
            self.face_detector = dlib.get_frontal_face_detector()
        else :
            raise TypeError("人脸检测算法不明确")

        # 当前人脸检测检测到的识别框信息: 数据格式 :  dlib.rectangle
        self.face_rectangle_now = None
        # 上一帧人脸检测检测到的识别框信息 : 数据格式 : dlib.rectangle
        self.face_rectangle_previous = None

        # CNN人脸检测方法置信度
        self.face_cnn_detection_confidence = None


        # 人脸68数据点预测器初始化
        self.face_predictor = dlib.shape_predictor(
                        self.FACE_PREDICTOR_FILENAME)

        # 人脸检测
        # 当前人脸检测特征点
        self.face_landmarks_now = None
        # 上一次人脸检测特征点
        self.face_landmarks_previous = None


        # 其他常量
        # 灰度图SHAPE
        self.FACE_DETECT_IMG_GARY_SHAPE = (self.height,self.width)
        # 彩色图SHAPE
        self.FACE_DETECT_IMG_COLOR_SHAPE = (self.height,self.width,3)

        # get_new_img 函数选择返回值标识
        # 只返回彩色
        self.RETURN_COLOR = 0
        # 只返回灰度
        self.RETURN_GRAY = 1
        # 返回彩色和灰度 按照:彩色,灰度的顺序
        self.RETURN_COLOR_GRAY = 2
        # 返回灰度和彩色 按照:灰度,彩色的顺序
        self.RETURN_GRAY_COLOR = 3

        # 当前摄像头采集到的彩色帧
        self.frame_color_now = None
        # 当前摄像头采集到的灰度帧
        self.frame_gray_now = None
        # 摄像头上一次采集到的彩色帧
        self.frame_color_previous = None
        # 摄像头上一次采集到的灰度帧
        self.frame_gray_previous = None


        # 光流参数
        self.KLT_PARAMS = dict(winSize=(15, 15), maxLevel=3,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               30, 0.01))

        # 一些Flag标识

        # 当前的face_landmarks_now 是由特征点计算而来 还是由KLT计算而来
        self.flag_face_landmarks_now = None
        # 特征常量
        self.FLAG_FACE_LANDMARKS_NOW_MADE_BY_PREDICTION = 0
        self.FLAG_FACE_LANDMARKS_NOW_MADE_BY_KLT = 1

        # 当前的face_landmarks_previous 是由特征点计算而来还有由KLT计算而来
        self.flag_face_landmarks_previous = None
        # 特征常量
        self.FLAG_FACE_LANDMARKS_PREVIOUS_MADE_BY_PREDICTION = 0
        self.FLAG_FACE_LANDMARKS_PREVIOUS_MADE_BY_KLT = 1

    def get_new_img(self,return_flag = 0):

        # 上一帧数据保存
        self.frame_color_previous = self.frame_color_now
        self.frame_gray_previous = self.frame_gray_now

        # 读取新的帧
        ret_bool,self.frame_color_now = self.cap.read()
        self.frame_gray_now = cv2.cvtColor(self.frame_color_now,cv2.COLOR_BGR2GRAY)

        # 根据返回值的flag确定返回的值
        if return_flag == self.RETURN_COLOR :
            return self.frame_color_now
        elif return_flag == self.RETURN_GRAY :
            return self.frame_gray_now
        elif return_flag == self.RETURN_COLOR_GRAY :
            return self.frame_color_now,self.frame_gray_now
        elif return_flag == self.RETURN_GRAY_COLOR :
            return self.frame_gray_now,self.frame_color_now
        else:
            raise TypeError("get_new_img : face_detect_mode 不对")

    def face_detect(self,img_gray,resampling_times):
        if img_gray.shape != self.FACE_DETECT_IMG_GARY_SHAPE :
            raise TypeError("输入图像与设定图像分辨率不相等,采集到的shape:{0}".format(img_gray.shape))

        # 上下帧数据交换:
        self.face_rectangle_previous = self.face_rectangle_now

        # 人脸识别 读取图像灰度 和重采样次数
        detections = self.face_detector(img_gray,resampling_times)

        # 检测是否育人脸,detections的长度
        length_of_detections = len(detections)
        # 进行循环判断,如果没有,则采样判断会一直进行下去.
        # 对于摄像头,这种方法是适用的
        # 对于视频文件,则是放弃本帧采集下一帧,因为采集的关键在于人脸,而不在于某一帧
        while length_of_detections == 0 :
            print("没有发现人脸,重新检测")
            img_color,img_gray = self.get_new_img(return_flag=self.RETURN_COLOR_GRAY)
            detections = self.face_detector(img_gray,resampling_times)
            length_of_detections = len(detections)
            # 当重新检测 检测到人脸后 更新数据
            if length_of_detections != 0 :
                self.frame_color_previous = self.frame_color_now
                self.frame_gray_previous = self.frame_gray_now
                self.frame_color_now = img_color
                self.frame_gray_now = img_gray

        # 根据人脸识别器处理识别器返回的数据
        if self.face_detect_mode == 0 :
            self.face_cnn_detection_confidence = detections[0].confidence
            self.face_rectangle_now = detections[0].rect

        elif self.face_detect_mode == 1 :
            self.face_rectangle_now = detections[0]

        return self.face_rectangle_now

    def face_landmarks_predict(self,img_gray,face_rectangle):
        if img_gray.shape != self.FACE_DETECT_IMG_GARY_SHAPE :
            raise TypeError("输入图像与设定图像分辨率不相等")

        # 特征点预测
        detection = self.face_predictor(img_gray,face_rectangle)

        # 特征点数据类型转换. 转换为 shape = (68,2) 的 Ndarray
        detection_list = list()

        for each in detection.parts():
            detection_list.append([each.x,each.y])

        # 本次和上次数据交换
        self.face_landmarks_previous = self.face_landmarks_now
        self.face_landmarks_now = np.array(detection_list)

        return self.face_landmarks_now

    def face_detect_show(self,waitting_time_in_ms,picname) :
        img,img_gray = self.get_new_img(return_flag=self.RETURN_COLOR_GRAY)
        face_rects = self.face_detect(img_gray,0)
        left = face_rects.left()
        top = face_rects.top()
        right = face_rects.right()
        bottom = face_rects.bottom()
        cv2.rectangle(img,(left,top),(right,bottom),[0,255,0],2)
        cv2.imshow("{}".format(picname),img)
        return_key = cv2.waitKey(waitting_time_in_ms)
        if return_key == ord('a') :
            pass

    def face_KeyArea_rotate_fit(self,img,landmarks):
        '''
        输入待处理的人脸整体图像 输入68个坐标点
        返回旋转切割好的待检测区域图像
        '''
        # 变量重命名,因为懒得改代码
        # 提取左右眉毛中心点 -------------

        # 左眉毛索引起点与终点(图片的左右,不是人的左右眼)
        LEFT_EYEBROW_START = 17
        LEFT_EYEBROW_END = 22

        # 右眉毛索引起点与终点(图片的左右,不是人的左右眼)
        RIGHT_EYEBROW_START = 22
        RIGHT_EYEBROW_END = 27

        # 取出左右眼眉毛点集合
        left_arrow_points = landmarks[LEFT_EYEBROW_START:LEFT_EYEBROW_END]
        right_arrow_points = landmarks[RIGHT_EYEBROW_START:RIGHT_EYEBROW_END]

        # 取集合平均点
        left_arrow_center = np.mean(left_arrow_points,axis=0).astype("int")
        right_arrow_center = np.mean(right_arrow_points,axis=0).astype("int")

        # 旋转参数准备 -------------------

        # 设置旋转中心坐标
        rotate_center = np.mean([left_arrow_center,right_arrow_center],axis=0).astype("int")

        # 计算旋转角度
        delta_x = right_arrow_center[0] - left_arrow_center[0]
        delta_y = right_arrow_center[1] - left_arrow_center[1]
        rotate_angle_rad = math.atan2(delta_y,delta_x)
        rotate_angle_deg = rotate_angle_rad / math.pi * 180.0

        # 确定仿射变换矩阵
        rotate_matrix = cv2.getRotationMatrix2D(tuple(rotate_center),rotate_angle_deg,scale=1)

        # 仿射变换
        rotate_img = cv2.warpAffine(img,rotate_matrix,(img.shape[1],img.shape[0]))

        # 关键点坐标变换函数
        def rotate_single_point(center,point,angle_rad,row_pixels):
            x1, y1 = point[0],point[1]
            x2, y2 = center[0],center[1]
            y1 = row_pixels - y1
            y2 = row_pixels - y2
            angle = angle_rad
            x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
            y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
            y = row_pixels - y
            return int(x),int(y)

        # 脸部切割
        landmarks_rotated = list()
        for each_landmark in landmarks :
            each_landmark_rotated = rotate_single_point(rotate_center,each_landmark,rotate_angle_rad,img.shape[0])
            landmarks_rotated.append(each_landmark_rotated)
        landmarks_rotated = np.array(landmarks_rotated)

        face_shaped = rotate_img[landmarks_rotated[19][1]:landmarks_rotated[33][1],
                                 landmarks_rotated[17][0]:landmarks_rotated[26][0]]

        return face_shaped

    def face_KLT_track(self,landmarks_previous,frame_gray_previous,frame_gray_now):
        # 数据结构整理 由原来的 shape : (68,2) 变为 (68,1,2)
        landmarks_previous_68_1_2 = landmarks_previous.reshape(-1,1,2).astype("float32")

        # 光流跟踪
        landmarks_now_shape68_1_2 , status , error_array =  cv2.calcOpticalFlowPyrLK(frame_gray_previous,frame_gray_now,
                                                                                     landmarks_previous_68_1_2,None,
                                                                                     **self.KLT_PARAMS)
        # 新旧数据更新
        self.face_landmarks_previous = self.face_landmarks_now
        self.face_landmarks_now = landmarks_now_shape68_1_2.reshape(68,2).astype("int") # 数据结构适配
        return self.face_landmarks_now

    def face_track_once(self,face_resampling_times,KLT_track_times):
        # 创建返回图像列表
        img_list = list()

        # 收集人脸照片
        frame_color,frame_gray = self.get_new_img(self.RETURN_COLOR_GRAY)

        # 开始人脸检测
        face_rectangle = self.face_detect(frame_gray,face_resampling_times)

        # 人脸特征点识别
        face_landmarks = self.face_landmarks_predict(self.frame_gray_now,face_rectangle)

        # 添加修正图片
        img_list.append(self.face_KeyArea_rotate_fit(self.frame_color_now,face_landmarks))

        for i in range(KLT_track_times):
            # 获得新图片
            self.get_new_img()
            # 获得光流追踪的特征点
            face_landmarks_in_circle = self.face_KLT_track(self.face_landmarks_now,self.frame_gray_previous,self.frame_gray_now)
            # 添加修正图片
            img_list.append(self.face_KeyArea_rotate_fit(self.frame_color_now,face_landmarks_in_circle))

        return img_list

    # 下面暂存了要修改的返回时间的部分，等待与数字信号处理算法的对接
    '''
    def face_track_once(self,face_resampling_times,KLT_track_times,track_return_flag = face_cam_api.TRACK_FLAG_IMG):
        # 帧时刻列表：单位位秒
        time_list = list()

        # 创建返回图像列表
        img_list = list()
        
        # 

        # 先采集，再计算

        frame_color, frame_gray = self.get_new_img(self.RETURN_COLOR_GRAY)

        # 开始人脸检测
        face_rectangle = self.face_detect(frame_gray, face_resampling_times)

        # 人脸特征点识别
        face_landmarks = self.face_landmarks_predict(self.frame_gray_now, face_rectangle)

        # 添加修正图片
        img_list.append(self.face_KeyArea_rotate_fit(self.frame_color_now, face_landmarks))

        # 为时间列表首项添加时间
        # 创建帧列表
        frame_gray_list = list()
        frame_color_list = list()

        # 开始采集
        if self.stream_type == self.STREAM_VIDEO :
            time_list.append(1.0 / self.video_fps)
            for i in range(KLT_track_times):
                frame_color_temp,frame_gray_temp = self.get_new_img(return_flag=self.RETURN_COLOR_GRAY)
                frame_color_list.append(frame_color_temp)
                frame_gray_list.append(frame_gray_temp)
                time_list.append(1.0/self.video_fps)
        elif self.stream_type == self.STREAM_WEBCAM :
            

        if self.stream_type == self.STREAM_VIDEO:
            # 收集人脸照片
            frame_color,frame_gray = self.get_new_img(self.RETURN_COLOR_GRAY)

            # 开始人脸检测
            face_rectangle = self.face_detect(frame_gray,face_resampling_times)

            # 人脸特征点识别
            face_landmarks = self.face_landmarks_predict(self.frame_gray_now,face_rectangle)

            # 添加修正图片
            img_list.append(self.face_KeyArea_rotate_fit(self.frame_color_now,face_landmarks))

            for i in range(KLT_track_times):
                # 获得新图片
                self.get_new_img()
                # 获得光流追踪的特征点
                face_landmarks_in_circle = self.face_KLT_track(self.face_landmarks_now,self.frame_gray_previous,self.frame_gray_now)
                # 添加修正图片
                img_list.append(self.face_KeyArea_rotate_fit(self.frame_color_now,face_landmarks_in_circle))

            for i in range(KLT_track_times+1):
                time_list.append(1.0/self.video_fps)

        elif self.stream_type == self.STREAM_WEBCAM :
            # 收集人脸照片
            frame_color,frame_gray = self.get_new_img(self.RETURN_COLOR_GRAY)

            # 开始人脸检测
            face_rectangle = self.face_detect(frame_gray,face_resampling_times)

            # 人脸特征点识别
            face_landmarks = self.face_landmarks_predict(self.frame_gray_now,face_rectangle)

            # 添加修正图片
            img_list.append(self.face_KeyArea_rotate_fit(self.frame_color_now,face_landmarks))

            for i in range(KLT_track_times):
                # 获得新图片
                self.get_new_img()
                # 获得光流追踪的特征点
                face_landmarks_in_circle = self.face_KLT_track(self.face_landmarks_now,self.frame_gray_previous,self.frame_gray_now)
                # 添加修正图片
                img_list.append(self.face_KeyArea_rotate_fit(self.frame_color_now,face_landmarks_in_circle))


        if track_return_flag == face_cam_api.TRACK_FLAG_IMG :
            return img_list
        elif track_return_flag == face_cam_api.TRACK_FLAG_IMG_TIME :
            return img_list,time_list
        '''


'''
def rotate_fit(img,landmarks):
    # 变量重命名,因为懒得改代码


    # 提取左右眉毛中心点 -------------

    # 左眉毛索引起点与终点(图片的左右,不是人的左右眼)
    LEFT_EYEBROW_START = 17
    LEFT_EYEBROW_END = 22

    # 右眉毛索引起点与终点(图片的左右,不是人的左右眼)
    RIGHT_EYEBROW_START = 22
    RIGHT_EYEBROW_END = 27

    # 取出左右眼眉毛点集合
    left_arrow_points = landmarks[LEFT_EYEBROW_START:LEFT_EYEBROW_END]
    right_arrow_points = landmarks[RIGHT_EYEBROW_START:RIGHT_EYEBROW_END]

    # 取集合平均点
    left_arrow_center = np.mean(left_arrow_points,axis=0).astype("int")
    right_arrow_center = np.mean(right_arrow_points,axis=0).astype("int")


    # 旋转参数准备 -------------------

    # 设置旋转中心坐标
    rotate_center = np.mean([left_arrow_center,right_arrow_center],axis=0).astype("int")

    # 计算旋转角度
    delta_x = right_arrow_center[0] - left_arrow_center[0]
    delta_y = right_arrow_center[1] - left_arrow_center[1]
    rotate_angle_rad = math.atan2(delta_y,delta_x)
    rotate_angle_deg = rotate_angle_rad / math.pi * 180.0

    # 确定仿射变换矩阵
    rotate_matrix = cv2.getRotationMatrix2D(tuple(rotate_center),rotate_angle_deg,scale=1)

    # 仿射变换
    rotate_img = cv2.warpAffine(img,rotate_matrix,(img.shape[1],img.shape[0]))

    # 关键点坐标变换函数
    def rotate_single_point(center,point,angle_rad,row_pixels):
        x1, y1 = point[0],point[1]
        x2, y2 = center[0],center[1]
        y1 = row_pixels - y1
        y2 = row_pixels - y2
        angle = angle_rad
        x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
        y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
        y = row_pixels - y
        return int(x),int(y)

    landmarks_rotated = list()
    for each_landmark in landmarks :
        each_landmark_rotated = rotate_single_point(rotate_center,each_landmark,rotate_angle_rad,img.shape[0])
        landmarks_rotated.append(each_landmark_rotated)
    landmarks_rotated = np.array(landmarks_rotated)



    face_shaped = rotate_img[landmarks_rotated[19][1]:landmarks_rotated[33][1],
                             landmarks_rotated[17][0]:landmarks_rotated[26][0]]

    return face_shaped
'''

if __name__ == "__main__" :
    Cap = face_cam_api(0,1920,1080,0)

    imgs = Cap.face_track_once(0,20)

    count = 0
    for each in imgs:
        count = count + 1
        cv2.imshow("{0},{1}:{2}".format(each.shape[0],each.shape[1],count),each)

    cv2.waitKey(0)