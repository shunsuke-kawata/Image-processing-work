import cv2
import dlib
import numpy as np
from imutils import face_utils

#顔を検出するカスケード分類器ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)

#取得したlandmarkを表示する関数(デバック用関数)
def display_landmarks(landmarks,frame):
    index = 0
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 2, (0, 0, 255), thickness=-1)
        cv2.putText(frame,str(index), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
        index+=1

#取得した画像フレームの処理を行うメインの関数
def operate_frame(frame,color):
    #処理を高速にするためにグレースケール化
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(frame_gray, 1)

    #検出した顔の数だけ処理を行う
    for face in faces:
        #landmarkの検出
        landmark = face_predictor(frame_gray, face)
        landmark_np = face_utils.shape_to_np(landmark)

        #マスク画像
        mask = np.zeros(frame.shape, dtype=np.uint8)

        """
        center_x:目の中心のx座標
        center_y:目の中心のy座標
        eye_width:目の横幅
        eye_height:目の縦幅
        """
        center_x =int((landmark_np[36][0]+landmark_np[39][0])//2-1)
        center_y = int(landmark_np[27][1]-2)
        eye_width = int(landmark_np[39][0]-landmark_np[36][0]+4)
        eye_height = int(((landmark_np[41][1]-landmark_np[37][1])+(landmark_np[40][1]-landmark_np[38][1]))//2)+4

        #目の位置に同じ大きさの楕円を描画する
        cv2.ellipse(mask, ((center_x, center_y), (eye_width, eye_height), 0), color=color, thickness=-1, lineType=cv2.LINE_8)
        #AND演算によって画像を切り取る
        img = mask & frame
    
        #目の位置から画像を取得
        resized_img = img[int(landmark_np[37][1])-7:int(landmark_np[41][1])+3, int(landmark_np[36][0])-5:int(landmark_np[39][0])+5]
        resized_height, resized_width = resized_img.shape[:2]

        #位置を調整して画像を置き換える
        forehead_x,forehead_y = int(landmark_np[27][0]-resized_height/2-4),int(2*landmark_np[21][1]-landmark_np[29][1])
        #マスク画像におけるおおよそ額の位置に目の画像を貼り付ける
        img[forehead_y:forehead_y+resized_height,forehead_x:forehead_x+resized_width]=resized_img
        #マスク画像における元々の目の位置を黒に変更
        img[int(landmark_np[37][1])-20:int(landmark_np[41][1])+20, int(landmark_np[36][0])-20:int(landmark_np[39][0])+20]=(0,0,0)

        #右目の画像と元々のフレームを合成
        frame = cv2.addWeighted(src1=frame,alpha=0.7,src2=img,beta=1.0,gamma=0)

        # #マスク画像と合成画像を出力
        # cv2.imwrite("mask.jpg",img)
        # cv2.imwrite("frame.jpg",frame)
        
        #landmarkの表示
        display_landmarks(landmark_np,frame)
        
    return frame

if __name__=='__main__':
    #カメラの映像をキャプチャする
    cap = cv2.VideoCapture(0)
    #表示する目のRGBをtupleで指定する
    color = tuple([255,255,255])
    while(True):
        #画像のキャプチャした映像からのフレームの読み込み
        ret, frame = cap.read()

        #目の色を変更する
        if(cv2.waitKey(1) & 0xFF == ord('r')):
            color=tuple([0,0,255])
            print('r was clicked')
        elif(cv2.waitKey(1) & 0xFF == ord('g')):
            color = tuple([0,255,0])
            print('g was clicked')
        elif(cv2.waitKey(1) & 0xFF == ord('b')):
            color=tuple([255,0,0])
            print('b was clicked')
        elif(cv2.waitKey(1) & 0xFF == ord('w')):
            color=tuple([255,255,255])
            print('w was clicked')
        else:
            pass

        operated_frame = operate_frame(frame,color)
        cv2.imshow('合成画像',operated_frame)
        #qのASCII表現によって終了判定
        
        #qを押すと終了する
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('q was clicked')
            break
    #カメラの解放とウィンドウの削除
    cap.release()
    cv2.destroyAllWindows()
