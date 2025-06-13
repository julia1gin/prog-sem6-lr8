# подключаем библиотеку компьютерного зрения
import cv2
# подключаем модуль для работы с аргументами при вызове
import argparse

# подключаем парсер аргументов командной строки
parser=argparse.ArgumentParser()
# добавляем аргумент для работы с изображениями
parser.add_argument('--image')
# сохраняем аргумент в отдельную переменную
args=parser.parse_args()
# прописываем цвет по умолчанию
color = (0,255,0)
# функция определения лиц
def highlightFace(net, frame, conf_threshold=0.7):
    # делаем копию текущего кадра
    frameOpencvDnn=frame.copy()
    # высота и ширина кадра
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # преобразуем картинку в двоичный пиксельный объект
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    # устанавливаем этот объект как входной параметр для нейросети
    net.setInput(blob)
    # выполняем прямой проход для распознавания лиц
    detections=net.forward()
    # переменная для рамок вокруг лица
    faceBoxes=[]
    # перебираем все блоки после распознавания
    for i in range(detections.shape[2]):
        # получаем результат вычислений для очередного элемента
        confidence=detections[0,0,i,2]
        # если результат превышает порог срабатывания — это лицо
        if confidence>conf_threshold:
            # формируем координаты рамки
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            # добавляем их в общую переменную
            faceBoxes.append([x1,y1,x2,y2])
            # рисуем рамку на кадре
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), color, int(round(frameHeight/150)), 8)
    # возвращаем кадр с рамками
    return frameOpencvDnn,faceBoxes

# загружаем веса для распознавания лиц
faceProto="opencv_face_detector.pbtxt"
# и конфигурацию самой нейросети — слои и связи нейронов
faceModel="opencv_face_detector_uint8.pb"
# точно так же загружаем модели для определения пола и возраста
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"

# настраиваем свет
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# итоговые результаты работы нейросетей для пола и возраста
genderList=['Male ','Female']
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# запускаем нейросеть по распознаванию лиц
faceNet=cv2.dnn.readNet(faceModel,faceProto)
# и запускаем нейросети по определению пола и возраста
genderNet=cv2.dnn.readNet(genderModel,genderProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)

# если был указан аргумент с картинкой — берём картинку как источник
video=cv2.VideoCapture(args.image if args.image else 0)
# пока не нажата любая клавиша — выполняем цикл

while cv2.waitKey(1)<0:
    frame = cv2.imread("example4.jpg")
    # определяем лица в кадре
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    # перебираем все найденные лица в кадре
    for faceBox in faceBoxes:
        # получаем изображение лица на основе рамки
        face=frame[max(0,faceBox[1]):
                   min(faceBox[3],frame.shape[0]-1),max(0,faceBox[0])
                   :min(faceBox[2], frame.shape[1]-1)]
        # получаем на этой основе новый бинарный пиксельный объект
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        # отправляем его в нейросеть для определения пола
        genderNet.setInput(blob)
        # получаем результат работы нейросети
        genderPreds=genderNet.forward()
        # выбираем пол на основе этого результата
        gender=genderList[genderPreds[0].argmax()]
        # отправляем результат в переменную с полом
        print(f'Gender: {gender}')

        # делаем то же самое для возраста
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        # добавляем текст возле каждой рамки в кадре
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        # выводим итоговую картинку
        cv2.imshow("Detecting age and gender", resultImg)