#OpenCV - 명함인식 구현하기
import numpy as np
import cv2




#Apply Perspectine Transform
def order_points(pts):
    rect=np.zeros((4,2), dtype="float32") #일단 최종적으로 넘길 array도 x,y 쌍이 4개인 array이기 때문에

    s=pts.sum(axis=1) #sum(): 넘겨받은 배열의 각 행이나 열의 합을 도출하는 함수
    #axis=1: 각 행에 대한 합을 계산하는 것

    rect[0]=pts[np.argmin(s)] #x+y의 최대값
    rect[2]=pts[np.argmax(s)] #x+y의 최소값
    #을 이용해서 각 꼭지점의 위치를 파악한 후 반환한다.

    diff=np.diff(pts, axis=1)
    rect[1]=pts[np.argmin(diff)] #y-x의 최소값
    rect[3]=pts[np.argmax(diff)] #y-x의 최대값
    #을 이용해서 각 꼭지점의 위치를 파악한 후 반환한다.

    return rect




#Edge Detection
def auto_scan_image():
    image=cv2.imread('C:\\Users\\inaee\\Desktop\\2020-1MAKERS\\OCR\\square_real\\real_55.jpg') #이미지를 읽어옴
    orig=image.copy() #원본 이미지는 따로 복사해놓음

    r=800.0 / image.shape[0]
    dim=(int(image.shape[1]*r), 800)
    image=cv2.resize(image, dim, interpolation=cv2.INTER_AREA) #이미지를 resize

    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #이미지의 색공간을 BGR에서 GRAY로 바꿈
    gray=cv2.GaussianBlur(gray, (3,3), 0) #GaussianBlur를 통해 blur효과를 줌. 외곽 검출을 더 쉽게 함.
    edged=cv2.Canny(gray, 75, 200) #Canny Edge Detection을 통해서 edge를 검출하게 함

    print("STEP 1: Edge Detection") #엣지가 검출된 이미지를 출력

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL) #윈도우 사이즈 지정
    cv2.namedWindow('Edged', cv2.WINDOW_NORMAL) #윈도우 사이즈 지정
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.imwrite("C:\\Users\\inaee\\Desktop\\2020-1MAKERS\\OCR\\square_real\\real_edged\\real_edged_55.jpg",edged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




    #Find Contours of Paper
    (_, cnts, _)=cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #findContours를 통해 contours들을 반환받음
    # _: 계층관계는 필요가 없기 때문에 contour만 명시적으로 반환을 받겠다

    cnts=sorted(cnts, key=cv2.contourArea, reverse=True)[:5] #반환 받은 contour를 외곽이 그린 면적이 큰 순서대로 정렬해서 5개만 받아온다
    #cv2.contourArea는 contour가 그린 면적을 의미한다

    #그렇게 받아온 contour를 순차적으로 탐색하면서
    for c in cnts:
        peri=cv2.arcLength(c, True) #contour가 그리는 길이를 반환
        approx=cv2.approxPolyDP(c, 0.02 * peri, True) #그 길이에 2%정도 오차를 해서 approxPolyDP를 통해 도형을 조금 근사해서 구함

        #도형을 근사해서 외곽을 추출한 꼭지점이 4개라면 그것이 명암의 외곽으로 본다
        #정리하자면, 찾아낸 외곽을 통해서 가장 큰 것 순서대로 꼭지점이 4개인 것을 찾아냈을때 바로 그것이 명함의 외곽이다라고 보는 것이다

        if len(approx)==4:
            screenCnt = approx
            break

    print("STEP 2: Find Contours of Paper")

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2) #drawContours를 통해 contours를 그림
    cv2.imshow("Outline", image)
    cv2.imwrite("C:\\Users\\inaee\\Desktop\\2020-1MAKERS\\OCR\\square_real\\real_outline\\real_outline_55.jpg", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)




    rect=order_points(screenCnt.reshape(4,2)/r) #contours에서 4개의 꼭지점을 topLeft, topRight, bottomRight, bottomLeft
    #parameter: 구했던 contour배열을 4행2열의 배열로 재정렬해서 (x,y)쌍이 4개로 묶인 것을 넘김
    (topLeft,topRight,bottomRight,bottomLeft)=rect

    #두 개의 너비와 높이를 계산해서
    w1=abs(bottomRight[0]-bottomLeft[0])
    w2=abs(topRight[0]-topLeft[0])
    h1=abs(topRight[1]-bottomRight[1])
    h2=abs(topLeft[1]-bottomLeft[1])

    #최대 너비와 최대 높이를 계산한다
    maxWidth=max([w1,w2])
    maxHeight=max([h1,h2])

    #그리고 반환될 좌표의 위치를 초기화해서
    dst=np.float32([[0,0], [maxWidth-1,0], [maxWidth-1,maxHeight-1], [0,maxHeight-1]])

    #getPerspectiveTransform()함수를 통해서 나머지 픽셀을 옮기는 매트릭스 M에 반환한다.
    M=cv2.getPerspectiveTransform(rect, dst)

    #그 매트릭스 M를 warpPerspective()에 넣음으로써 최종적으로 반듯한 사각형으로 반환된 이미지를 받게 된다
    warped=cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    #show the original and scanned images
    print("STEP 3: Apply perspective transform")
    cv2.imshow("Warped", warped)
    cv2.imwrite("C:\\Users\\inaee\\Desktop\\2020-1MAKERS\\OCR\\square_real\\real_warped\\real_warped_55.jpg", warped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)





    #STEP 4: Apply Adaptive Threshold
    warped=cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) #cvtColor()를 통해서 BGR을 GRAY로 바꾸고

    #adaptiveThreshold()를 통해서 모든 흑백 이미지를 모두 0과 1, 즉 흰색과 검은색으로 바꿈
    warped=cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    print("STEP 4: Apply Adaptive Threshold")
    cv2.imshow("Original", orig)
    cv2.imshow("Scanned", warped)
    cv2.imwrite("C:\\Users\\inaee\\Desktop\\2020-1MAKERS\\OCR\\square_real\\real_scanned\\real_scanned_55.jpg", warped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__=='__main__':
    auto_scan_image()
