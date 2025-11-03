import os
import cv2


cap = cv2.VideoCapture(0)

directory = 'Image/'


while True:
    _, frame = cap.read()  
      
    
    count = {
        '0': len(os.listdir(directory + "/Hello")),
        '1': len(os.listdir(directory + "/Left")),
        '2': len(os.listdir(directory + "/Love")),
        '3': len(os.listdir(directory + "/Ok")),
        '4': len(os.listdir(directory + "/Right")),
        '5': len(os.listdir(directory + "/Thankyou")),
        '6': len(os.listdir(directory + "/Thumbdown")),
        '7': len(os.listdir(directory + "/Thumbup")),
        '8': len(os.listdir(directory + "/Peace")),
        '9': len(os.listdir(directory + "/Fist")),
        'a': len(os.listdir(directory + "/Loser")),
        'b': len(os.listdir(directory + "/Up")),
        'c': len(os.listdir(directory + "/Please")),
    }

    row = frame.shape[1]
    col = frame.shape[0]

    
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

    
    cv2.imshow("data", frame)
    cv2.imshow("ROI", frame[40:400, 0:300])

    
    frame = frame[40:400, 0:300]

    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory + 'Hello/' + str(count['0']) + '.png', frame)
    elif interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory + 'Left/' + str(count['1']) + '.png', frame)
    elif interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory + 'Love/' + str(count['2']) + '.png', frame)
    elif interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory + 'Ok/' + str(count['3']) + '.png', frame)
    elif interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory + 'Right/' + str(count['4']) + '.png', frame)
    elif interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory + 'Thankyou/' + str(count['5']) + '.png', frame)
    elif interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory + 'Thumbdown/' + str(count['6']) + '.png', frame)
    elif interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory + 'Thumbup/' + str(count['7']) + '.png', frame)
    elif interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory + 'Peace/' + str(count['8']) + '.png', frame)
    elif interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory + 'Fist/' + str(count['9']) + '.png', frame)
    elif interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory + 'Loser/' + str(count['a']) + '.png', frame)
    elif interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory + 'Up/' + str(count['b']) + '.png', frame)
    elif interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory + 'Please/' + str(count['c']) + '.png', frame)
    


cap.release()
cv2.destroyAllWindows()
