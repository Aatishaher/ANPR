import cv2
import easyocr
import pandas as pd

harcascade = "model/haarcascade_russian_plate_number.xml"
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

min_area = 500
count = 0

# Create an empty list to store plate data
plate_data_list = []

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    detected_plate_text = ""

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x:x + w]
            cv2.imshow("ROI", img_roi)

            # Perform OCR on the region of interest
            results = reader.readtext(img_roi)

            for (bbox, text, prob) in results:
                detected_plate_text += text + " "

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Append the detected plate text to the list
        plate_data_list.append(detected_plate_text.strip())

        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plates Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)

        # Save the DataFrame to a single Excel file
        columns = ['Plate Number']
        plate_data_df = pd.DataFrame(plate_data_list, columns=columns)
        plate_data_df.to_excel(f'plates/plate_data_all.xlsx', index=False)
        break  # Break the loop after saving the file

# Release the video capture when the loop is exited
cap.release()
cv2.destroyAllWindows()
