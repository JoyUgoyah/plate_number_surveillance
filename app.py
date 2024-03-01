import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from ultralytics import YOLO
import easyocr

    # Load YOLO model
model = YOLO('yolov8s.pt')  # ensure the model file is in the correct path

    # Initialize OCR reader
reader = easyocr.Reader(['en'])

def main():

    # Title
    st.title("License Plate Surveillance Prototype")


    # Initialize DataFrame
    df = pd.DataFrame(columns=['Plate_number'])

    

    
# Start webcam capture
    cap = cv2.VideoCapture(0)

    #if 'cap' not in st.session_state:
        #st.session_state['cap'] = cv2.VideoCapture(0)

    #cap = st.session_state['cap']
    

    # Placeholder for the video frame
    frame_placeholder = st.empty()

    # Placeholder to display DataFrame
    df_placeholder = st.empty()


        # Process video
    while cap.isOpened():
    # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(source=frame, classes=[2], show=True)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame using Streamlit's st.image
            frame_placeholder.image(frame, channels="RGB")

                    # Visualize the results on the frame
            #annotated_frame = results[0].plot()

        # Display the annotated frame
            #frame = cv2.imshow("YOLOv8 Inference", annotated_frame)
            #frame_placeholder.image(frame)
            
            # results = model.predict(source="0", classes=[0], stream=True, show=True)
            # results.save_crop() 
            # Visualize the results on the frame
            #annotated_frame = results[0].plot()

            # Display the annotated frame
            #cv2.imshow("YOLOv8 Inference", annotated_frame)



            for i, r in enumerate(results):
            
                # Visualize the results on the frame
                # Save result image
                array = r.plot(conf=False, labels=False, probs=False)

                # Save image to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                    cv2.imwrite(temp.name, array)


                # To demonstrate, let's open this image again
                # temp_image = Image.open(temp.name)
                #temp_image.show()

                # OCR text extraction
    
            # Read and english
                plate_num = reader.readtext(temp.name, detail = 0, paragraph=True)

                # Convert list to string
                plate_num = ''.join(plate_num)

                # Check if plate number already exists in DataFrame
                if plate_num not in df['Plate_number'].values:
            # Create a DataFrame from the value
                    new_row_df = pd.DataFrame([plate_num], columns=df.columns)

            # Concatenate the new DataFrame with the existing DataFrame
                    df = pd.concat([df, new_row_df], ignore_index=True)
                    
                print(df.head())

                # Display the DataFrame
                df_placeholder.write(df)

            stop_button_pressed = st.button("Stop")

            # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break                

        else:
            # Break the loop if the end of the video is reached
            break

        # Display DataFrame
        st.write(df)

if __name__ == "__main__":
    main()