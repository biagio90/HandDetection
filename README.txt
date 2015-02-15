Hand traking by Biagio Brattoli.

This software can detect the hand using the webcam. It is a simple version with no memory that uses the skin color to identify the hand's pixels.

Steps:
- Run the program
- Position the hand with the palm over the sample squares
- press any button
- The hand contour is drawn in red
- press a button to close

Weakness:
- It works better with high contrast background
- The face can interfere

This software needs OpenCV in order to compile.
Compile using:
g++ -o"HandTracking"  ./HandTracking.cpp -lopencv_highgui -lopencv_imgproc -lopencv_core
