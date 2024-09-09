# Image Processing
<p>Iã Jensen </p>
<p>ece.iajensen@gmail.com</p>

## Description
This project will take a jpg image and apply a filter onto it. There are three to choose from: blur, sharpen, and black-and-white. Sharpen comes with a strength modifier which sharepns the image more or less depending on what is needed.

## Usage
imageprocessing.py comes with a built in usage statement when activated from the command line.
![image](https://github.com/user-attachments/assets/5c315b46-c63f-4e16-ab3a-f8c46d631f5f)
As shown in the above image, all you have to do is add the type of filter you want (blur, sharpen, or bw), the image's file path (easiest to have it in the same folder) you want to transform, and the name of the output image.
If you are using the sharpen function, make sure to add a strength modifier (default is 1.0).

## Installations
For this application, make sure to have pandas, imageio, and numpy installed.
<p>If not, go to the commandline and type "pip install {package}" and replace {package} with the package you want installed.</p>

![image](https://github.com/user-attachments/assets/d32d84f5-72ee-434a-aafb-1c8917bbeb13)

After you have installed all the necessary packages, you can finally use the application.

## Example

![Wires](https://github.com/user-attachments/assets/c1b3b297-bdf3-46d0-9162-ccacd7abb7bf)

Black and White

![Wires_bw](https://github.com/user-attachments/assets/9f2ae9e8-7f3c-4248-84ec-0d92e8b19fed)

Blur

![Wires_blur](https://github.com/user-attachments/assets/0d5495c1-eb13-4439-8f63-33c4513c5fae)

Sharpen (default)

![Wires_sharpen](https://github.com/user-attachments/assets/4906b6a5-bdde-4b4d-90b2-ec93171effe2)

Sharpen (strength of 3)

![Wires_sharpen_3](https://github.com/user-attachments/assets/5f446f48-e887-4139-8945-72ac13fc9147)

## Work in Progress
- Currently working on a way to blur and sharpen non black-and-white images
- Attempting to allow any gausian kernel to be implemeneted which would allow more complex image manipulation.
