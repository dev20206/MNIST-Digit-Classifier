# MNIST Digit Classifier

MNIST Digit Classifier is a comprehensive project that integrates a deep learning model for handwritten digit recognition with an interactive drawing application built using Pygame. This project serves as an educational and practical tool for exploring machine learning concepts and building real-time applications for handwritten digit recognition.

## Features

- **Neural Network Model**: The project includes a convolutional neural network (CNN) model trained on the MNIST dataset, capable of accurately classifying handwritten digits from zero to nine. The model architecture is implemented using PyTorch, providing flexibility and efficiency in training and inference.

- **Interactive Drawing Application**: Utilizing the Pygame library, the project offers an intuitive drawing interface where users can draw digits directly on the screen using a mouse or touch input. As users draw, the application displays white circles, simulating the act of drawing with a pen.

- **Real-Time Recognition**: Upon releasing the mouse button after drawing a digit, the application captures the drawn area and preprocesses it. The preprocessed image is then fed into the pre-trained neural network model for digit recognition. In real-time, the model predicts the digit represented by the user's drawing, providing instant feedback to the user.

- **User-Friendly Interface**: The drawing application features a user-friendly interface with options to save drawn images for further analysis or training, clear the drawing board, and interact with the model seamlessly.

## Usage

To use the MNIST Digit Classifier:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Run the `app.py` script to launch the drawing application.
4. Draw digits on the screen using the mouse or touch input.
5. Release the mouse button after drawing a digit to see the model's prediction in real-time.
6. Explore additional features such as saving drawn images and clearing the drawing board.

## Acknowledgments

Special thanks to the creators of the PyTorch and Pygame libraries for providing powerful tools for deep learning and game development, respectively. Additionally, we acknowledge the contributors to the MNIST dataset, which has been instrumental in advancing research in machine learning and computer vision.
