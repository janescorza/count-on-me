# Count On Me (to learn how to count ✨) 🎓✋
**"Count On Me"** is not just a learning tool; it's an adventure into the world of numbers and neural networks made fun and interactive for young learners. Using nothing more than a simple webcam, users can learn how to express numbers through hand signs, and an efficient neural network will cheer them on with every correct guess!

## Features 🌟
- **Identify the Number Mode**: A playful challenge that tests learners' ability to recognize numbers from hand signs, and the neural network's ability to identify the numbers shown, making it a small competition between the learner and AI.
- **Practice Hand Signs Mode**: A fun way to encourage learners to express numbers through hand signs, with immediate AI feedback.

## Getting Started 🚀
Ready to dive into the fun? Here’s how to get the ball rolling:

1. **Clone the Repository**: Run `git clone https://github.com/your-username/count-on-me.git`
2. **Install Requirements**: Execute `pip install -r requirements.txt` to get the necessary libraries.
3. **Launch the Application**: Kick off your adventure with `python main.py`.

## Prerequisites 📋
You'll need Python 3.6 or later, and make sure your computer is equipped with a webcam to interact in real-time!

## Modes of Operation 🔄
### Mode 1: Identify the Number
Interact with the program to go through hand signs, and guess the numbers they represent. It’s like a visual quiz that keeps minds engaged and learning! Then , the program will display the correct number for feedback while also showing what the AI predicted.

### Mode 2: Perform the Hand Sign
Now, flip the script! The AI picks a number, and you show the hand sign. It's your move to shine and show what you've learned. The Neural Network will evaluate the hand sign you show and display its prediction for the number you are showing. 

## Dataset and Neural Network Considerations 🧠
**Count On Me** is driven by a neural network trained on **1800 examples** from a consistent set of data. Understanding the importance of data distribution in training neural networks is crucial:
- **Webcam Input vs. Training Data**: Since the images captured from a webcam can differ significantly in terms of quality, background, and lighting compared to the training data. This variation can lead to discrepancies in model performance since neural networks perform best on data similar to their training set. Future enhancements may involve collecting a more diverse dataset from various distributions to enhance the model's accuracy and robustness, especially when analyzing images from webcams.

- **Hand_per_label Folder**: This folder contains images from the test set, which the model has not seen during training. By testing the model with these images, we can evaluate its adaptability and generalization to new, unseen conditions. This is a crucial step in demonstrating how the model performs in real-world scenarios where variability is much higher than in the controlled environment of the training dataset.


## How It Works 🔍
Drawing from what I learned during the Deep Learning Specialization and building my first deep learning project, the puppy-detector, where I crafted a neural network from scratch, "Count On Me" levels up the game using TensorFlow. This project not only demostrates my grasp on neural networks but also sets the stage for my next adventures with convolutional neural networks and word embeddings. Stay tuned!

## Contributing 🤝
Got ideas or suggestions? I'm all ears! Feel free to fork the repository, submit pull requests, or send over your thoughts on improvements.
