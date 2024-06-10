from flask import Flask, render_template, request, redirect, url_for, flash
import random
import markdown
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = "your_secret_key"

conceptual_objectives = {
        "lecture_1": [
            "Explain the challenges a computer vision algorithm is facing when dealing with real variations of an object in images",
            "Explain the representation of a coloured image as 3-dimensional array",
            "Describe every step of the data-driven approach in image classification",
            "Explain the training and prediction procedure of the Nearest Neighbor Classifier",
            "Explain the role and differences of the distance metrics L1 and L2 in Nearest Neighbor Classifiers",
            "Explain the difference between a k-nearest neighbor classifier and a nearest neighbor classifier with respect to classification accuracy and overfitting",
            "Explain the role of hyperparameters in the case of a k-nearest neighbor classifier and how to tune them",
            "Explain the advantages and disadvantages of k-fold cross validation versus using the train/validation/test split and how to properly use the test set"
        ],
        "lecture_2": [
            "Identify and explain the role of the major components of linear classifiers, in particular the score function, the loss function and the ground truth",
            "Write down the score function of a linear classifier and explain why it is linear",
            "Illustrate the score function of a linear classifier in the case of CIFAR-10 images projected on a 2-dimensional plane",
            "Interpret the rows of the learned weight matrix of a linear classifier as template (images) for each class",
            "Explain the structure of any loss function",
            "Explain the role of the loss function in the learning process and how it affects the weights",
            "Explain how the softmax function applied to the score values allows to interpret the score values as unnormalized log-probabilities",
            "Define the cross-entropy loss of a softmax classifier",
            "Explain how a regularization term in the loss function is influencing the values of the weight matrix and how this may control overfitting",
            "Explain the goal of optimization, in particular to understand which parts of a score function are fixed and which parameters can be learned and how this is carried out with respect to the loss function",
            "Explain the meaning of a gradient as the direction of steepest descent and how this relates to optimization",
            "Name three activation functions and explain the advantages and disadvantages of these activation functions with respect to the optimization process",
            "Recognize the topology of fully connected neural networks in terms of number of hidden layers and hidden units",
            "Count the number of learnable parameters in a given fully connected neural network",
            "Explain how to optimize the number of hidden layers and hidden units and how they may affect the problem of overfitting",
            "Explain how backpropagation is related to the optimization process of a neural network"
        ],
        "lecture_3": [
            "Explain Batch Normalization and how batch normalization  contributes to increase the performance of neural networks",
            "Draw the correct conclusions from the shape of the loss function with respect to the need for an appropriate adaptation of the learning rate and of the batch size",
            "Draw the correct conclusions fro the shape of the accuracy curve with respect to overfitting and to the need for regularization",
            "Explain the working principles of dropout and why it may increase robustness of a neural network and prevent training from overfitting"
        ],
        "lecture_4": [
            "Explain how the input and output dimensions of ConvNets differ from the input and output dimensions of fully connected neural networks",
            "Explain how filters of the ConvNets compute the activation map",
            "Explain how the number of filters in a convolutional layer is related to the dimension of the activation volume",
            "Compute the spatial arrangement of the output volume as a function of the number of filters, the stride and the amount of zero-padding",
            "Compute the required amount of zero-padding in order to obtain the same output volume as the input volume for a given receptive field and stride 1",
            "Explain the assumption underlying paramater sharing of all neurons in a depth slice by using the same filter weights and the same bias",
            "Explain the function of pooling layers",
            "Explain how max pooling is affecting the size of the output volume",
            "Know layer patterns of CNN architectures",
            "Explain how transfer learning is applied"
        ],
        "lecture_5": [
            "Explain how to prepare data by means of standardization, tokenization, indexing and one-hot-encoding or word-embeddings",
            "Define the term language model and describe its purpose",
            "Explain the constituents and structure of a RNN",
            "Explain the advantages (e.g. with respect to N-grams) and disadvantages of RNNs",
            "Explain the vanishing gradient problem of RNNs and how LSTM is solving this issue"
        ],
        "lecture_6": [
            "Explain the structure of sequence-to-sequence modeling and how it is relevant for machine translation",
            "Explain attention and its advantages with respect to the bottleneck and vanishing gradient problem",
            "Explain self-attention and how it relates to attention",
            "Relate query, keys and values with input sequence and output sequence in a sentiment classification problem",
            "Relate query, keys and values with source sequence and target sequence in a machine translation problem",
            "Explain the structure of multi-headed attention layer",
            "List those NLP models that are aware of order",
            "Explain positional embedding",
            "Explain the structure of Transformer decoder",
            "Explain the functionality of causal padding and the reason why it needs to be integrated"
        ]
}


# Function to get a random question
def get_random_question():
    lecture = random.choice(list(conceptual_objectives.keys()))
    question = random.choice(conceptual_objectives[lecture])
    return question


# Home route to display a random question
@app.route('/')
def home():
    question = get_random_question()
    return render_template('index.html', question=question)


# Route to handle generating a new question
@app.route('/new_question')
def new_question():
    question = get_random_question()
    return render_template('index.html', question=question)


# Route to handle answer submission and feedback
@app.route('/submit', methods=['POST'])
def submit():
    question = request.form['question']
    answer = request.form['answer']

    feedback = get_llm_feedback(question, answer)

    return render_template('feedback.html', question=question, answer=answer, feedback=feedback)


def get_llm_feedback(question, answer):
    from llama_index.llms.groq import Groq
    llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

    feedback_markdown = llm.complete(f"""
    Given a following question and an answer, provide feedback on the answer and suggest improvements.
    Here follows the Question:{question}
    And here follows the Answer: {answer}\n
    """)
    feedback_html = markdown.markdown(feedback_markdown.text, extensions=["extra"])
    return feedback_html


if __name__ == '__main__':
    app.run(debug=True)
