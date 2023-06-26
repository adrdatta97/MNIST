# MNIST
Machine Learning on MNIST Dataset

<p align="center">
  <strong><em>"The key to artificial intelligence has always been the representation."</em></strong><br>
  — Jeff Hawkins
</p>

The MNIST dataset, a subset of the NIST dataset, exemplifies this progress as machines approach human-level performance. In this study, a Python-based neural network model is presented, achieving 92% accuracy in recognizing handwritten digits. By eschewing higher-level APIs like TensorFlow and Keras, the model showcases the fundamentals of machine learning. The work encompasses detailed data exploration, processing techniques, neural network fundamentals, training, validation, and concludes with key insights.

Classification machine learning model created to recognize handwritten digits from the MNIST dataset.
The dataset used in this work contains 41,000 images of 28x28 resolution, obtained from the QMUL portal.

<p align="left">
  <strong><em>Data Preprocessing Techniques : </em></strong>

<b>Shuffling</b> : Randomly shuffling the data to ensure a fair mix of instances in the training and cross-validation sets.<br>
<b>Train/Test sets</b> : Splitting the dataset into 80% training and 20% cross-validation sets. <br>
<b>Normalization </b> : Scaling pixel values from the range of 0-255 to 0-1 to facilitate gradient calculations. <br>
<b>Transposition </b> : Transposing the train set vectors to match the input format required by the neural network model.<br>
<b>One hot encoding </b>: Encoding labels as vectors with a "1" in the index corresponding to the digit and "0" elsewhere. <br>


Processed dataset ready for neural network training:<br>
X train shape: (784, 33,600) <br>
Y train shape: (33,600,)<br>
X val shape: (784, 8,400)<br>
Y val shape: (8,400,)<br>

</p>


* Deep learning is a branch of machine learning that revolves around training interconnected neural networks. These networks, made up of layers of neurons, have the remarkable ability to map inputs to outputs. By adjusting the weight coefficients of the connections between neurons, the networks learn and improve their performance over time. It's like teaching a complex system to understand and interpret data, enabling it to make accurate predictions and decisions.


  * <b> Forward Propagation </b> - In the process of forward propagation in neural networks, each neuron performs a calculation by combining its inputs and a bias term, and then applies an activation function, such as ReLU. This step helps in capturing non-linear relationships and optimizing the network's performance. The resulting output is evaluated using a loss function, like cross-entropy, which measures the deviation between predicted and true values. This procedure, known as forward propagation, plays a crucial role in training and classifying data within neural networks.


  * <b> Back Propagation </b> - The back-propagation algorithm is like the backbone of a neural network. It involves calculating gradients of the loss with respect to the weights and biases, and adjusting them in the right direction to reach the best possible outcome. It's like fine-tuning the network's parameters to improve its performance. This process, represented by equations and diagrams, helps the network learn and make better predictions.


  * <b> Training and Validation </b> - In this project, we randomly initialize and scale the weights and biases of the neural network. We use activation functions like ReLU and softmax to introduce non-linearities and calculate class probabilities. Through forward propagation, we calculate layer activations and measure the loss with the cross-entropy function. Backpropagation helps us adjust the parameters by calculating gradients and using the gradient descent algorithm. We train multiple models with different architectures, evaluating them based on loss and accuracy. The learning rate and number of iterations are carefully chosen to balance convergence and overfitting.


<p align="left">
  <strong><em>Results : </em></strong>
  After training four models for 150 iterations, the best performing model had an architecture of [784, 256, 128, 64, 10] and achieved a test accuracy of 92%. It demonstrated good generalization with a train set accuracy of 92.7%. The least performing model had a simpler architecture with only one hidden layer of 32 neurons and obtained a test accuracy of 83%. The graph of the best model's loss/accuracy showed a plateau around the 50th iteration, with a slight increase at the 80th iteration. The model successfully classified all digits in the validation set with 92% accuracy.
</p>


<p align="center">
  <img width="246" alt="Screenshot 2023-06-26 at 19 17 02" src="https://github.com/adrdatta97/MNIST/assets/117360902/9c447388-10d8-464e-b38d-c362a32266e6">
</p>


