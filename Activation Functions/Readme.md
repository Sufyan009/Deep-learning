## Activation Functions in Neural Networks

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Here are some common activation functions:

### 1. **Sigmoid Function**
* Maps input values to a range between 0 and 1.
* Used in logistic regression and older neural networks.
* **Formula:**
  ```
  sigmoid(x) = 1 / (1 + exp(-x))
  ```
* **Limitation:** Can suffer from the vanishing gradient problem for large negative or positive inputs.

### 2. **Tanh (Hyperbolic Tangent)**
* Maps input values to a range between -1 and 1.
* Often used in hidden layers of neural networks.
* **Formula:**
  ```
  tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
  ```
* **Advantage:** Zero-centered output, which can improve learning.

### 3. **ReLU (Rectified Linear Unit)**
* Maps negative input values to 0 and positive values to themselves.
* Widely used in deep neural networks.
* **Formula:**
  ```
  ReLU(x) = max(0, x)
  ```
* **Advantages:** Efficient to compute, avoids the vanishing gradient problem.

### 4. **Leaky ReLU**
* Similar to ReLU, but allows a small gradient for negative inputs.
* Helps to address the "dying ReLU" problem.
* **Formula:**
  ```
  LeakyReLU(x) = max(αx, x)
  ```
  where `α` is a small positive number (e.g., 0.01).

### 5. **ELU (Exponential Linear Unit)**
* Combines the advantages of ReLU and Leaky ReLU.
* Can handle the vanishing gradient problem and provides a non-zero mean output.
* **Formula:**
  ```
  ELU(x) = {
      x, if x > 0
      α * (exp(x) - 1), otherwise
  }
  ```

**Choosing the Right Activation Function:**

* **ReLU** is a popular choice for many deep learning architectures due to its simplicity and effectiveness.
* **Leaky ReLU** and **ELU** can be used to mitigate the vanishing gradient problem.
* **Sigmoid** and **Tanh** are less commonly used in modern deep learning models.

The choice of activation function depends on the specific task and network architecture. Experimentation and careful consideration of the problem domain are key to selecting the best activation function.
