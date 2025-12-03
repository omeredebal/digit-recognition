import numpy as np

class DigitClassifier:
    """TensorFlow olmadan çalışan hafif CNN inference"""
    
    def __init__(self, weights_path='model/weights.npy'):
        weights = np.load(weights_path, allow_pickle=True).item()
        
        # Conv2D Layer 0
        self.conv1_w = np.array(weights['layer_0']['weights'][0])
        self.conv1_b = np.array(weights['layer_0']['weights'][1])
        
        # Conv2D Layer 2
        self.conv2_w = np.array(weights['layer_2']['weights'][0])
        self.conv2_b = np.array(weights['layer_2']['weights'][1])
        
        # Dense Layer 6
        self.dense1_w = np.array(weights['layer_6']['weights'][0])
        self.dense1_b = np.array(weights['layer_6']['weights'][1])
        
        # Dense Layer 8 (output)
        self.dense2_w = np.array(weights['layer_8']['weights'][0])
        self.dense2_b = np.array(weights['layer_8']['weights'][1])
        
        print("✅ Model ağırlıkları yüklendi!")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def conv2d(self, x, weights, bias):
        """3x3 convolution with valid padding"""
        h, w, c_in = x.shape
        kh, kw, _, c_out = weights.shape
        
        out_h = h - kh + 1
        out_w = w - kw + 1
        output = np.zeros((out_h, out_w, c_out))
        
        for i in range(out_h):
            for j in range(out_w):
                region = x[i:i+kh, j:j+kw, :]
                for k in range(c_out):
                    output[i, j, k] = np.sum(region * weights[:, :, :, k]) + bias[k]
        
        return output
    
    def max_pool2d(self, x, pool_size=2):
        """2x2 max pooling"""
        h, w, c = x.shape
        out_h = h // pool_size
        out_w = w // pool_size
        output = np.zeros((out_h, out_w, c))
        
        for i in range(out_h):
            for j in range(out_w):
                region = x[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size, :]
                output[i, j, :] = np.max(region, axis=(0, 1))
        
        return output
    
    def predict(self, x):
        """Forward pass"""
        # Input: (1, 28, 28, 1) -> (28, 28, 1)
        x = x.reshape(28, 28, 1)
        
        # Conv1 + ReLU + MaxPool
        x = self.conv2d(x, self.conv1_w, self.conv1_b)  # (26, 26, 32)
        x = self.relu(x)
        x = self.max_pool2d(x)  # (13, 13, 32)
        
        # Conv2 + ReLU + MaxPool
        x = self.conv2d(x, self.conv2_w, self.conv2_b)  # (11, 11, 64)
        x = self.relu(x)
        x = self.max_pool2d(x)  # (5, 5, 64)
        
        # Flatten
        x = x.flatten()  # (1600,)
        
        # Dense1 + ReLU
        x = np.dot(x, self.dense1_w) + self.dense1_b
        x = self.relu(x)
        
        # Dense2 (output)
        x = np.dot(x, self.dense2_w) + self.dense2_b
        x = self.softmax(x)
        
        return x.reshape(1, -1)
