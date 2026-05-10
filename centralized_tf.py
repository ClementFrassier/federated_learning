import tensorflow as tf
import numpy as np

# Define device
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    DEVICE = "GPU"
else:
    DEVICE = "CPU"

# Télécharge, normalise et prépare les ensembles de données CIFAR-10 pour l'entraînement et le test
# Downloads, normalizes, and prepares the CIFAR-10 datasets for training and testing
def load_data():
    """Load CIFAR-10 (training and test set)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize data between -1 and 1 to match PyTorch's Normalize((0.5,), (0.5,))
    x_train = (x_train.astype('float32') / 127.5) - 1.0
    x_test = (x_test.astype('float32') / 127.5) - 1.0
    
    # Create tf.data.Dataset
    trainloader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(32)
    testloader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
    return trainloader, testloader

# Construit, compile et retourne l'architecture du réseau de neurones (équivalent à la classe Net PyTorch)
# Builds, compiles, and returns the neural network architecture (equivalent to PyTorch Net)
def load_model():
    """Returns an instance of our Net model initialized and ready to run."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10) # No softmax, using from_logits=True in loss
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

# Entraîne le modèle sur les données d'entraînement pour un nombre d'époques donné
# Trains the model on the training data for a specified number of epochs
def train(net, trainloader, epochs):
    """Train the model on the training set."""
    net.fit(trainloader, epochs=epochs)

# Évalue les performances du modèle sur les données de test et retourne la perte (loss) et la précision (accuracy)
# Evaluates model performance on the test set and returns loss and accuracy
def test(net, testloader):
    """Validate the model on the test set."""
    loss, accuracy = net.evaluate(testloader, verbose=0)
    return loss, accuracy

if __name__ == "__main__":
    print(f"Hardware computing initialized on: {DEVICE}")
    model = load_model()
    trainloader, testloader = load_data()
    
    print("Starting training...")
    train(model, trainloader, epochs=2)
    
    print("Starting evaluation...")
    loss, accuracy = test(model, testloader)
    print(f"Final test loss: {loss:.4f}")
    print(f"Final test accuracy: {accuracy:.4f}")
