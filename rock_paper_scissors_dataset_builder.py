import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

# Carregar o conjunto de dados Rock, Paper, Scissors
builder = tfds.builder("rock_paper_scissors")
builder.download_and_prepare()
dataset = builder.as_dataset(split="train")

# Pré-processamento
def preprocess_image(image, label):
    # Redimensionar as imagens para o tamanho desejado (300x300)
    image = tf.image.resize(image, (300, 300))
    # Normalizar os valores dos pixels para o intervalo [0, 1]
    image = image / 255.0
    return image, label

# Aplicar pré-processamento ao conjunto de dados
dataset = dataset.map(preprocess_image)

# Dividir o conjunto de dados em treinamento, validação e teste
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = int(0.1 * len(dataset))

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size).take(test_size)

# Data Augmentation para o conjunto de treinamento
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Definir a arquitetura da CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
train_generator = datagen.flow_from_dataframe(
    dataframe=train_dataset,
    x_col='image',
    y_col='label',
    target_size=(300, 300),
    batch_size=32,
    class_mode='raw'
)

val_generator = val_dataset.batch(32)

model.fit(train_generator,
          validation_data=val_generator,
          epochs=10)

# Avaliação no conjunto de teste
test_generator = test_dataset.batch(32)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')
