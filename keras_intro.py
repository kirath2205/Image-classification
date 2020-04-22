import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Flatten
def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    if(training):
        return train_images,train_labels
    elif(not training):
        return test_images,test_labels

def print_stats(images,labels):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    number_labels=[0,0,0,0,0,0,0,0,0,0]
    print(len(images))
    print(len(images[0]),"x",len(images[0][0]))
    for i in labels:
        number_labels[i]=number_labels[i]+1
    for index in range(len(class_names)):
        print(index,". ",class_names[index],"-",number_labels[index])

def view_image(image,label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    k1=axs.imshow(image,aspect='equal')
    axs.set_title(class_names[label])
    fig.colorbar(k1, ax=axs,shrink=1)
    plt.show(axs)

def build_model():
    model=tf.keras.models.Sequential()
    model.add(keras.layers.Dense(128,input_shape=(28,28),activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10))
    model.compile('adam',keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    return model


def train_model(model,images,label,T):
    model.fit(images,label,epochs=T)

def evaluate_model(model,images,labels,show_loss=True):
    test_loss,test_accuracy=model.evaluate(images,labels,verbose=0)
    if(show_loss):
        print("Loss:",round(test_loss,2))
    print("Accuracy:",round(test_accuracy*100,2),"%")
    print()

def predict_label(model,images,index):
    model.add(keras.layers.Softmax())
    k=model.predict(images)
    prediction=(k[index]).copy()
    labels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    for i in range(len(labels)):
        for j in range(0,len(labels)-1):
            if(prediction[j]<prediction[j+1]):
                t=prediction[j]
                prediction[j]=prediction[j+1]
                prediction[j+1]=t
                t2=labels[j]
                labels[j]=labels[j+1]
                labels[j+1]=t2
    print(labels[0],":",round((prediction[0]*100),2))
    print(labels[1],":",round((prediction[1]*100),2))
    print(labels[2],":",round((prediction[2]*100),2))

#model=build_model()
#images,labels=get_dataset(True)
#view_image(images[9],labels[9])
#train_model(model,images,labels,5)
#test_images,test_labels=get_dataset(False)

#evaluate_model(model,test_images,test_labels,True)
#predict_label(model,test_images,0)
#print_stats(images,labels)


