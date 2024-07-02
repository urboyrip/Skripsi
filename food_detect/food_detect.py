# In[ ]:

import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import splitfolders
import tkinter as tk
from calorie_calculator import CalorieCalculatorApp
from food_identifier import CalorieApp
from tutorial import TutorApp

tf.keras.backend.clear_session()

# In[ ]:

# Masukkan Path dataset
dataset_dir = 'D:/unair/Bangkit/konversi datmin/dataset gambar/Dataset'

# Split direktori dataset menjadi dua folder train dan val 
splitfolders.ratio(dataset_dir, output="food-data", seed=1337, ratio=(.7, .3), group_prefix=None)

# Path data train dan data test
training_dir = os.path.join('food-data/', 'train')
testing_dir = os.path.join('food-data/', 'val')

# In[ ]:

# Membuat imagedatagenerator
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
        training_dir,
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
    )

validation_generator = validation_datagen.flow_from_directory(
        testing_dir,
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
    )

pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

# Index nama-nama kelas makanan
class_indices = train_generator.class_indices
print(class_indices)
class_names = list(class_indices.keys())
print(class_names)
num_class = len(class_names)
print(num_class)
# In[ ]:

# Membuat class callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.96):
            print("\nReached 96% accuracy so cancelling training!")
            self.model.stop_training = True

# Membuat model
pretrained_model.trainable = False

inputs = pretrained_model.input

x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

outputs = tf.keras.layers.Dense(num_class, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

print(model.summary())

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[myCallback()]
)

# Plot grafik akurasi model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

# Plot grafik loss model
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

model.save_weights('foodModel_height1.h5')
print('Model Saved!')
model.save('food_model_all.h5')
print('Model Saved!')

# In[ ]:

# Testing model
df = pd.read_excel('D:/unair/Bangkit/konversi datmin/dataset_kalori_makanan.xlsx')
model = tf.keras.models.load_model('D:/unair/Bangkit/konversi datmin/food_detect/food_model_all.h5')
obj_tes = 'D:/unair/Bangkit/konversi datmin/buat tes/susu.jpg'

img = tf.keras.utils.load_img(
    obj_tes, target_size=(224, 224)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array/255.0
img_array = tf.expand_dims(img_array, axis = 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

predicted_class = np.argmax(predictions)
predicted_class_name = class_names[predicted_class]

nama = 'Nama'
kalori = 'Kalori_100gr'
nilai_kolom2 = 0

for index, row in df.iterrows():
    if row[nama] == predicted_class_name:
        nilai_kolom2 = row[kalori]
        break
    
print(
    "Ini adalah gambar {} dengan confidence sebesar {:.2f}"
    .format(predicted_class_name, 100 * np.max(score))
)

if nilai_kolom2 is not None:
    print("Dalam 100gr {} terdapat {} kkal".format(predicted_class_name,nilai_kolom2))
else:
    print("Tidak ada nilai yang sesuai.")

# In[ ]:

df = pd.read_excel('D:/unair/Bangkit/konversi datmin/dataset_kalori_makanan.xlsx')
model = tf.keras.models.load_model('D:/unair/Bangkit/konversi datmin/food_detect/food_model_all.h5')

kalori_user = float(input("Masukkan jumlah kalori harian anda : "))
print("Kalori harian anda adalah : ",kalori_user)

obj_tes = input("Masukkan path gambar yang ingin anda identifikasi : ")
# D:/unair/Bangkit/konversi datmin/buat tes/susu.jpg

img = tf.keras.utils.load_img(
    obj_tes, target_size=(224, 224)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array/255.0
img_array = tf.expand_dims(img_array, axis = 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

predicted_class = np.argmax(predictions)
predicted_class_name = class_names[predicted_class]

nama = 'Nama'
kalori = 'Kalori_100gr'
nilai_kolom2 = 0

for index, row in df.iterrows():
    if row[nama] == predicted_class_name:
        nilai_kolom2 = row[kalori]
        break
    
print(
    "Ini adalah gambar {} dengan confidence sebesar {:.2f}"
    .format(predicted_class_name, 100 * np.max(score))
)

if nilai_kolom2 is not None:
    print("Dalam 100gr {} terdapat {} kkal".format(predicted_class_name,nilai_kolom2))
else:
    print("Tidak ada nilai yang sesuai.")

while kalori_user >= nilai_kolom2 :
    kalori_user  = kalori_user - nilai_kolom2
    print("sisa kalori hari ini : ",kalori_user)

    jawab = input("Apakah anda ingin memasukkan gambar lain? (ya/tidak) : ")

    if jawab == 'ya':

        df = pd.read_excel('D:/unair/Bangkit/konversi datmin/dataset_kalori_makanan.xlsx')
        model = tf.keras.models.load_model('D:/unair/Bangkit/konversi datmin/food_detect/food_model_all.h5')

        obj_tes = input("Masukkan path gambar yang ingin anda identifikasi : ")

        # D:/unair/Bangkit/konversi datmin/buat tes/susu.jpg

        img = tf.keras.utils.load_img(
            obj_tes, target_size=(224, 224)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array/255.0
        img_array = tf.expand_dims(img_array, axis = 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class]

        nama = 'Nama'
        kalori = 'Kalori_100gr'
        nilai_kolom2 = 0

        for index, row in df.iterrows():
            if row[nama] == predicted_class_name:
                nilai_kolom2 = row[kalori]
                break
            
        print(
            "Ini adalah gambar {} dengan confidence sebesar {:.2f}"
            .format(predicted_class_name, 100 * np.max(score))
        )

        if nilai_kolom2 is not None:
            print("Dalam 100gr {} terdapat {} kkal".format(predicted_class_name,nilai_kolom2))
        else:
            print("Tidak ada nilai yang sesuai.")

    else :
        print("sisa kalori hari ini : ",kalori_user)
        break
    
print("Pastikan untuk memenuhi kalori harian anda")

# In[ ]:

'''
import tkinter as tk
from tkinter import messagebox

def calculate_calories():
    bb = float(entry_bb.get())
    tb = float(entry_tb.get())
    jk = float(entry_jk.get())
    umur = float(entry_umur.get())
    kegiatan = float(entry_kegiatan.get())

    # Kebutuhan Kalori Utama
    if jk == 0:
        kk = 30 * bb
    elif jk == 1:
        kk = 25 * bb

    # Faktor Kegiatan
    if kegiatan == 0:
        kk_keg = kk + (kk * (10 / 100))
    elif kegiatan == 1:
        kk_keg = kk + (kk * (20 / 100))
    elif kegiatan == 2:
        kk_keg = kk + (kk * (30 / 100))
    elif kegiatan == 3:
        kk_keg = kk + (kk * (40 / 100))
    elif kegiatan == 4:
        kk_keg = kk + (kk * (50 / 100))

    # Faktor Umur
    kk_um = 0
    if 59 >= umur >= 40:
        kk_um = (kk - (kk * (5 / 100)))
    elif 69 >= umur >= 60:
        kk_um = (kk - (kk * (10 / 100)))
    elif umur >= 70:
        kk_um = (kk - (kk * (20 / 100)))

    # Faktor IMT
    imt = bb / ((tb / 100) * (tb / 100))
    if imt < 18.5:
        kk_imt = (kk + (kk * (25 / 100)))
    elif 22.9 >= imt >= 18.5:
        kk_imt = kk
    elif imt >= 23:
        kk_imt = (kk - (kk * (25 / 100)))

    # Jumlah Kebutuhan Kalori Akhir
    if kk_um == 0:
        kk_akhir = ((kk + kk_keg + kk_imt) / 3)
    elif kk_um != 0:
        kk_akhir = ((kk + kk_keg + kk_um + kk_imt) / 4)

    messagebox.showinfo("Hasil Perhitungan", f"IMT Anda adalah: {imt}\nKebutuhan kalori akhir Anda adalah: {kk_akhir}")

# Membuat window aplikasi
root = tk.Tk()
root.title("Perhitungan Kebutuhan Kalori")

# Membuat label dan entry untuk memasukkan data
label_bb = tk.Label(root, text="Masukkan Berat Badan (kg):")
label_bb.grid(row=0, column=0)
entry_bb = tk.Entry(root)
entry_bb.grid(row=0, column=1)

label_tb = tk.Label(root, text="Masukkan Tinggi Badan (cm):")
label_tb.grid(row=1, column=0)
entry_tb = tk.Entry(root)
entry_tb.grid(row=1, column=1)

label_jk = tk.Label(root, text="Masukkan Jenis Kelamin (Pria = 0, Wanita = 1):")
label_jk.grid(row=2, column=0)
entry_jk = tk.Entry(root)
entry_jk.grid(row=2, column=1)

label_umur = tk.Label(root, text="Masukkan Umur:")
label_umur.grid(row=3, column=0)
entry_umur = tk.Entry(root)
entry_umur.grid(row=3, column=1)

label_kegiatan = tk.Label(root, text="Masukkan Kegiatan (Pensiunan/Tidak Bekerja = 0, Ringan = 1, Sedang = 2, Berat = 3, Sangat Berat = 4):")
label_kegiatan.grid(row=4, column=0)
entry_kegiatan = tk.Entry(root)
entry_kegiatan.grid(row=4, column=1)

# Membuat tombol untuk melakukan perhitungan
button = tk.Button(root, text="Hitung", command=calculate_calories)
button.grid(row=5, columnspan=2)

# Menjalankan aplikasi
root.mainloop()
'''
# In[]
'''
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import tensorflow as tf
import numpy as np

def browse_image():
    global obj_tes
    obj_tes = filedialog.askopenfilename(initialdir="/", title="Pilih Gambar", filetypes=[("Image Files", "*.jpg; *.jpeg; *.png")])
    label_image.config(text=obj_tes)

def calculate_calories():
    global kalori_user
    kalori_user = float(entry_kalori.get())
    label_kalori.config(text=f"Kalori harian anda adalah : {kalori_user} kkal")

def identify_image():
    global obj_tes, kalori_user

    if kalori_user is None:
        messagebox.showerror("Error", "Silakan masukkan jumlah kalori harian anda terlebih dahulu.")
        return

    if obj_tes is None:
        messagebox.showerror("Error", "Silakan pilih gambar terlebih dahulu.")
        return

    df = pd.read_excel('D:/unair/Bangkit/konversi datmin/dataset_kalori_makanan.xlsx')
    model = tf.keras.models.load_model('D:/unair/Bangkit/konversi datmin/food_detect/food_model_all.h5')

    img = tf.keras.utils.load_img(
        obj_tes, target_size=(224, 224)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array/255.0
    img_array = tf.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]

    nama = 'Nama'
    kalori = 'Kalori_100gr'
    nilai_kolom2 = 0

    for index, row in df.iterrows():
        if row[nama] == predicted_class_name:
            nilai_kolom2 = row[kalori]
            break

    result_label.config(text=f"Ini adalah gambar {predicted_class_name} dengan confidence sebesar {100 * np.max(score):.2f}")

    if nilai_kolom2 is not None:
        result_label2.config(text=f"Dalam 100gr {predicted_class_name} terdapat {nilai_kolom2} kkal")
    else:
        result_label2.config(text="Tidak ada nilai yang sesuai.")

    while kalori_user >= nilai_kolom2:
        kalori_user -= nilai_kolom2
        result_label3.config(text=f"Sisa kalori hari ini : {kalori_user:.2f} kkal")
        break
        
    if kalori_user == 0:
        messagebox.showinfo("Informasi", "Kebutuhan kalori harian anda sudah tercukupi. Pastikan untuk memenuhi kebutuhan gizi lainnya.")

    elif kalori_user < nilai_kolom2:
        messagebox.showwarning("Peringatan", "Kalori yang dimasukkan lebih besar dari kebutuhan kalori harian. Harap periksa kembali.")



# Membuat window aplikasi
root = tk.Tk()
root.title("Identifikasi Makanan dan Hitung Kalori")

# Membuat label dan tombol untuk memilih gambar
browse_button = tk.Button(root, text="Pilih Gambar", command=browse_image)
browse_button.grid(row=0, column=0)

label_image = tk.Label(root, text="Path Gambar:")
label_image.grid(row=0, column=1)

# Membuat label dan entry untuk memasukkan jumlah kalori
label_kalori = tk.Label(root, text="Masukkan Jumlah Kalori Harian:")
label_kalori.grid(row=1, column=0)

entry_kalori = tk.Entry(root)
entry_kalori.grid(row=1, column=1)

calculate_button = tk.Button(root, text="Hitung Kalori", command=calculate_calories)
calculate_button.grid(row=1, column=2)

# Membuat tombol untuk melakukan identifikasi gambar
identify_button = tk.Button(root, text="Identifikasi Gambar", command=identify_image)
identify_button.grid(row=2, column=0, columnspan=3)

# Label untuk hasil identifikasi dan perhitungan kalori
result_label = tk.Label(root, text="")
result_label.grid(row=3, column=0, columnspan=3)

result_label2 = tk.Label(root, text="")
result_label2.grid(row=4, column=0, columnspan=3)

result_label3 = tk.Label(root, text="")
result_label3.grid(row=5, column=0, columnspan=3)

# Menjalankan aplikasi
root.mainloop()

'''

# In[ ]:

# Membuat class callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > 0.96):
            print("\nReached 96% accuracy so cancelling training!")
            self.model.stop_training = True

# Membuat early stopping apabila vall_loss terlalu besar
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Menggunakan validation loss untuk monitoring
    patience=5,           # Tunggu 5 epoch sebelum berhenti jika tidak ada peningkatan
    verbose=1,            
    restore_best_weights=True  # Mengembalikan bobot terbaik ketika berhenti
)

# Membuat model
pretrained_model.trainable = False

inputs = pretrained_model.input

x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dropout(0.3)(x)  # Dropout layer dengan tingkat dropout 50%
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)  # Dropout layer dengan tingkat dropout 50%

outputs = tf.keras.layers.Dense(num_class, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

print(model.summary())

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.TruePositives(name='tp'),
             tf.keras.metrics.FalsePositives(name='fp'),
             tf.keras.metrics.TrueNegatives(name='tn'),
             tf.keras.metrics.FalseNegatives(name='fn')
            ]
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[myCallback(),early_stopping]
)

# Plot grafik akurasi model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss'] 
precision = history.history['precision']
val_precision = history.history['val_precision']
recall = history.history['recall']
val_recall = history.history['val_recall']
f1_score = [2*(p*r)/(p+r+1e-10) for p, r in zip(precision, recall)]
val_f1_score = [2*(p*r)/(p+r+1e-10) for p, r in zip(val_precision, val_recall)]

epochs = range(len(acc))

# Plot grafik akurasi
plt.figure(figsize=(8, 6))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot grafik loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot grafik presisi
plt.figure(figsize=(8, 6))
plt.plot(epochs, precision, 'r', label='Training precision')
plt.plot(epochs, val_precision, 'b', label='Validation precision')
plt.title('Training and validation precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

# Plot grafik recall
plt.figure(figsize=(8, 6))
plt.plot(epochs, recall, 'r', label='Training recall')
plt.plot(epochs, val_recall, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

# Plot grafik F1-score
plt.figure(figsize=(8, 6))
plt.plot(epochs, f1_score, 'r', label='Training F1-score')
plt.plot(epochs, val_f1_score, 'b', label='Validation F1-score')
plt.title('Training and validation F1-score')
plt.xlabel('Epoch')
plt.ylabel('F1-score')
plt.legend()
plt.show()

plt.tight_layout()
plt.show()

model.save_weights('foodModel_height1.h5')
print('Model Saved!')
model.save('food_model_all.h5')
print('Model Saved!')

# In[]
class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalkulasi Kalori Harian Penderita Diabetes dan Identifikasi Makanan serta Minuman")

        # Deskripsi
        description_label = tk.Label(root, text="Selamat Datang User. \n Silakan Pilih Salah Satu Fungsi di Bawah ini:", padx=10, pady=10, font = ("Times", 20, 'bold'))
        description_label.pack()

        # Button to open Tutorial
        self.tutor_button = tk.Button(root, text="Tata Cara Penggunaan", command=self.open_tutorial, padx=10, pady=5, font = ("Helvetica", 12))
        self.tutor_button.pack(pady=10)

        # Button to open Calorie Calculator window
        self.calorie_button = tk.Button(root, text="Kalkulator Kebutuhan Kalori\nHarian Penderita Diabetes", command=self.open_calorie_calculator, padx=10, pady=5, font = ("Helvetica", 12))
        self.calorie_button.pack(pady=10)

        # Button to open Food Identifier window
        self.food_button = tk.Button(root, text="Pengidentifikasi Makanan dan Minuman", command=self.open_food_identifier, padx=10, pady=5, font = ("Helvetica", 12))
        self.food_button.pack(pady=10)

    def open_calorie_calculator(self):
        # Create a Toplevel window for Calorie Calculator
        calorie_window = tk.Toplevel(self.root)
        self.calorie_app = CalorieCalculatorApp(calorie_window)

    def open_food_identifier(self):
        # Create a Toplevel window for Food Identifier
        food_window = tk.Toplevel(self.root)
        self.food_app = CalorieApp(food_window)
    
    def open_tutorial(self):
        tutor_window = tk.Toplevel(self.root)
        self.tutor_app = TutorApp(tutor_window)

# Run the main application
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("700x350")
    app = MainWindow(root)
    root.mainloop()


# In[ ]:

# Membuat imagedatagenerator
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

#Mencoba banyak batch sizes
batch_sizes = [16, 32, 64, 128]
for batch_size in batch_sizes:

    train_generator = training_datagen.flow_from_directory(
        training_dir,
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
    )

    validation_generator = validation_datagen.flow_from_directory(
        testing_dir,
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
    )

    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    # Index nama-nama kelas makanan
    class_indices = train_generator.class_indices
    print(class_indices)
    class_names = list(class_indices.keys())
    print(class_names)
    num_class = len(class_names)
    print(num_class)

    # Membuat class callback
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('val_accuracy') > 0.96):
                print("\nReached 96% accuracy so cancelling training!")
                self.model.stop_training = True

    # Membuat early stopping apabila vall_loss terlalu besar
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',   # Menggunakan validation loss untuk monitoring
        patience=5,           # Tunggu 5 epoch sebelum berhenti jika tidak ada peningkatan
        verbose=1,            
        restore_best_weights=True  # Mengembalikan bobot terbaik ketika berhenti
    )

    # Membuat model
    pretrained_model.trainable = False

    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dropout(0.3)(x)  # Dropout layer dengan tingkat dropout 50%
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)  # Dropout layer dengan tingkat dropout 50%

    outputs = tf.keras.layers.Dense(num_class, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    print(model.summary())

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn')
                ]
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=100,
        callbacks=[myCallback(),early_stopping]
    )

    # Plot grafik akurasi model
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss'] 
    precision = history.history['precision']
    val_precision = history.history['val_precision']
    recall = history.history['recall']
    val_recall = history.history['val_recall']
    f1_score = [2*(p*r)/(p+r+1e-10) for p, r in zip(precision, recall)]
    val_f1_score = [2*(p*r)/(p+r+1e-10) for p, r in zip(val_precision, val_recall)]

    epochs = range(len(acc))

    # Plot grafik akurasi
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot grafik loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot grafik presisi
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, precision, 'r', label='Training precision')
    plt.plot(epochs, val_precision, 'b', label='Validation precision')
    plt.title('Training and validation precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Plot grafik recall
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, recall, 'r', label='Training recall')
    plt.plot(epochs, val_recall, 'b', label='Validation recall')
    plt.title('Training and validation recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # Plot grafik F1-score
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, f1_score, 'r', label='Training F1-score')
    plt.plot(epochs, val_f1_score, 'b', label='Validation F1-score')
    plt.title('Training and validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.show()

    plt.tight_layout()
    plt.show()

    model.save_weights('foodModel_height1.h5')
    print('Model Saved!')
    model.save('food_model_all.h5')
    print('Model Saved!')
# %%
