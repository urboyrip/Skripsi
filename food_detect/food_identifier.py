
#In[]:

import numpy as np
import pandas as pd
import os.path
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import splitfolders
from PIL import Image, ImageTk
import tkinter as tk
import pickle
from tkinter import filedialog, messagebox
from food_list import ExcelViewerApp
from datetime import datetime


# In[ ]:

# Masukkan Path dataset
dataset_dir = 'D:/unair/Bangkit/konversi datmin/dataset gambar/Dataset'

# Split direktori dataset menjadi dua folder train dan val 
splitfolders.ratio(dataset_dir, output="food-data", seed=1337, ratio=(.8, .2), group_prefix=None)

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
    batch_size=50,
)

validation_generator = validation_datagen.flow_from_directory(
    testing_dir,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=50,
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

#In[]:

class CalorieApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Identifikasi Makanan dan Minuman")

        # Membuka food_list
        self.open_excel_viewer()

        self.label_title = tk.Label(root, text="Pengidentifikasi \n Makanan dan Minuman", font=("Times", 20, "bold"))
        self.label_title.grid(row=0, column=1, columnspan=1, pady=10)

        # Membuat label dan entry untuk memasukkan jumlah kalori
        self.label_kalori_input = tk.Label(root, text="Masukkan Jumlah Kalori Harian :", font=("Helvetica", 12), anchor='w')
        self.label_kalori_input.grid(row=1, column=0, sticky='w', padx=(25, 0))

        self.entry_kalori = tk.Entry(root, font=("Helvetica", 12))
        self.entry_kalori.grid(row=1, column=1)

        self.calculate_button = tk.Button(root, text="Simpan Kalori", command=self.calculate_calories, font=("Times", 13), pady=5, padx=10)
        self.calculate_button.grid(row=1, column=2, pady=10, padx=(0, 25))

        self.label_kalori_display = tk.Label(root, text="Kalori harian anda adalah : ", font=("Helvetica", 12), anchor='w')
        self.label_kalori_display.grid(row=2, column=0, sticky='w', padx=(25, 0))

        self.label_kalori = tk.Label(root, font=("Helvetica", 12))
        self.label_kalori.grid(row=2, column=1)

        # Membuat label dan tombol untuk memilih gambar
        self.browse_button = tk.Button(root, text="Pilih Gambar", command=self.browse_image, font=("Times", 13), pady=5, padx=10)
        self.browse_button.grid(row=3, column=2, pady=10, padx=(0, 25))

        self.label_image_path = tk.Label(root, text="Path Gambar :", font=("Helvetica", 12), anchor='w')
        self.label_image_path.grid(row=3, column=0, sticky='w', padx=(25, 0))

        # Keterangan path gambar
        self.label_image = tk.Label(root, font=("Helvetica", 12))
        self.label_image.grid(row=3, column=1)

        # Label untuk menampilkan gambar
        self.image_display_label = tk.Label(root)
        self.image_display_label.grid(row=4, column=1, pady=10)

        # Label dan entry untuk memasukkan jumlah porsi
        self.label_porsi = tk.Label(root, text="Masukkan Jumlah Takaran :", font=("Helvetica", 12), anchor='w')
        self.label_porsi.grid(row=5, column=0, sticky='w', padx=(25, 0))

        self.entry_porsi = tk.Entry(root, font=("Helvetica", 12))
        self.entry_porsi.grid(row=5, column=1)

        # Membuat tombol untuk melakukan identifikasi gambar
        self.identify_button = tk.Button(root, text="Identifikasi Kalori Makanan/Minuman", command=self.identify_image, font=("Times", 13), pady=5, padx=10)
        self.identify_button.grid(row=6, column=1, pady=10)

        # Label untuk hasil identifikasi dan perhitungan kalori
        self.result_label = tk.Label(root, font=("Helvetica", 12))
        self.result_label.grid(row=7, column=1)

        self.result_label2 = tk.Label(root, font=("Helvetica", 12))
        self.result_label2.grid(row=8, column=1)

        self.result_label3 = tk.Label(root, font=("Helvetica", 12))
        self.result_label3.grid(row=9, column=1, pady=(0, 10))

        self.obj_tes = None
        self.kalori_user = None
        self.jumlah_porsi = None  # Jumlah porsi default

        # Load kalori sebelumnya
        self.load_state()

        # Load model hanya sekali
        self.model = tf.keras.models.load_model('D:/unair/Bangkit/konversi datmin/food_detect/food_model_all.h5')

        # Close window
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # Membuka food_list
    def open_excel_viewer(self):
        new_window = tk.Toplevel(self.root)
        new_window.geometry("600x600")
        self.food_list = ExcelViewerApp(new_window, 'D:/unair/Bangkit/konversi datmin/dataset_kalori_makanan.xlsx')

    def browse_image(self):
        self.obj_tes = filedialog.askopenfilename(initialdir="/", title="Pilih Gambar", filetypes=[("Image Files", "*.jpg; *.jpeg; *.png")])
        self.label_image.config(text=self.obj_tes)
        self.display_image()

    def display_image(self):
        if self.obj_tes:
            img_display = Image.open(self.obj_tes)
            img_display = img_display.resize((300, 300), Image.LANCZOS)
            img_display = ImageTk.PhotoImage(img_display)
            self.image_display_label.config(image=img_display)
            self.image_display_label.image = img_display

    def calculate_calories(self):
        try:
            self.kalori_user = float(self.entry_kalori.get())
            self.label_kalori.config(text=f"{self.kalori_user} kkal")
            self.save_to_excel(self.kalori_user, 'Tambah')  # Panggil fungsi pencatatan untuk penambahan kalori
        except ValueError:
            messagebox.showerror("Error", "Masukkan jumlah kalori yang valid.")

    def identify_image(self):
        if self.kalori_user is None:
            messagebox.showerror("Error", "Silakan masukkan jumlah kalori harian anda terlebih dahulu.")
            return

        if self.obj_tes is None:
            messagebox.showerror("Error", "Silakan pilih gambar terlebih dahulu.")
            return

        try:
            df = pd.read_excel('D:/unair/Bangkit/konversi datmin/dataset_kalori_makanan.xlsx')
        except FileNotFoundError:
            messagebox.showerror("Error", "File dataset tidak ditemukan.")
            return

        img = tf.keras.utils.load_img(self.obj_tes, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0
        img_array = tf.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class]

        nama = 'Nama'
        kalori = 'Kalori_100gr'
        konstanta = 'Konstanta'
        nilai_kolom2 = 0

        # Meminta pengguna untuk memasukkan jumlah porsi
        try:
            self.jumlah_porsi = int(self.entry_porsi.get())
        except ValueError:
            messagebox.showerror("Error", "Masukkan jumlah takaran yang valid.")
            return

        for index, row in df.iterrows():
            if row[nama] == predicted_class_name:
                nilai_kolom2 = row[kalori] * row[konstanta] * self.jumlah_porsi
                break

        self.result_label.config(text=f"Ini adalah gambar {predicted_class_name} dengan confidence sebesar {100 * np.max(score):.2f}")

        if nilai_kolom2 > 0:
            self.result_label2.config(text=f"Dalam {self.jumlah_porsi} porsi {predicted_class_name} terdapat {nilai_kolom2:.2f} kkal")
        else:
            self.result_label2.config(text="Tidak ada nilai yang sesuai.")

        if self.kalori_user >= nilai_kolom2:
            self.kalori_user -= nilai_kolom2
            self.result_label3.config(text=f"Sisa kalori hari ini : {self.kalori_user:.2f} kkal")
            self.save_to_excel(self.kalori_user, 'Kurang', predicted_class_name, nilai_kolom2)  # Panggil fungsi pencatatan untuk pengurangan kalori
        else:
            messagebox.showwarning("Peringatan", "Kalori yang dimasukkan lebih besar dari kebutuhan kalori harian. Harap masukkan gambar makanan dan minuman yang lain.")

        if self.kalori_user <= 0:
            messagebox.showinfo("Informasi", "Kebutuhan kalori harian anda sudah tercukupi. Pastikan untuk memenuhi kebutuhan gizi lainnya.")

        # Simpan state kalori
        self.save_state()

    def save_to_excel(self, kalori_user, operasi, makanan=None, kalori_makanan=None):
        data = {
            'Timestamp': [datetime.now()],
            'Kalori User': [kalori_user],
            'Operasi': [operasi],
            'Makanan': [makanan],
            'Kalori Makanan': [kalori_makanan]
        }
        df = pd.DataFrame(data)

        try:
            existing_df = pd.read_excel('D:/unair/Bangkit/konversi datmin/kalori_data.xlsx')
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            # Jika file tidak ada, buat file baru
            pass

        df.to_excel('D:/unair/Bangkit/konversi datmin/kalori_data.xlsx', index=False)

    def save_state(self):
        state = {'kalori_user': self.kalori_user}
        with open('state.pkl', 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        try:
            with open('state.pkl', 'rb') as f:
                state = pickle.load(f)
                self.kalori_user = state.get('kalori_user')
                if self.kalori_user is not None:
                    self.label_kalori.config(text=f"{self.kalori_user} kkal")
        except FileNotFoundError:
            pass

    def on_closing(self):
        self.save_state()
        self.root.destroy()

# Menjalankan aplikasi
if __name__ == "__main__":
    root = tk.Tk()
    app = CalorieApp(root)
    root.mainloop()


# %%
