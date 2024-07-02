import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar_y = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=canvas.xview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

class TutorApp:
    def __init__(self, root):
        self.root = root
        root.title("Tutorial Penggunaan")

        scrollable_frame = ScrollableFrame(root)
        scrollable_frame.pack(fill="both", expand=True)

        # Deskripsi
        description_label1 = tk.Label(scrollable_frame.scrollable_frame, text="Pedoman Penggunaan Kalkulator Kebutuhan Kalori", padx=10, pady=5, font=("Helvetica", 11, "bold"))
        description_label1.grid(row=0, column=0, columnspan=3, sticky='w')
        description_label2 = tk.Label(scrollable_frame.scrollable_frame, text="1. Tekan tombol 'Kalkulator Kebutuhan Kalori' pada window utama", padx=10, pady=5, font=("Helvetica", 11))
        description_label2.grid(row=1, column=0, columnspan=3, sticky='w')
        description_label3 = tk.Label(scrollable_frame.scrollable_frame, text="2. Isi form yang ada pada window 'Kalkulator Kebutuhan Kalori' sesuai dengan data diri anda", padx=10, pady=5, font=("Helvetica", 11))
        description_label3.grid(row=2, column=0, columnspan=3, sticky='w')
        description_label4 = tk.Label(scrollable_frame.scrollable_frame, text="3. Apabila sudah terisi semua formnya, maka anda dapat menekan tombol 'Hitung'", padx=10, pady=5, font=("Helvetica", 11))
        description_label4.grid(row=3, column=0, columnspan=3, sticky='w')
        description_label5 = tk.Label(scrollable_frame.scrollable_frame, text="4. Window informasi 'Hasil Perhitungan' yang berisikan nilai IMT anda dan jumlah kebutuhan kalori harian anda akan ditampilkan seperti pada gambar berikut :", padx=10, pady=5, font=("Helvetica", 11))
        description_label5.grid(row=4, column=0, columnspan=3, sticky='w')
        description_label6 = tk.Label(scrollable_frame.scrollable_frame, text="5. Setelah selesai, anda dapat menekan tombol silang pada window 'Hasil Perhitungan' serta window 'Kalkulator Kebutuhan Kalori' dan anda akan kembali menuju window utama", padx=10, pady=5, font=("Helvetica", 11))
        description_label6.grid(row=6, column=0, columnspan=3, sticky='w')

        description_label7 = tk.Label(scrollable_frame.scrollable_frame, text="Pedoman Penggunaan Pengidentifikasi Makanan dan Minuman", padx=10, pady=5, font=("Helvetica", 11, "bold"))
        description_label7.grid(row=7, column=0, columnspan=3, sticky='w')
        description_label8 = tk.Label(scrollable_frame.scrollable_frame, text="1. Tekan tombol 'Pengidentifikasi Makanan dan Minuman'", padx=10, pady=5, font=("Helvetica", 11))
        description_label8.grid(row=8, column=0, columnspan=3, sticky='w')
        description_label9 = tk.Label(scrollable_frame.scrollable_frame, text="2. Isi jumlah kebutuhan kalori harian anda, kemudian tekan tombol 'Hitung Kalori'", padx=10, pady=5, font=("Helvetica", 11))
        description_label9.grid(row=9, column=0, columnspan=3, sticky='w')
        description_label11 = tk.Label(scrollable_frame.scrollable_frame, text="3. Tekan tombol 'Pilih Gambar' dan cari gambar yang ingin anda input-kan kedalam sistem", padx=10, pady=5, font=("Helvetica", 11))
        description_label11.grid(row=10, column=0, columnspan=3, sticky='w')
        description_label12 = tk.Label(scrollable_frame.scrollable_frame, text="4. Jika gambar sudah sesuai, maka anda dapat menekan tombol 'Identifikasi Gambar'", padx=10, pady=5, font=("Helvetica", 11))
        description_label12.grid(row=11, column=0, columnspan=3, sticky='w')
        description_label13 = tk.Label(scrollable_frame.scrollable_frame, text="5. Gambar yang telah anda input-kan akan diidentifikasi dan ditampilkan jumlah kalorinya, seperti gambar berikut :", padx=10, pady=5, font=("Helvetica", 11))
        description_label13.grid(row=12, column=0, columnspan=3, sticky='w')
        description_label14 = tk.Label(scrollable_frame.scrollable_frame, text="6. Jumlah kalori yang anda isi pada langkah No 2 akan dikurangi dengan jumlah kalori makanan yang anda input-kan dan menampilkan hasilnya sebagai perhitungan sisa kalori untuk hari ini", padx=10, pady=5, font=("Helvetica", 11))
        description_label14.grid(row=14, column=0, columnspan=3, sticky='w')
        description_label15 = tk.Label(scrollable_frame.scrollable_frame, text="7. Apabila anda ingin menginputkan gambar makanan lain, silahkan tekan kembali tombol 'Pilih Gambar' dan mengulangi langkah no 4 hingga 7", padx=10, pady=5, font=("Helvetica", 11))
        description_label15.grid(row=15, column=0, columnspan=3, sticky='w')
        description_label16 = tk.Label(scrollable_frame.scrollable_frame, text="8. Apabila kebutuhan kalori anda hari ini sudah tercukupi atau jumlah kalori anda sama dengan nol (0) maka sistem\nakan menampilkan window informasi mengenai kebutuhan kalori anda yang telah tercukupi, seperti gambar berikut :", padx=10, pady=5, font=("Helvetica", 11))
        description_label16.grid(row=16, column=0, columnspan=3, sticky='w')
        description_label17 = tk.Label(scrollable_frame.scrollable_frame, text="9. Apabila gambar yang anda inputkan memiliki jumlah kalori yang lebih besar daripada sisa kalori kebutuhan harian anda\nmaka sistem akan menampilkan window peringatan mengenai kalori makanan yang berlebih, seperti gambar berikut :", padx=10, pady=5, font=("Helvetica", 11))
        description_label17.grid(row=18, column=0, columnspan=3, sticky='w')
        description_label18 = tk.Label(scrollable_frame.scrollable_frame, text="10. Setelah selesai menggunakan, anda dapat menekan tombol silang pada window peringatan ataupun window informasi\nkemudian menekan silang pada window 'Identifikasi Makanan dan Minuman' dan anda akan kembali menuju window utama", padx=10, pady=5, font=("Helvetica", 11))
        description_label18.grid(row=20, column=0, columnspan=3, sticky='w')

        # Add images
        self.add_image(scrollable_frame.scrollable_frame, "HP.png", 5, 0)
        self.add_image(scrollable_frame.scrollable_frame, "IG.png", 13, 0)
        self.add_image(scrollable_frame.scrollable_frame, "WP.png", 19, 0)
        self.add_image(scrollable_frame.scrollable_frame, "WF.png", 17, 0)

    def add_image(self, parent, image_path, row, column):
        try:
            img = Image.open(image_path)
            if image_path == "IG.png":
                img = img.resize((450, 400), Image.LANCZOS)  # Resize khusus untuk IG.png
            else:
                img = img.resize((350, 150), Image.LANCZOS)  # Resize default
            img = ImageTk.PhotoImage(img)
            img_label = tk.Label(parent, image=img)
            img_label.image = img
            img_label.grid(row=row, column=column, padx=10, pady=5)
        except Exception as e:
            print(f"Error loading image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x500")
    app = TutorApp(root)
    root.mainloop()
# %%
