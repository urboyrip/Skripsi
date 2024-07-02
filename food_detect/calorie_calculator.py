# In[]

import tkinter as tk
from tkinter import messagebox

class CalorieCalculatorApp:
    def __init__(self, root):
        self.root = root
        root.title("Kalkulator Kebutuhan Kalori Harian Penderita Diabetes")

        self.label_description = tk.Label(root, text="Kalkulator Kebutuhan Kalori\nHarian Penderita Diabetes", font = ("Times", 20, "bold"))
        self.label_description.grid(row=0, columnspan=2, pady=10)

        self.label_description = tk.Label(root, text="Masukkan data berikut untuk \n menghitung kebutuhan kalori harian Anda:", font = ("Times", 14))
        self.label_description.grid(row=1, columnspan=2, pady=10)

        self.label_bb = tk.Label(root, text="Masukkan Berat Badan (kg):", font = ("Helvetica", 12), anchor='w')
        self.label_bb.grid(row=2, column=0,pady=5, sticky = 'w', padx=(15,0))
        self.entry_bb = tk.Entry(root, font = ("Helvetica", 12))
        self.entry_bb.grid(row=2, column=1,pady=5)

        self.label_tb = tk.Label(root, text="Masukkan Tinggi Badan (cm):", font = ("Helvetica", 12), anchor='w')
        self.label_tb.grid(row=3, column=0,pady=5, sticky = 'w', padx=(15,0))
        self.entry_tb = tk.Entry(root, font = ("Helvetica", 12))
        self.entry_tb.grid(row=3, column=1,pady=5)

        self.label_jk = tk.Label(root, text="Masukkan Jenis Kelamin \n(Pria = 0, Wanita = 1):", font = ("Helvetica", 12), anchor='w')
        self.label_jk.grid(row=4, column=0,pady=5, sticky = 'w', padx=(15,0))
        self.entry_jk = tk.Entry(root, font = ("Helvetica", 12))
        self.entry_jk.grid(row=4, column=1,pady=5)

        self.label_umur = tk.Label(root, text="Masukkan Umur:", font = ("Helvetica", 12), anchor='w')
        self.label_umur.grid(row=5, column=0,pady=5, sticky = 'w', padx=(15,0))
        self.entry_umur = tk.Entry(root, font = ("Helvetica", 12))
        self.entry_umur.grid(row=5, column=1,pady=5)

        self.label_kegiatan = tk.Label(root, text="Masukkan Kegiatan\n(Tidak Bekerja = 0, \nRingan = 1, \nSedang = 2, \nBerat = 3, \nSangat Berat = 4):", font = ("Helvetica", 12), anchor='w')
        self.label_kegiatan.grid(row=6, column=0,pady=5, sticky = 'w', padx=(15,0))
        self.entry_kegiatan = tk.Entry(root, font = ("Helvetica", 12))
        self.entry_kegiatan.grid(row=6, column=1,pady=5)

        self.label_description = tk.Label(root, text=" Tidak Bekerja : Pensiunan \n Ringan : Pegawai Kantor, Guru, IRT \n Sedang : Mahasiswa, Pegawai Industri Ringan, Militer (Kondisi Tidak Perang) \n Berat : Petani, Buruh, Atlet, Militer (Kondisi Perang) \n Sangat Berat : Tukang Gali, Tukang Becak, Kuli", font = ("Helvetica", 11), anchor = 'w', justify = 'left')
        self.label_description.grid(row=6, column = 2, pady=5, sticky = 'w', padx=(15,0))

        self.button = tk.Button(root, text="Hitung", font = ("Times", 13), command=self.calculate_calories, pady=5, padx=10)
        self.button.grid(row=7, columnspan=2, pady=10)

    def calculate_calories(self):
        bb = float(self.entry_bb.get())
        tb = float(self.entry_tb.get())
        jk = float(self.entry_jk.get())
        umur = float(self.entry_umur.get())
        kegiatan = float(self.entry_kegiatan.get())

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
    
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("950x500")
    app = CalorieCalculatorApp(root)
    root.mainloop()
# %%
