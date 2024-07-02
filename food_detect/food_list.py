import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

class ExcelViewerApp:
    def __init__(self, root, file_path):
        self.root = root
        self.root.title("Daftar Makanan")

        self.style = ttk.Style()
        self.style.configure("Treeview.Heading", font=("Helvetica", 14))
        self.style.configure("Treeview", font=("Helvetica", 12), rowheight=25)

        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.title_label = tk.Label(self.frame, text="Data Gramasi Makanan", font=("Helvetica", 16, "bold"))
        self.title_label.pack(pady=10)

        self.tree_frame = tk.Frame(self.frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree_scroll_y = tk.Scrollbar(self.tree_frame, orient=tk.VERTICAL)
        self.tree_scroll_x = tk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL)
        
        self.tree = ttk.Treeview(self.tree_frame, columns=("Nama", "Satu Porsi/Centong", "Gramasi"), show="headings",
                                 yscrollcommand=self.tree_scroll_y.set, xscrollcommand=self.tree_scroll_x.set)
        
        self.tree_scroll_y.config(command=self.tree.yview)
        self.tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree_scroll_x.config(command=self.tree.xview)
        self.tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree.heading("Nama", text="Nama", anchor=tk.CENTER)
        self.tree.heading("Satu Porsi/Centong", text="Takaran", anchor=tk.CENTER)
        self.tree.heading("Gramasi", text="Gramasi", anchor=tk.CENTER)
        
        self.tree.column("Nama", anchor=tk.CENTER)
        self.tree.column("Satu Porsi/Centong", anchor=tk.CENTER)
        self.tree.column("Gramasi", anchor=tk.CENTER)

        self.tree.pack(fill=tk.BOTH, expand=True)

        self.load_excel(file_path)

    def load_excel(self, file_path):
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read Excel file:\n{str(e)}")
            return

        if not {"Nama", "Satu Porsi/Centong", "Gramasi"}.issubset(df.columns):
            messagebox.showerror("Error", "Excel file must contain 'Nama', 'Satu Porsi/Centong', and 'Gramasi' columns")
            return

        for row in self.tree.get_children():
            self.tree.delete(row)

        for _, row in df.iterrows():
            self.tree.insert("", tk.END, values=(row["Nama"], row["Satu Porsi/Centong"], row["Gramasi"]))

if __name__ == "__main__":
    root = tk.Tk()
    app = ExcelViewerApp(root,'D:/unair/Bangkit/konversi datmin/dataset_kalori_makanan.xlsx')
    root.mainloop()
