import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class SegmentationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Metode Segmentasi Citra")
        self.master.geometry("1100x650")
        self.master.configure(bg="#1e1e2f")

        # Variabel untuk menyimpan citra
        self.original_bgr = None
        self.segmented_bgr = None

        # ----- FRAME ATAS: judul + tombol open -----
        top_frame = tk.Frame(master, bg="#1e1e2f")
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)

        title_label = tk.Label(
            top_frame,
            text="Demo Metode Segmentasi Citra",
            fg="white",
            bg="#1e1e2f",
            font=("Segoe UI", 20, "bold")
        )
        title_label.pack(side=tk.LEFT)

        open_btn = tk.Button(
            top_frame,
            text="Buka Gambar...",
            command=self.open_image,
            font=("Segoe UI", 11, "bold"),
            bg="#4e9af1",
            fg="white",
            activebackground="#3478c0",
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        open_btn.pack(side=tk.RIGHT)

        # ----- FRAME TENGAH: display gambar -----
        mid_frame = tk.Frame(master, bg="#1e1e2f")
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        # Frame untuk gambar asli
        left_frame = tk.Frame(mid_frame, bg="#25253a", bd=1, relief=tk.RIDGE)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)

        left_title = tk.Label(
            left_frame,
            text="Citra Asli",
            bg="#25253a",
            fg="white",
            font=("Segoe UI", 12, "bold")
        )
        left_title.pack(pady=5)

        self.original_label = tk.Label(left_frame, bg="#25253a")
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame untuk hasil segmentasi
        right_frame = tk.Frame(mid_frame, bg="#25253a", bd=1, relief=tk.RIDGE)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)

        right_title = tk.Label(
            right_frame,
            text="Hasil Segmentasi",
            bg="#25253a",
            fg="white",
            font=("Segoe UI", 12, "bold")
        )
        right_title.pack(pady=5)

        self.segmented_label = tk.Label(right_frame, bg="#25253a")
        self.segmented_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Simpan referensi PhotoImage agar tidak di-GC
        self._orig_photo = None
        self._seg_photo = None

        # ----- FRAME BAWAH: tombol metode + kontrol -----
        bottom_frame = tk.Frame(master, bg="#1e1e2f")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=10)

        # Tombol metode
        btn_style = dict(
            font=("Segoe UI", 10, "bold"),
            bg="#34344a",
            fg="white",
            activebackground="#56567a",
            relief=tk.FLAT,
            padx=10,
            pady=5
        )

        edge_btn = tk.Button(
            bottom_frame, text="Edge Detection (Canny)",
            command=self.apply_edge, **btn_style
        )
        edge_btn.pack(side=tk.LEFT, padx=5)

        thr_btn = tk.Button(
            bottom_frame, text="Thresholding (Otsu)",
            command=self.apply_threshold, **btn_style
        )
        thr_btn.pack(side=tk.LEFT, padx=5)

        kmeans_btn = tk.Button(
            bottom_frame, text="Clustering (k-means)",
            command=self.apply_kmeans, **btn_style
        )
        kmeans_btn.pack(side=tk.LEFT, padx=5)

        reset_btn = tk.Button(
            bottom_frame, text="Reset",
            command=self.reset_segmented, **btn_style
        )
        reset_btn.pack(side=tk.LEFT, padx=5)

        # Slider untuk jumlah cluster k
        slider_frame = tk.Frame(bottom_frame, bg="#1e1e2f")
        slider_frame.pack(side=tk.RIGHT)

        self.k_value = tk.IntVar(value=3)
        k_label = tk.Label(
            slider_frame, text="k (cluster):",
            bg="#1e1e2f", fg="white", font=("Segoe UI", 9)
        )
        k_label.pack(side=tk.LEFT)

        k_slider = ttk.Scale(
            slider_frame, from_=2, to=8,
            orient=tk.HORIZONTAL, variable=self.k_value,
            length=150
        )
        k_slider.pack(side=tk.LEFT, padx=5)

    # ==================== FUNGSI UTILITAS ====================

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Gagal membaca file gambar.")
            return

        self.original_bgr = img
        self.segmented_bgr = None

        self.show_image(self.original_bgr, is_original=True)
        self.segmented_label.config(image="", text="")

    def show_image(self, bgr_img, is_original=False):
        """Tampilkan citra BGR ke label (original / segmented)."""
        if bgr_img is None:
            return

        # Konversi BGR -> RGB
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # Resize agar muat di GUI (maks 500x500)
        max_size = 500
        h, w = rgb_img.shape[:2]
        scale = min(max_size / h, max_size / w, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(rgb_img)
        photo = ImageTk.PhotoImage(pil_img)

        if is_original:
            self._orig_photo = photo
            self.original_label.config(image=self._orig_photo)
        else:
            self._seg_photo = photo
            self.segmented_label.config(image=self._seg_photo)

    def check_image_loaded(self):
        if self.original_bgr is None:
            messagebox.showwarning("Perhatian", "Silakan buka gambar terlebih dahulu.")
            return False
        return True

    # ==================== METODE SEGMENTASI ====================

    def apply_edge(self):
        """Metode Discontinuity: Edge Detection (Canny)."""
        if not self.check_image_loaded():
            return

        gray = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Ubah ke 3 channel untuk ditampilkan sebagai BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        self.segmented_bgr = edges_bgr
        self.show_image(self.segmented_bgr, is_original=False)

    def apply_threshold(self):
        """Metode Similarity: Thresholding (Otsu)."""
        if not self.check_image_loaded():
            return

        gray = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)

        # Otsu otomatis mencari nilai threshold terbaik
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.segmented_bgr = thresh_bgr
        self.show_image(self.segmented_bgr, is_original=False)

    def apply_kmeans(self):
        """Metode Similarity: Clustering k-means."""
        if not self.check_image_loaded():
            return

        k = max(2, int(self.k_value.get()))

        img = self.original_bgr
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)

        # K-means criteria & eksekusi
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented = segmented.reshape((img.shape))

        self.segmented_bgr = segmented
        self.show_image(self.segmented_bgr, is_original=False)

    def reset_segmented(self):
        if self.original_bgr is None:
            return
        self.segmented_bgr = None
        self.segmented_label.config(image="", text="")
        messagebox.showinfo("Reset", "Hasil segmentasi di-reset. Silakan pilih metode lagi.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
