from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
import wx
import time

COLOR_WHITE = (247, 247, 247)
COLOR_BLACK = (59, 89, 152)
COLOR_BLUE = (0, 0, 255)
COLOR_RED = (255, 0, 0)

SIZE_WIDTH = 650
SIZE_HEIGHT = 400


class ResultsWindow:
    counter = 0
    solved = False

    def __init__(self, master, images):
        self.images = ['img/puchatek_r_0.png', 'img/puchatek_r_1.png']
        self.master = master
        self.frame = Frame(self.master)
        next_button = Button(self.frame, text='Next image', command=self.next_image)
        previous_button = Button(self.frame, text='Previous image', command=self.previous_image)
        solve_button = Button(self.frame, text='Solve', command=self.solve)

        img = Image.open(self.images[self.counter])
        img.thumbnail((SIZE_WIDTH - 10, SIZE_HEIGHT - 40), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.panel = Label(self.frame, image=img)
        self.panel.grid(row=0, columnspan=3, sticky=SW)

        next_button.grid(row=1, column=2)
        previous_button.grid(row=1, column=0)
        solve_button.grid(row=1, column=1)
        self.frame.pack()
        self.frame.mainloop()

    def set_image(self):
        img = Image.open(self.images[self.counter])
        img.thumbnail((SIZE_WIDTH - 10, SIZE_HEIGHT - 40), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

    def next_image(self):
        self.counter += 1
        if self.counter > len(self.images) - 1:
            self.counter = 0
        self.set_image()

    def previous_image(self):
        self.counter -= 1
        if self.counter < 0:
            self.counter = len(self.images) - 1
        self.set_image()

    def solve(self):
        if self.solved:
            return
        time.sleep(2)
        messagebox.showinfo('Success', 'PUZZLE SOLVED!')
        self.images.append('img/puchatek_r.png')
        self.counter = len(self.images) - 1
        self.set_image()
        self.solved = True


class UI:
    def test(self):
        self.newWindow.grab_release()
        self.newWindow.destroy()

    root = Tk()

    def __init__(self):

        self.root.maxsize(SIZE_WIDTH, SIZE_HEIGHT)
        self.root.minsize(SIZE_WIDTH, SIZE_HEIGHT)
        self.root.resizable(0, 0)
        self.root.winfo_toplevel().title('Puzzle Solver')
        FONT = 'Microsoft Sans Serif'
        font_header = (FONT, 45)
        font_footer = (FONT, 10)
        header = Label(self.root, text='Puzzle Solver', font=font_header)
        footer = Label(self.root, text='by Przybylowski Paweł, Ptak Bartosz, Walkowiak Mikolaj', font=font_footer)
        header.pack()
        footer.place(x=10, y=375)

        load_images_button = Button(self.root, text='Select image(s)', command=self.load_images, width=50, height=5)
        load_images_button.place(x=145, y=150)

        self.root.mainloop()

    def show_results_window(self, master, images):
        self.newWindow = Toplevel(master)
        self.newWindow.winfo_toplevel().title('Puzzle Solver - Puzzle Pieces Preview')
        self.newWindow.maxsize(SIZE_WIDTH, SIZE_HEIGHT)
        self.newWindow.minsize(SIZE_WIDTH, SIZE_HEIGHT)
        self.newWindow.resizable(0, 0)
        self.newWindow.grab_set()
        self.newWindow.protocol("WM_DELETE_WINDOW", self.test)
        self.app = ResultsWindow(self.newWindow, images)

    def load_images(self):
        app = wx.App(None)
        style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE
        wildcard = "Image files (*.png,*.bmp;*.gif;*.jpg)|*.png;*.bmp;*.gif;*.jpg"
        dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPaths()
            messagebox.showinfo('Info', 'Loaded images!')
            self.show_results_window(self.root, path)
        else:
            path = None
            messagebox.showinfo('Error', 'Load images first!')
        dialog.Destroy()


if __name__ == "__main__":
    UI()
# class ResultsDisplay:
#     def __init__(self, images):
#         for image in images:
#             results_display = Tk()
#             results_display.maxsize(SIZE_WIDTH, SIZE_HEIGHT)
#             results_display.minsize(SIZE_WIDTH, SIZE_HEIGHT)
#             results_display.resizable(0, 0)
#             results_display.winfo_toplevel().title('Puzzle Solver Results')
#
#             img = Image.open(image)
#             img.thumbnail((SIZE_WIDTH-10, SIZE_HEIGHT-10), Image.ANTIALIAS)
#             img = ImageTk.PhotoImage(img)
#             panel = Label(results_display, image=img)
#             panel.pack(side="bottom", fill="both", expand="yes")
#             results_display.mainloop()
