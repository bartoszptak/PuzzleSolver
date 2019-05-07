from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
import wx

COLOR_WHITE = (247,247,247)
COLOR_BLACK = (59,89,152)
COLOR_BLUE = (0, 0, 255)
COLOR_RED = (255, 0, 0)

SIZE_WIDTH = 650
SIZE_HEIGHT = 400


class ResultsWindow:
    def __init__(self, master, images):
        self.images = images
        self.master = master
        self.frame = Frame(self.master)
        # img = Image.open(self.images[0])
        # img.thumbnail((SIZE_WIDTH-10, SIZE_HEIGHT-10), Image.ANTIALIAS)
        # img = ImageTk.PhotoImage(img)
        # panel = Label(self.frame, image=img)
        # panel.pack(side="bottom", fill="both", expand="yes")
        img = ImageTk.PhotoImage(Image.open(images[0]))
        label = Label(self.frame, image=img).pack()
        self.frame.pack()

        # on exit
        # UI.root.deiconify()


class UI:

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

        load_images_button = Button(self.root, text='Select image(s)', command = self.load_images, width=50, height=5)
        load_images_button.pack()

        self.root.mainloop()

    def show_results_window(self, master, images):
        self.root.withdraw()
        self.newWindow = Toplevel(master)
        self.newWindow.winfo_toplevel().title('Puzzle Solver Results')
        self.newWindow.maxsize(SIZE_WIDTH, SIZE_HEIGHT)
        self.newWindow.minsize(SIZE_WIDTH, SIZE_HEIGHT)
        self.newWindow.resizable(0, 0)
        self.app = ResultsWindow(self.newWindow, images)

    def load_images(self):
        app = wx.App(None)
        style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE
        wildcard = "Image files (*.png,*.bmp;*.gif;*.jpg)|*.png;*.bmp;*.gif;*.jpg"
        dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPaths()
        else:
            path = None
        dialog.Destroy()

        messagebox.showinfo('Info', 'Loaded images!')
        self.show_results_window(self.root, path)


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
