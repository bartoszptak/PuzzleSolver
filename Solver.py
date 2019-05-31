import cv2
import Preprocessing

class Solver:
    pre = Preprocessing.Preprocessing()
    PuzzlePieces = []
    def __init__(self):
       pass
        
    def load(self, files): #wczytuje zdjęcia puzzli
        print('Files: {}'.format(len(files)))
        
        for file in files:
            img1 = cv2.imread(file)
            splited = self.pre.split_puzzles_from_image(img1)
            self.pre.get_slides(splited, self.PuzzlePieces)
        
        

    def solve(self):
        return self.PuzzlePieces
        