import pygame
import wx

COLOR_WHITE = (247,247,247)
COLOR_BLACK = (59,89,152)
COLOR_BLUE = (0, 0, 255)
COLOR_RED = (255, 0, 0)

SIZE_WIDTH = 650
SIZE_HEIGHT = 400

x = pygame.init()
pygame.font.init()

FONT = 'Microsoft Sans Serif'
font_header = pygame.font.SysFont(FONT, 45)
font_footer = pygame.font.SysFont(FONT, 16)
font_button = pygame.font.SysFont(FONT, 20)

game_display = pygame.display.set_mode((SIZE_WIDTH,SIZE_HEIGHT))
pygame.display.set_caption('Puzzle Solver')
game_display.fill(COLOR_WHITE)

game_exit = False


def get_path():
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE 
    wildcard = "Image files (*.png,*.bmp;*.gif;*.jpg)|*.png;*.bmp;*.gif;*.jpg"
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPaths()
    else:
        path = None
    dialog.Destroy()
    return path


def image_button():
    x = int(SIZE_WIDTH/2)-100
    y = int(SIZE_HEIGHT/2)-30
    width = 200
    height = 60

    pygame.draw.rect(game_display, COLOR_BLACK, pygame.Rect(x, y, width, height))
    game_display.blit(font_button.render('Select image(s)', False, COLOR_WHITE), (int(SIZE_WIDTH/2)-100+34,int(SIZE_HEIGHT/2)-30+18))

while not game_exit:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_exit = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            button_click = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            x = int(SIZE_WIDTH/2)-100
            y = int(SIZE_HEIGHT/2)-30
            width = 200
            height = 60
            if x + width > mouse_pos[0] > x and y + height > mouse_pos[1] > y:
                if button_click[0] == 1:
                    paths = get_path()
                    if paths is not None:
                        print(paths)
                        ## TO DO: wszystko tu się musi wykonać




    game_display.fill(COLOR_WHITE)
    game_display.blit(font_header.render('Puzzle Solver', False, COLOR_BLACK), (int(SIZE_WIDTH/2)-130,20))

    image_button()
    
    game_display.blit(font_footer.render('by Przybyłowski Paweł, Ptak Bartosz, Walkowiak Mikołaj', False, COLOR_BLACK), (10,SIZE_HEIGHT-25))
    pygame.display.update()

pygame.quit()
quit()