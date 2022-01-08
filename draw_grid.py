from PIL import Image, ImageDraw, ImageColor
import pprint

def remove_gray(image):
    white,black = map(ImageColor.getrgb, ["white", "black"])
    for x in range(image.width):
        for y in range(image.height):
            if image.getpixel((x,y)) not in (white,black):
                image.putpixel((x,y),white)

def draw_grid(path):
    image = Image.new("RGB", (500,500), (255,255,255))
    draw = ImageDraw.Draw(image)
    top_left = (9,9)
    for row in range(6):
        for col in range(6):
            rtl = (top_left[0]+80*col, top_left[1]+80*row)
            rbr = (rtl[0]+81, rtl[1]+81)
            draw.rectangle(xy=(rtl,rbr),fill=(255,255,255), outline=(0,0,0), width=2)
    image.save(path)
    return image

matrix = [[None for col in range(6)] for row in range(6)]
top_left = (11,11)
for row in range(6):
    for col in range(6):
        rtl = (top_left[0]+80*col, top_left[1]+80*row)
        rbr = (rtl[0]+77, rtl[1]+77)
        matrix[row][col] = (rtl,rbr)
pprint.pprint(matrix)
