import BunchLib as bl
import sys
import os

def main():
    if (len(sys.argv) < 2):
        print("Usage: main.py <image path>")
    else:
        imgPath = sys.argv[1]
        if (os.path.isdir(imgPath)):
            print("NYI")
        else:
            bl.colorizeGorskiiImgWirth(imgPath)

if __name__ == "__main__":
    main()