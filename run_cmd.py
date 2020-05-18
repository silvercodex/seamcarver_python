from seamcarver import *
import sys

gradients = ["abs","norm","sobel1_norm","sobel1_abs","sobel3_norm","sobel3_abs"]
if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("not enough paramters")
        sys.exit()
    path = sys.argv[1]
    method = sys.argv[2]
    if method != "forward" and method != "backward":
        print("incorrect method, choose forward or backward")
        sys.exit()
    
    operation = sys.argv[3]
    if operation not in ["elongate","compress"]:
        print("incorrect operation, choose elongate or compress")
        sys.exit()
    try:
        target_width = int(sys.argv[4])
    except:
        print("invalid target width")
        sys.exit()
    gradient = sys.argv[5]
    if gradient not in gradients:
        print("incorrect gradient, choose one of " + gradients +  " defaulting to abs")
        gradient = gradients[0]
    dest = sys.argv[6]

    if operation == "elongate":
        image = cv2.imread(path)
        image_new = elongate_horizontal(image ,target_width, gradient, method == "forward")
    else:
        image = cv2.imread(path)
        image_new = compress_horizontal(image ,target_width, gradient, method == "forward")
    cv2.imwrite(dest,image_new)


    
    