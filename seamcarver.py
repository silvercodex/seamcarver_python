import numpy as np
import cv2
import cython





def vertical_seam_forward_energy_map(image, energy_func = "abs", P = True):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float64)

    if energy_func == "abs":
        y = np.concatenate([np.zeros((1,image.shape[1])),np.diff(image, axis = 0)], axis = 0)
        x = np.concatenate([np.zeros((image.shape[0],1)),np.diff(image, axis = 1)], axis = 1)
        energy =  np.abs(x) + np.abs(y)
    elif energy_func == "norm": 
        y = np.concatenate([np.zeros((1,image.shape[1])),np.diff(image, axis = 0)], axis = 0)
        x = np.concatenate([np.zeros((image.shape[0],1)),np.diff(image, axis = 1)], axis = 1)
        energy =  np.sqrt(x**2 + y**2)
    elif energy_func == "sobel1_abs":
        energy = np.abs(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=1)) + np.abs(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=1))
    elif energy_func == "sobel3_abs":
        energy = np.abs(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)) + np.abs(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3))
    elif energy_func == "sobel1_norm":
        energy = np.sqrt(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=1)**2 + cv2.Sobel(image,cv2.CV_64F,0,1,ksize=1)**2) 
    elif energy_func == "sobel3_norm":
        energy = np.sqrt(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)**2 + cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)**2) 
    else:
        return None

    C_u = np.zeros(image.shape)
    C_l = np.zeros(image.shape)
    C_r = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if j>0 and (j < image.shape[1]-1):
                C_u[i,j] = np.abs(image[i,j+1] - image[i,j-1])
                if i >0:
                    C_l[i,j] = C_u[i,j] + np.abs(image[i-1,j] - image[i,j-1]) 
                    C_r[i,j] = C_u[i,j] + np.abs(image[i-1,j] - image[i,j+1]) 
            else:
                C_u[i,j] = C_l[i,j] = C_r[i,j] = energy[i,j]

    M = np.zeros(energy.shape)

    energy = energy*P
    for i in range(1,image.shape[0]):
        for j in range(image.shape[1]):
            if j+1 >= image.shape[1]:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j-1]+ C_l[i,j], M[i-1,j]+C_u[i,j]])
            elif j - 1 <=0:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j]+C_u[i,j], M[i-1,j+1]+C_r[i,j]])
            else:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j-1] + C_l[i,j], M[i-1,j]+C_u[i,j], M[i-1,j+1]+C_r[i,j]])

    M = M/M.max()*255
    return M.astype(np.uint8)



def vertical_seam_energy_map(image, energy_func = "abs"):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float64)

    if energy_func == "abs":
        y = np.concatenate([np.zeros((1,image.shape[1])),np.diff(image, axis = 0)], axis = 0)
        x = np.concatenate([np.zeros((image.shape[0],1)),np.diff(image, axis = 1)], axis = 1)
        energy =  np.abs(x) + np.abs(y)
    elif energy_func == "norm": 
        y = np.concatenate([np.zeros((1,image.shape[1])),np.diff(image, axis = 0)], axis = 0)
        x = np.concatenate([np.zeros((image.shape[0],1)),np.diff(image, axis = 1)], axis = 1)
        energy =  np.sqrt(x**2 + y**2)
    elif energy_func == "sobel1_abs":
        energy = np.abs(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=1)) + np.abs(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=1))
    elif energy_func == "sobel3_abs":
        energy = np.abs(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)) + np.abs(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3))
    elif energy_func == "sobel1_norm":
        energy = np.sqrt(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=1)**2 + cv2.Sobel(image,cv2.CV_64F,0,1,ksize=1)**2) 
    elif energy_func == "sobel3_norm":
        energy = np.sqrt(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)**2 + cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)**2) 
    else:
        return None

    M = energy.copy()
    for i in range(1,image.shape[0]):
        for j in range(image.shape[1]):
            if j+1 >= image.shape[1]:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j-1],M[i-1,j]])
            elif j - 1 <=0:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j],M[i-1,j+1]])
            else:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j-1],M[i-1,j],M[i-1,j+1]])
    M = M/M.max()*255
    return M.astype(np.uint8)


def vertical_seam_forward(image, energy_func = "abs", P = True):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float64)

    if energy_func == "abs":
        y = np.concatenate([np.zeros((1,image.shape[1])),np.diff(image, axis = 0)], axis = 0)
        x = np.concatenate([np.zeros((image.shape[0],1)),np.diff(image, axis = 1)], axis = 1)
        energy =  np.abs(x) + np.abs(y)
    elif energy_func == "norm": 
        y = np.concatenate([np.zeros((1,image.shape[1])),np.diff(image, axis = 0)], axis = 0)
        x = np.concatenate([np.zeros((image.shape[0],1)),np.diff(image, axis = 1)], axis = 1)
        energy =  np.sqrt(x**2 + y**2)
    elif energy_func == "sobel1_abs":
        energy = np.abs(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=1)) + np.abs(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=1))
    elif energy_func == "sobel3_abs":
        energy = np.abs(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)) + np.abs(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3))
    elif energy_func == "sobel1_norm":
        energy = np.sqrt(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=1)**2 + cv2.Sobel(image,cv2.CV_64F,0,1,ksize=1)**2) 
    elif energy_func == "sobel3_norm":
        energy = np.sqrt(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)**2 + cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)**2) 
    else:
        return None

    C_u = np.zeros(image.shape)
    C_l = np.zeros(image.shape)
    C_r = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if j>0 and (j < image.shape[1]-1):
                C_u[i,j] = np.abs(image[i,j+1] - image[i,j-1])
                if i >0:
                    C_l[i,j] = C_u[i,j] + np.abs(image[i-1,j] - image[i,j-1]) 
                    C_r[i,j] = C_u[i,j] + np.abs(image[i-1,j] - image[i,j+1]) 
            else:
                C_u[i,j] = C_l[i,j] = C_r[i,j] = energy[i,j]

    M = np.zeros(energy.shape)

    energy = energy*P
    for i in range(1,image.shape[0]):
        for j in range(image.shape[1]):
            if j+1 >= image.shape[1]:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j-1]+ C_l[i,j], M[i-1,j]+C_u[i,j]])
            elif j - 1 <=0:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j]+C_u[i,j], M[i-1,j+1]+C_r[i,j]])
            else:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j-1] + C_l[i,j], M[i-1,j]+C_u[i,j], M[i-1,j+1]+C_r[i,j]])

    MIN = M[-1].min()
    end = []
    for j in range(image.shape[1]):
        if MIN == M[-1,j]:
            end.append([j])



    if len(end) != 1:
        broke = False
        for i in reversed(range(image.shape[0]-1)):          
            for index,path in enumerate(end):
                MIN = 100000
                column = path[-1]
                current = column
                if column+1 < image.shape[1]:
                    if M[i,column+1] < MIN:
                        current = column+1
                        MIN = M[i,column+1]
                if column - 1 > 0:
                    if M[i,column-1] < MIN:
                        current = column-1
                        MIN = M[i,column-1]
                if M[i,column] < MIN:
                    current = column
                    MIN = M[i,column]  
                end[index].append(current)
            e = np.array(end)
            MINIMUM = M[i,e[:,-1]].min()

            drop = []
            for path in end:
                if M[i,path[-1]] != MINIMUM:
                    drop.append(path)
            for d in drop:
                end.remove(d)

            if len(end) == 1:
                end = end[0][0]
                broke = True
                break
        if not broke:
            end = end[0][0]
    else:
        end = end[0][0]



    path = [end]
    for i in reversed(range(image.shape[0]-1)):
        MIN = 100000

        column = path[-1]
        current = column
        if column+1 < image.shape[1]:
            if M[i,column+1] < MIN:
                current = column+1
                MIN = M[i,column+1]
        if column - 1 > 0:
            if M[i,column-1] < MIN:
                current = column-1
                MIN = M[i,column-1]
        if M[i,column] < MIN:
            current = column
            MIN = M[i,column]  
        path.append(current)
    return np.array(list(reversed(path)))




def vertical_seam(image, energy_func = "abs"):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float64)

    if energy_func == "abs":
        y = np.concatenate([np.zeros((1,image.shape[1])),np.diff(image, axis = 0)], axis = 0)
        x = np.concatenate([np.zeros((image.shape[0],1)),np.diff(image, axis = 1)], axis = 1)
        energy =  np.abs(x) + np.abs(y)
    elif energy_func == "norm": 
        y = np.concatenate([np.zeros((1,image.shape[1])),np.diff(image, axis = 0)], axis = 0)
        x = np.concatenate([np.zeros((image.shape[0],1)),np.diff(image, axis = 1)], axis = 1)
        energy =  np.sqrt(x**2 + y**2)
    elif energy_func == "sobel1_abs":
        energy = np.abs(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=1)) + np.abs(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=1))
    elif energy_func == "sobel3_abs":
        energy = np.abs(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)) + np.abs(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3))
    elif energy_func == "sobel1_norm":
        energy = np.sqrt(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=1)**2 + cv2.Sobel(image,cv2.CV_64F,0,1,ksize=1)**2) 
    elif energy_func == "sobel3_norm":
        energy = np.sqrt(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)**2 + cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)**2) 
    else:
        return None

    M = energy.copy()
    for i in range(1,image.shape[0]):
        for j in range(image.shape[1]):
            if j+1 >= image.shape[1]:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j-1],M[i-1,j]])
            elif j - 1 <=0:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j],M[i-1,j+1]])
            else:
                M[i,j] = energy[i,j] +  np.min([M[i-1,j-1],M[i-1,j],M[i-1,j+1]])
    MIN = M[-1].min()
    end = []
    for j in range(image.shape[1]):
        if MIN == M[-1,j]:
            end.append([j])

    if len(end) != 1:
        broke = False
        for i in reversed(range(image.shape[0]-1)):          
            for index,path in enumerate(end):
                MIN = 100000
                column = path[-1]
                current = column
                if column+1 < image.shape[1]:
                    if M[i,column+1] < MIN:
                        current = column+1
                        MIN = M[i,column+1]
                if column - 1 > 0:
                    if M[i,column-1] < MIN:
                        current = column-1
                        MIN = M[i,column-1]
                if M[i,column] < MIN:
                    current = column
                    MIN = M[i,column]  
                end[index].append(current)
            e = np.array(end)
            MINIMUM = M[i,e[:,-1]].min()

            drop = []
            for path in end:
                if M[i,path[-1]] != MINIMUM:
                    drop.append(path)
            for d in drop:
                end.remove(d)

            if len(end) == 1:
                end = end[0][0]
                broke = True
                break
        if not broke:
            end = end[0][0]
    else:
        end = end[0][0]
    path = [end]
    for i in reversed(range(image.shape[0]-1)):
        MIN = 100000

        column = path[-1]
        current = column
        if column+1 < image.shape[1]:
            if M[i,column+1] < MIN:
                current = column+1
                MIN = M[i,column+1]
        if column - 1 > 0:
            if M[i,column-1] < MIN:
                current = column-1
                MIN = M[i,column-1]
        if M[i,column] < MIN:
            current = column
            MIN = M[i,column]  
        path.append(current)
    return np.array(list(reversed(path)))


def compress_horizontal(image, target_width,energy_func = "abs", forward = False, P = False):
    image_new = image.copy()

    def process(im, p):
        if p >0 and p < len(im)-1:
            return np.array(list(im[:p]) + list(im[p+1:]))
        elif p == 0:
            return np.array(list(im[p+1:]))
        else:
            return np.array(list(im[:p]))

    while image_new.shape[1] > target_width:
        if forward:
            path = vertical_seam_forward(image_new,energy_func,P)
        else:
            path = vertical_seam(image_new,energy_func)
        image1 = image_new[:,:,0]
        image2 = image_new[:,:,1]
        image3 = image_new[:,:,2]

        image1 = np.array([process(im, p) for p, im in zip(path,list(image1))])

        image2 = np.array([process(im, p) for p, im in zip(path,list(image2))])

        image3 = np.array([process(im, p) for p, im in zip(path,list(image3))])
        image_new = np.concatenate([np.expand_dims(image1, axis = 2),np.expand_dims(image2, axis = 2),np.expand_dims(image3, axis = 2)], axis = 2)
        #print(image_new.shape)
    return image_new

def compress_horizontal_demo(image, target_width,energy_func = "abs", forward = False, P = False):
    image_new = image.copy()
    #image_new_temp = image.copy()
    k = image_new.shape[1]-target_width
    path = []

    def process(im, p):
        if p >0 and p < len(im)-1:
            return np.array(list(im[:p]) + list(im[p+1:]))
        elif p == 0:
            return np.array(list(im[1:]))
        else:
            return np.array(list(im[:p]))
    for i in range(k):
        if forward:
            path_new = vertical_seam_forward(image_new,energy_func,P)
        else:
            path_new = vertical_seam(image_new,energy_func)
        image1 = image_new[:,:,0]
        image2 = image_new[:,:,1]
        image3 = image_new[:,:,2]

        image1 = np.array([process(im, p) for p, im in zip(path_new,list(image1))])

        image2 = np.array([process(im, p) for p, im in zip(path_new,list(image2))])

        image3 = np.array([process(im, p) for p, im in zip(path_new,list(image3))])

        image_new = np.concatenate([np.expand_dims(image1, axis = 2),np.expand_dims(image2, axis = 2),np.expand_dims(image3, axis = 2)], axis = 2)
        path.append(path_new)


    #for i in reversed(range(1,k)):
    #    for j in reversed(range(i)):
    #        path[i] = path[i] + (path[i]>path[j])
    path = list(reversed(path))
    path = np.array(path)
    #offset = path.mean(axis = 1).argsort()
    #path.sort(axis = 0)

    def process(im, p, c):
        if p >0 and p < len(im)-1:
            return np.array(list(im[:p]) +[c]+ list(im[p:]))
        elif p == 0:
            return np.array([c] + list(im))
        else:
            return np.array(list(im) +[c])
    for i in range(k):
        for j in range(i):
            path[i] = path[i] + (path[i]>=path[j])
        path_new = path[i] 
        image1 = image_new[:,:,0]
        image2 = image_new[:,:,1]
        image3 = image_new[:,:,2]
        image1 = np.array([process(im, p, 0) for p, im in zip(path_new,list(image1))]).astype(np.uint8)
        image2 = np.array([process(im, p, 0) for p, im in zip(path_new,list(image2))]).astype(np.uint8)
        image3 = np.array([process(im, p, 255) for p, im in zip(path_new,list(image3))]).astype(np.uint8)
        image_new = np.concatenate([np.expand_dims(image1, axis = 2),np.expand_dims(image2, axis = 2),np.expand_dims(image3, axis = 2)], axis = 2)
    return image_new


def elongate_horizontal(image, target_width,energy_func = "abs", forward = False, P = False):
    image_new = image.copy()
    image_new_temp = image.copy()
    k = target_width-image_new.shape[1]
    path = []

    def process(im, p):
        if p >0 and p < len(im)-1:
            return np.array(list(im[:p]) + list(im[p+1:]))
        elif p == 0:
            return np.array(list(im[p+1:]))
        else:
            return np.array(list(im[:p]))

    for i in range(k):
        if forward:
            path_new = vertical_seam_forward(image_new_temp,energy_func,P)
        else:
            path_new = vertical_seam(image_new_temp,energy_func)
        image1 = image_new_temp[:,:,0]
        image2 = image_new_temp[:,:,1]
        image3 = image_new_temp[:,:,2]

        image1 = np.array([process(im, p) for p, im in zip(path_new,list(image1))])

        image2 = np.array([process(im, p) for p, im in zip(path_new,list(image2))])

        image3 = np.array([process(im, p) for p, im in zip(path_new,list(image3))])

        image_new_temp = np.concatenate([np.expand_dims(image1, axis = 2),np.expand_dims(image2, axis = 2),np.expand_dims(image3, axis = 2)], axis = 2)
        path.append(path_new)


    for i in reversed(range(1,k)):
        for j in reversed(range(i)):
            path[i] = path[i] + (path[i]>=path[j])
    #for i in reversed(range(1,k)):
    #    for j in reversed(range(i)):
    #        path[i] = path[i] + (path[i]>=path[j])
    path = np.array(path)


    def process(im, p):
        if p >0 and p < len(im)-1:
            return np.array(list(im[:p]) +[int((float(im[p-1])+float(im[p]))/2)]+ list(im[p:]))
        elif p == 0:
            return np.array([im[p]] + list(im))
        else:
            return np.array(list(im) +[im[p]])

    for i in range(k):
        for j in range(i):
            path[i] = path[i] + (path[i]>=path[j])
        path_new = path[i] 
        image1 = image_new[:,:,0]
        image2 = image_new[:,:,1]
        image3 = image_new[:,:,2]

        

        image1 = np.array([process(im, p) for p, im in zip(path_new,list(image1))]).astype(np.uint8)
        image2 = np.array([process(im, p) for p, im in zip(path_new,list(image2))]).astype(np.uint8)
        image3 = np.array([process(im, p) for p, im in zip(path_new,list(image3))]).astype(np.uint8)
        image_new = np.concatenate([np.expand_dims(image1, axis = 2),np.expand_dims(image2, axis = 2),np.expand_dims(image3, axis = 2)], axis = 2)
    return image_new


def elongate_horizontal_demo(image, target_width,energy_func = "abs",forward = False, P = False):
    image_new = image.copy()
    image_new_temp = image.copy()
    k = target_width-image_new.shape[1]
    path = []
    def process(im, p):
        if p >0 and p < len(im)-1:
            return np.array(list(im[:p]) + list(im[p+1:]))
        elif p == 0:
            return np.array(list(im[p+1:]))
        else:
            return np.array(list(im[:p]))
    for i in range(k):
        if forward:
            path_new = vertical_seam_forward(image_new_temp,energy_func,P)
        else:
            path_new = vertical_seam(image_new_temp,energy_func)
        image1 = image_new_temp[:,:,0]
        image2 = image_new_temp[:,:,1]
        image3 = image_new_temp[:,:,2]

        image1 = np.array([process(im, p) for p, im in zip(path_new,list(image1))])

        image2 = np.array([process(im, p) for p, im in zip(path_new,list(image2))])

        image3 = np.array([process(im, p) for p, im in zip(path_new,list(image3))])

        image_new_temp = np.concatenate([np.expand_dims(image1, axis = 2),np.expand_dims(image2, axis = 2),np.expand_dims(image3, axis = 2)], axis = 2)
        path.append(path_new)


    for i in reversed(range(1,k)):
        for j in reversed(range(i)):
            path[i] = path[i] + (path[i]>=path[j])
    #for i in reversed(range(1,k)):
    #    for j in reversed(range(i)):
    #        path[i] = path[i] + (path[i]>=path[j])

    path = np.array(path)
    #offset = path.mean(axis = 1).argsort()
    #path.sort(axis = 0)
    def process(im, p, c):
        if p >0 and p < len(im)-1:
            return np.array(list(im[:p]) +[c]+ list(im[p:]))
        elif p == 0:
            return np.array([c] + list(im))
        else:
            return np.array(list(im) +[c])
    for i in range(k):
        for j in range(i):
            path[i] = path[i] + (path[i]>=path[j])
        path_new = path[i] 
        image1 = image_new[:,:,0]
        image2 = image_new[:,:,1]
        image3 = image_new[:,:,2]
        image1 = np.array([process(im, p, 0) for p, im in zip(path_new,list(image1))]).astype(np.uint8)
        image2 = np.array([process(im, p, 0) for p, im in zip(path_new,list(image2))]).astype(np.uint8)
        image3 = np.array([process(im, p, 255) for p, im in zip(path_new,list(image3))]).astype(np.uint8)
        image_new = np.concatenate([np.expand_dims(image1, axis = 2),np.expand_dims(image2, axis = 2),np.expand_dims(image3, axis = 2)], axis = 2)
    return image_new