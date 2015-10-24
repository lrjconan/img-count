from matplotlib import pyplot as plt
import selective_search
import cv2

if __name__ == '__main__':
    image_filenames = [
        '000015.jpg',
        'cat.jpg'
    ]
    boxes = selective_search.get_windows(image_filenames)
    print boxes
    print(boxes)
    print(boxes[0].shape)
    print(boxes[1].shape)
    print(type(boxes))
    print(type(boxes[0]))
    for fn in image_filenames:
        img = cv2.imread(fn, 0)
        print img.shape
        for i in xrange(boxes[0].shape[0]):
            box = boxes[0][i]
            cv2.rectangle(img, 
                (box[0], box[1]), 
                (box[2], box[3]), 
                color=(255, 0, 0), 
                thickness=1)
        plt.imshow(img, cmap='gray')
        plt.show()
