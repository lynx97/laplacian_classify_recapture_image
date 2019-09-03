from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from PIL import Image
import cv2
import numpy as np
from itertools import repeat
import base64
import io
from scipy.ndimage.measurements import label

tf.app.flags.DEFINE_string('server', 'localhost:8089',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('type', '', 'type of the id 0: CCCD mat truoc, 1: CCCD mat sau, 2: CMND mat truoc, 3: CMND mat sau')
FLAGS = tf.app.flags.FLAGS

#------------------------------------------------------------------------------
#Get coordinates of patches when sliding 2D
#------------------------------------------------------------------------------
def slide2d(sz, K, S):
    H, W = sz
    i = 0; j = 0
    n_H, n_W = 0, 0
    coords = []
    while True:
        if i+K > H:
            break
        n_W = 0
        while True:
            if j+K > W:
                break
            coords.append((i, j))
            j += S
            n_W += 1
        i += S
        j = 0
        n_H += 1
    return coords, n_H, n_W
def crop_patches(img, coords, patch_sz):
    def crop(img, coord, patch_sz):
        i, j = coord
        patch = img[i:i+patch_sz, j:j+patch_sz, ...].transpose((2,0,1))
        return patch

    patch_obj = map(crop, repeat(img), coords, repeat(patch_sz))
    patches = list(patch_obj)
    return np.array(patches)

def fusion(max_component, length_patches, threshole_range):
    #threshole_range: tuple
    val = max_component/length_patches
    if threshole_range[0] == threshole_range[1]:
        if(val >= threshole_range[1]):
            return 1
        else:
            return 0
    if val >= threshole_range[1]:
        return 1
    elif val < threshole_range[0]:
        return 0
    else:
        return 2

def laplacian_filter(src):
    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3,3),np.float32)
    kernel[0][0] = 0
    kernel[0][1] = -1
    kernel[0][2] = 0
    kernel[1][0] = -1
    kernel[1][1] = 4
    kernel[1][2] = -1
    kernel[2][0] = 0
    kernel[2][1] = -1
    kernel[2][2] = 0
    dst = cv2.filter2D(src_gray,-1,kernel)
    dst = np.expand_dims(dst, axis=-1)
    return dst

def predict_batches(batches, stub):
    #batches (size, patch, patch, 3)
    #laplacian -> predict_proba
    batch_laplacian = np.empty([len(batches), batches.shape[1], batches.shape[2], 1], dtype=int)
    for i in range(len(batches)):
        lap_item = laplacian_filter(batches[i])
        batch_laplacian[i] = lap_item
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "recapture"
    request.model_spec.signature_name = "predict"
    request.inputs["images"].CopyFrom(
        tf.contrib.util.make_tensor_proto(batch_laplacian, dtype="float32", allow_broadcast=True))

    # Call the TFServing Predict API
    results = stub.Predict.future(request, None)
    pred_proba = results.result().outputs["scores"].float_val       #list scores

    pred_proba = np.array(pred_proba) 
    pred_proba = pred_proba.reshape(-1, 2)                      #reshape ve size (len(batches), 2)

    return pred_proba

def my_post_process(softmax_one, thresh):
    for i in range(len(softmax_one)):
        if(softmax_one[i] >= thresh):
            softmax_one[i] = 1
        else:
            softmax_one[i] = 0
    return softmax_one

def get_max_connected_component(lll, w, h, patch_size):
    col = h // patch_size
    row = w // patch_size
    lll = lll.reshape(col, row)
    structure = np.ones((3, 3), dtype=np.int)
    structure[0][0] = 0
    structure[0][2] = 0
    structure[2][0] = 0
    structure[2][2] = 0
    labeled, ncomponents = label(lll, structure)
    max_num = 0
    for i in range(ncomponents):
        tmp = (labeled == (i+1)).sum()
        if(tmp > max_num):
            max_num = tmp
    return max_num

def predict_on_image(img_input, type_img, stub, threshole_range, PATCH):
    THRESHOLE = 0.5                         #threshole for softmax
    if(type_img == 1):                      #1: CCCD mat sau, cat phan van tay va phan ma vach
        img = img_input[224:, 192:]
    elif(type_img == 3):                    #3: CMND mat sau, cat phan van tay
        img = img_input[:, 224:]
    else:
        img = img_input
    coords, _, _ = slide2d(sz=img.shape[:2], K=PATCH, S=PATCH)
    patches = crop_patches( img=img, coords=coords, patch_sz=PATCH)             #shape (3, PATCH, PATCH)
    batch = np.array(patches)
    batch = np.moveaxis(batch, 1, -1)       #convert ve dang (PATCH, PATCH, 3)
    softmaxs = predict_batches(batch, stub)
    lb = my_post_process(softmaxs[:,1], THRESHOLE)  #labels
    max_component = get_max_connected_component(lb, img.shape[1], img.shape[0], PATCH)
    decision = fusion(max_component, len(patches), threshole_range)
    return decision

def inference(image_input, type_img, stub, threshole_1, threshole_2, PATCH):
    #image: numpy array of image
    #type_img: type of id
    #threshole_1, threshole_2:  threshole lan 1, lan 2
    decision = predict_on_image(image_input, 9999, stub, threshole_1, PATCH)        #9999 la so bat ki
    if(decision == 2):
        decision = predict_on_image(cv2.dilate(image_input, None), type_img, stub, threshole_2, PATCH)
    return decision

def main(_):
    params = {}
    params["channel"] = "RGB"
    patch_size = 64
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Change test image to RGB mode
    img = np.array(Image.open(FLAGS.image).convert(params["channel"]))
    type_id = FLAGS.type  #type of the id 0: CCCD mat truoc, 1: CCCD mat sau, 2: CMND mat truoc, 3: CMND mat sau
    if(int(type_id) == 1):
        threshole_1 = (0.2, 0.45)
        threshole_2 = (0.1, 0.1)
    elif(int(type_id) == 2):
        threshole_1 = (0.23, 0.5)
        threshole_2 = (0.15, 0.15)
    else:
        threshole_1 = (0.17, 0.35)
        threshole_2 = (0.1, 0.1)
    decision = inference(img, int(type_id), stub, threshole_1, threshole_2, patch_size)    #1: la recapture    0: la anh chup binh thuong
    print("Detection result {}".format(decision))

if __name__ == '__main__':
  tf.app.run()
