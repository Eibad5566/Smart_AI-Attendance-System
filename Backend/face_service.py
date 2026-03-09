import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_embeddings_and_boxes(image):
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return [], []
    
    faces = mtcnn(image)
    embeddings = []
    valid_boxes = []
    
    with torch.no_grad():
        for i, face in enumerate(faces):
            if face is not None:
                emb = resnet(face.unsqueeze(0))
                embeddings.append(emb[0].numpy())
                valid_boxes.append(boxes[i])
                
    return embeddings, valid_boxes