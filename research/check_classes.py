import torch, json

ckpt = torch.load('research/models/mobilenetv2_best.pt', map_location='cpu', weights_only=False)
print('Checkpoint keys:', list(ckpt.keys()))
if 'classes' in ckpt:
    print('classes count:', len(ckpt['classes']))
    print('First 10 classes:', ckpt['classes'][:10])
if 'class_to_idx' in ckpt:
    print('class_to_idx sample:', list(ckpt['class_to_idx'].items())[:10])
if 'label_names' in ckpt:
    print('label_names sample:', ckpt['label_names'][:10])

with open('research/split_manifest.json') as f:
    manifest = json.load(f)
    print('manifest classes count:', len(manifest['classes']))
    print('manifest first 10:', manifest['classes'][:10])
