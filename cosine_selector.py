import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import shutil

# ----------- 1. 参数设置 -------------
DATA_ROOT = 'seed_dataset2/train'
SELECTED_SAVE_DIR = 'selected_by_cosine'  # 存放挑选出来的图片
TOPK = 5
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------- 2. 特征提取模型（用 resnet18 或你已有模型）-------------
model = resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # 提取的是倒数第二层特征
model = model.to(DEVICE)
model.eval()

# ----------- 3. 图像预处理 -------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------- 4. 遍历每一个类别 -------------
for class_id in sorted(os.listdir(DATA_ROOT)):
    class_path = os.path.join(DATA_ROOT, class_id)
    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class {class_id}...")
    image_paths = [os.path.join(class_path, fname) for fname in os.listdir(class_path)]
    features = []  # ✅ 初始化特征列表

    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model(img).squeeze(0)
            features.append(feat.cpu())
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    if len(features) < 2:
        print(f"⚠️  类别 {class_id} 中图片数量过少，跳过该类")
        continue

    features = torch.stack(features)
    normed_feat = F.normalize(features, dim=1)
    similarity_matrix = torch.mm(normed_feat, normed_feat.T)  # [N, N]

    avg_sim = similarity_matrix[~torch.eye(similarity_matrix.shape[0], dtype=bool)].mean().item()
    avg_sim_clamped = max(avg_sim, 0.1)  # 防止为 0
    max_select = int(avg_sim_clamped * 20)
    max_select = min(max_select, len(image_paths))  # 防止超出范围

    print(f"Class {class_id} 平均相似度：{avg_sim:.3f}，选图数量：{max_select}")

    image_scores = similarity_matrix.sum(dim=1)
    topk_indices = torch.topk(image_scores, max_select).indices

    save_dir = os.path.join(SELECTED_SAVE_DIR, class_id)
    os.makedirs(save_dir, exist_ok=True)

    for idx in topk_indices:
        fname = os.path.basename(image_paths[idx])
        shutil.copy(image_paths[idx], os.path.join(save_dir, fname))

print("\n✅ Done! 所有类别的相似图片已按自适应数量保存至：{}".format(SELECTED_SAVE_DIR))

