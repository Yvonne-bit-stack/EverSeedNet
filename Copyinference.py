import os
import torch
import logging
import argparse
from utils import factory
from utils.data_manager import DataManager
from datetime import datetime

def load_model_weights(model, model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cuda'))
    logging.info(f"State_dict keys: {state_dict.keys()}")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logging.info(f"Model weights loaded from {model_path}")
    logging.info(f"Missing keys: {missing_keys}")
    logging.info(f"Unexpected keys: {unexpected_keys}")


def inference(args):
    """
    模型推理验证
    :param args: 参数字典
    """
    # 设置日志
    logfilename = f"logs/inference_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting inference...")

    # 设置随机种子
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载数据管理器
    data_manager = DataManager(
        dataset_name=args["dataset"],
        shuffle=args["shuffle"],
        seed=args["seed"],
        init_cls=args["init_cls"],
        increment=args["increment"],
    )

    # 加载模型
    model = factory.get_model(args["model_name"], args)

    # 加载指定任务的模型权重
    task_id = args["task_id"]
    model_name = args["model_name"]
    dataset = args["dataset"]
    init_cls = args["init_cls"]
    increment = args["increment"]
    seed = args["seed"]
    convnet_type = args["convnet_type"]
    time_str = args["time_str"]

    model_path = os.path.join(args["model_dir"], f"task_{task_id}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found: {model_path}")

    load_model_weights(model, model_path)

    # 进行推理验证
    model.eval()
    test_loader = data_manager.get_test_loader(task_id)
    correct = 0
    total = 0

    device = torch.device(args["device"][0])
    model.to(device)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    logging.info(f"Task {task_id} Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for PyCIL")
    parser.add_argument("--data_dir", type=str, default="seed_dataset2", help="路径包含 train/val/test")
    parser.add_argument("--model_dir", type=str, default="models/der/iseeds/20/1/1993/resnet32/2025-05-28_21-25-31", help="保存 task_0.pth 的目录")
    parser.add_argument("--model_name", type=str, default="der", help="模型名称，需与 factory 中一致")
    parser.add_argument("--prefix", type=str, default="reproduce", help="日志前缀（用于一些模型初始化）")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--convnet_type", type=str, default="resnet32")
    parser.add_argument("--memory_size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="iseeds", help="数据集名称")
    parser.add_argument("--init_cls", type=int, default=20, help="初始类别数量")
    parser.add_argument("--increment", type=int, default=1, help="类别增量")
    parser.add_argument("--shuffle", action="store_true", help="是否打乱数据集")
    parser.add_argument("--task_id", type=int, default=0, help="任务编号")
    parser.add_argument("--time_str", type=str, default=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), help="时间戳，用于匹配权重文件")
    args = parser.parse_args()

    # 将 argparse.Namespace 转换为字典
    args = vars(args)

    inference(args)

