import os
import torch
import logging
import argparse
from utils import factory
from utils.data_manager import DataManager
from datetime import datetime
import time


def load_model_weights(model, model_path):
    """
    加载模型权重
    :param model: 模型实例
    :param model_path: 模型权重文件路径
    """
    state_dict = torch.load(model_path, map_location=torch.device('cuda'))
    logging.info(f"State_dict keys: {state_dict.keys()}")

    # 检查模型的键名
    model_keys = model._network.state_dict().keys()
    logging.info(f"Model keys: {model_keys}")

    # 检查不匹配的键
    missing_keys = [key for key in model_keys if key not in state_dict]
    unexpected_keys = [key for key in state_dict if key not in model_keys]

    if missing_keys:
        logging.warning(f"Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

    # 加载权重，忽略不匹配的键
    model._network.load_state_dict({k: v for k, v in state_dict.items() if k in model_keys})
    logging.info(f"Model weights loaded from {model_path}")

# def inference(args):
#     """
#     模型推理验证
#     :param args: 参数字典
#     """
#     # 设置日志
#     logfilename = f"logs/inference_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
#     os.makedirs("logs", exist_ok=True)
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(filename)s] => %(message)s",
#         handlers=[
#             logging.FileHandler(filename=logfilename),
#             logging.StreamHandler(),
#         ],
#     )
#     logging.info("Starting inference...")
#
#     # 设置随机种子
#     torch.manual_seed(args["seed"])
#     torch.cuda.manual_seed(args["seed"])
#     torch.cuda.manual_seed_all(args["seed"])
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#     # 加载数据管理器
#     data_manager = DataManager(
#         dataset_name=args["dataset"],
#         shuffle=args["shuffle"],
#         seed=args["seed"],
#         init_cls=args["init_cls"],
#         increment=args["increment"],
#     )
#
#     # 加载模型
#     model = factory.get_model(args["model_name"], args)
#
#     # 加载指定任务的模型权重
#     task_id = args["task_id"]
#     model_path = os.path.join(args["model_dir"], f"task_{task_id}.pth")
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model weights file not found: {model_path}")
#
#     load_model_weights(model, model_path)
#
#     # 推理开始
#     model.eval()
#     test_loader = data_manager.get_test_loader(task_id)
#     correct = 0
#     total = 0
#
#     all_preds = []
#     all_labels = []
#
#     device = torch.device(args["device"])
#     model.to(device)
#
#     start_time = time.time()
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
#
#             all_preds.extend(predicted.cpu().tolist())
#             all_labels.extend(targets.cpu().tolist())
#     end_time = time.time()
#
#     # 准确率和耗时统计
#     accuracy = 100 * correct / total
#     elapsed = end_time - start_time
#
#     logging.info(f"✅ 推理完成！")
#     logging.info(f"📊 Task {task_id} Accuracy: {accuracy:.2f}%")
#     logging.info(f"⏱️  耗时: {elapsed:.2f} 秒")
#     print(f"\n✅ 推理完成！📊 Accuracy: {accuracy:.2f}%, ⏱️ Time: {elapsed:.2f}s")
#
#     # 保存预测结果
#     output_file = os.path.join("logs", f"predictions_task_{task_id}.txt")
#     with open(output_file, "w") as f:
#         for pred in all_preds:
#             f.write(f"{pred}\n")
#     logging.info(f"📝 预测结果保存到: {output_file}")
def inference(args):
    """
    模型推理验证
    :param args: 参数字典
    """
    # 打印 args 的值
    for key, value in args.items():
        logging.info(f"{key}: {value}")

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

    # 打印 args["device"] 的值
    logging.info(f"Device: {args['device']}")

    # 检查 args["device"] 的值是否有效
    if not isinstance(args["device"], list) or len(args["device"]) == 0:
        raise ValueError("Invalid device list. Please provide a valid device list, e.g., ['cuda:0'] or ['cpu']")

    device_str = args["device"][0]
    valid_devices = ["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    if device_str not in valid_devices:
        raise ValueError(f"Invalid device string: {device_str}. Please provide a valid device string, e.g., 'cuda:0' or 'cpu'")

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
    model.eval()  # 确保调用 eval 方法
    test_loader = data_manager.get_test_loader(task_id)
    correct = 0
    total = 0

    device = torch.device(device_str)  # 确保设备类型正确
    model.to(device)  # 确保调用 to 方法

    with torch.no_grad():
        for idx, inputs, targets in test_loader:
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
    parser.add_argument("--device", type=str,  default=["0"])
    parser.add_argument("--dataset", type=str, default="iseeds", help="数据集名称")
    parser.add_argument("--init_cls", type=int, default=20, help="初始类别数量")
    parser.add_argument("--increment", type=int, default=1, help="类别增量")
    parser.add_argument("--shuffle", action="store_true", help="是否打乱数据集")
    parser.add_argument("--task_id", type=int, default=0, help="任务编号")
    parser.add_argument("--time_str", type=str, default=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), help="时间戳，用于匹配权重文件")
    args = parser.parse_args()

    args = vars(args)
    inference(args)

 # python inference.py --data_dir seed_dataset2 --model_dir logs/der/iseeds/20/1/reproduce_1993_resnet32 --model_name der --prefix reproduce --seed 1993 --convnet_type resnet32 --device cuda:0