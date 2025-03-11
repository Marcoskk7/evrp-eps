import torch
import torch.backends.cudnn as cudnn
import os
import pickle
import numpy as np
import pandas as pd
import random

def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(gpu):
    """
    Parameters
    ----------
    gpu: str or int
        GPU设置。可以是以下格式：
        - 字符串(如 "0,1,2"): 使用指定的多个GPU
        - 整数: 使用单个GPU（如 0）
        - -1: 使用CPU

    Returns
    -------
    use_cuda: bool
        是否使用GPU
    device: str
        主设备名称
    """
    if isinstance(gpu, str) and ',' in gpu:
        # 多GPU设置
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        assert torch.cuda.is_available(), "没有可用的GPU。"
        device = "cuda"
        use_cuda = True
        cudnn.benchmark = True
        print(f'使用多个GPU: {gpu}')
    elif isinstance(gpu, (int, str)):
        # 单GPU或CPU设置
        gpu = int(gpu)
        if gpu >= 0:
            assert torch.cuda.is_available(), "没有可用的GPU。"
            torch.cuda.set_device(gpu)
            device = f"cuda:{gpu}"
            use_cuda = True
            cudnn.benchmark = True
            print(f'使用单个GPU #{gpu}')
        else:
            device = "cpu"
            use_cuda = False
            print('使用CPU')
    
    if use_cuda:
        print(f'可用GPU数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    
    return use_cuda, device

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename):
    filedir = os.path.split(filename)[0]
    print(f"saving a dataset to {filename}...", end="", flush=True)
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    print("done")

def load_dataset(filename):
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)

def output_tour_to_csv(input, tours, output_dir):
    """
    Parameters
    ----------
    input: dict of torch.tensor
    tours: 3d list [batch_size x num_vehicles x num_steps_of_the_vehicles]
        tour of each vehicle
    output_dir: str
    """
    batch = 0
    tours = tours[batch]
    num_vehicles = len(tours)

    # add initial position to tour list
    initial_position = input["vehicle_initial_position_id"][batch]
    for i in range(num_vehicles):
        tours[i].insert(0, (initial_position[i].item(), 0.0, 0.0))

    # generate multi_index
    info_list = ["tour", "travel_time (h)", "charge_time (h)"]
    vehicle_id = []
    info_name = []
    for i in range(num_vehicles):
        for info in info_list:
            vehicle_id.append(str(i))
            info_name.append(info)
    df_ind = pd.DataFrame({"vehicle_id": vehicle_id, "info": info_name})
    multi_index = pd.MultiIndex.from_frame(df_ind)

    # generate pandas framework
    max_steps = np.max([len(tours[i]) for i in range(num_vehicles)])
    data = {}; data_w_time = {}
    for i in range(max_steps):
        step = str(i)
        data[step] = []; data_w_time[step] = []
        for j in range(num_vehicles):
            if i >= len(tours[j]):
                data[step].append(np.nan)
                for k in range(len(info_list)):
                    data_w_time[step].append(np.nan)
            else:
                data[step].append(tours[j][i][0])
                for k in range(len(info_list)):                    
                    data_w_time[step].append(tours[j][i][k])

    df_tour = pd.DataFrame(
        data,
        index = [str(i) for i in range(num_vehicles)]
    )
    df_tour.index.name = "vehicle_id"

    df_tour_w_time = pd.DataFrame(
        data_w_time,
        index=multi_index
    )

    # save csv files
    df_tour.to_csv(f"{output_dir}/tour.csv")
    df_tour_w_time.to_csv(f"{output_dir}/tour_w_time.csv")