import torch
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.models_setup import generate_model
from utils.utils import (train, validate)


def prepare_dataset(
    model,
    loader,
    config: {}
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_up = []
    data_down = []

    b = 0
    for data in loader:
        data.to(device)
        diff = (data.y - model(data)).detach()
        data.y = diff
        orig_list = data.to_data_list()
        upcount = 0
        dncount = 0
        c = 0
        for i in range(len(diff)):
            if b == 0 and c == 0:
                data_up.append(orig_list[i])
                data_down.append(orig_list[i])
            elif diff[i] >= 0:
                data_up.append(orig_list[i])
                upcount += 1
            else:
                data_down.append(orig_list[i])
                dncount += 1
            c += 1 
        #print('Batch', b, 'pos:', upcount, 'neg:', dncount)
        b += 1

    print('Total pos: ', len(data_up), 'neg:', len(data_down))

    loader_up = DataLoader(data_up, batch_size=config["batch_size"], shuffle=True)
    loader_down = DataLoader(data_down, batch_size=config["batch_size"], shuffle=True)

    return loader_up, loader_down

# PCIP: Prediction Interval Coverage Probability 
def compute_pcip(
        model,
        model_up,
        model_down,
        c_up,
        c_down,
        loader
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    covered = 0.0
    for data in loader:
        data.to(device)
        result_down = model(data) - model_down(data)*c_down
        result_up = model(data) - model_up(data)*c_up
        covered += ((result_down <= data.y) & (data.y <= result_up)).sum()
        
    print('COVERED IN PI:', covered, 'IN A TOTAL OF', len(loader.dataset), 'PCIP:', covered/(len(loader.dataset)))
    

def prediction_interval(
        model,
        train_loader,
        val_loader,
        test_loader,
        config: {}
        
):

    print(
        f"Starting the prediction interval computation"
    )
    train_loader_up, train_loader_down = prepare_dataset(
        model=model,
        loader=train_loader,
        config=config
    )

    val_loader_up, val_loader_down = prepare_dataset(
        model=model,
        loader=val_loader,
        config=config
    )

    print('UP TRAIN SIZE', len(train_loader_up.dataset), 'VAL SIZE', len(val_loader_up.dataset))
    print('DOWN TRAIN SIZE', len(train_loader_down.dataset), 'VAL SIZE', len(val_loader_down.dataset))

    model_up = generate_model(
        model_type=config["model_type"],
        input_dim=len(config["atom_features"]),
        dataset=train_loader_up.dataset,
        config=config
    )

    model_down = generate_model(
        model_type=config["model_type"],
        input_dim=len(config["atom_features"]),
        dataset=train_loader_down.dataset,
        config=config
    )

    optimizer_up = torch.optim.AdamW(model_up.parameters(), lr=config["learning_rate"])
    optimizer_down = torch.optim.AdamW(model_down.parameters(), lr=config["learning_rate"])

    scheduler_up = ReduceLROnPlateau(optimizer_up, mode="min", factor=0.5, patience=5, min_lr=0.00001)
    scheduler_down = ReduceLROnPlateau(optimizer_down, mode="min", factor=0.5, patience=5, min_lr=0.00001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_up.to(device)
    model_down.to(device)

    #--------- TRAIN THE UP & DOWN NETWORKS ---------
    num_epoch = config["num_epoch"]
    for epoch in range(0, num_epoch):
        train_mae = train(train_loader_up, model_up, optimizer_up, config["output_dim"])
        val_mae = validate(val_loader_up, model_up, config["output_dim"])
        scheduler_up.step(val_mae)
        print(f"Epoch: {epoch:02d}, Train MAE: {train_mae:.8f}, Val MAE: {val_mae:.8f}")

    for epoch in range(0, num_epoch):
        train_mae = train(train_loader_down, model_down, optimizer_down, config["output_dim"])
        val_mae = validate(val_loader_down, model_down, config["output_dim"])
        scheduler_down.step(val_mae)
        print(f"Epoch: {epoch:02d}, Train MAE: {train_mae:.8f}, Val MAE: {val_mae:.8f}")


    #--------- MOVE THE UPPER BOUND ---------
    n_iter = 30         # Parametrize this
    quantile = 0.90     # Parametrize this
    num_outlier = len(train_loader.dataset) * (1-quantile)/2


    c_up0 = 0.0
    c_up1 = 10.0
    f0 = 0.0
    f1 = 0.0
    for data in train_loader:
        data.to(device)
        f0 += (data.y >= model(data) + c_up0 * model_up(data)).sum()
        f1 += (data.y >= model(data) + c_up1 * model_up(data)).sum()

    f0 -= num_outlier
    f1 -= num_outlier

    iter = 0
    while iter <= n_iter and f0 != 0 and f1 != 0:

        c_up2 = (c_up0 + c_up1)/2.0

        f2 = 0.0
        for data in train_loader:
            data.to(device)
            f2 += (data.y >= model(data) + c_up2 * model_up(data)).sum()
        f2 -= num_outlier

        if f2 == 0: 
            break
        elif f2 > 0:
            c_up0 = c_up2
            f0 = f2
        else:
            c_up1 = c_up2
            f1 = f2
        print('{}, f0: {}, f1: {}, f2: {}'.format(iter, f0, f1, f2))
        iter += 1
    c_up = c_up2


    #--------- MOVE THE LOWER BOUND ---------
    c_down0 = 0.0
    c_down1 = 10.0
    f0 = 0.0
    f1 = 0.0
    for data in train_loader:
        data.to(device)
        f0 += (data.y >= model(data) + c_down0 * model_down(data)).sum()
        f1 += (data.y >= model(data) + c_down1 * model_down(data)).sum()

    f0 -= num_outlier
    f1 -= num_outlier

    iter = 0
    while iter <= n_iter and f0 != 0 and f1 != 0:

        c_down2 = (c_down0 + c_down1)/2.0

        f2 = 0.0
        for data in train_loader:
            data.to(device)
            f2 += (data.y >= model(data) + c_down2 * model_down(data)).sum()
        f2 -= num_outlier

        if f2 == 0: 
            break
        elif f2 > 0:
            c_down0 = c_down2
            f0 = f2
        else:
            c_down1 = c_down2
            f1 = f2
        print('{}, f0: {}, f1: {}, f2: {}'.format(iter, f0, f1, f2))
        iter += 1
    c_down = c_down2


    compute_pcip(model, model_up, model_down, c_up, c_down, test_loader)
    
