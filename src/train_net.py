import os
import torch
from src.model import build_model
from src.data.dataset import build_dataset
from src.data.sampler.graph_sampler import build_graph_sampler
from src.data.sampler.node_sampler import build_nodes_sampler
from torch.utils.data import IterableDataset, DataLoader
from src.model.optimizer import build_optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


def train_epoch(cfg, model, dataloader, optimizer, tb_writer, global_steps, scaler):
    """

    :param cfg:
    :param model:
    :param dataloader:
    :param optimizer:
    :param tb_writer:
    :param global_steps:
    :return:
    """
    model = model.train()
    device = torch.device('cuda:0')
    dataloader_it = iter(dataloader)
    loss_records = []
    for batch_id in range(cfg.TRAIN.BATCHES_PER_EPOCH):
        if not cfg.FP16:
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            pos_score, neg_score, loss, auc = model(pos_graph, neg_graph, blocks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with autocast():
                pos_graph, neg_graph, blocks = next(dataloader_it)
                # Copy to GPU
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                pos_score, neg_score, loss, auc = model(pos_graph, neg_graph, blocks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        loss = loss.item()
        if batch_id % 100 == 0:
            print(loss, auc)
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/batch', loss, global_steps)
                tb_writer.add_scalar('AUC/batch', auc, global_steps)

        loss_records.append([loss, auc])
        global_steps += 1
    return loss_records, global_steps


def train(cfg):
    output_path = cfg.OUTPUT_PATH
    assert output_path != ''
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    checkpoints_path = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)

    tb_path = os.path.join(output_path, 'tb')
    tb_writer = SummaryWriter(tb_path)

    cfg.dump(stream=open(os.path.join(output_path, 'config.yaml'), 'w'))

    # load data
    g, train_g, [train_user_ids, val_user_ids, test_user_ids] = build_dataset(cfg)

    # load model
    model = build_model(g, cfg)

    # load samplers
    nodes_sampler = build_nodes_sampler(train_g, cfg)
    neighbor_sampler, collator = build_graph_sampler(train_g, cfg)
    dataloader = DataLoader(nodes_sampler, collate_fn=collator.collate_train, num_workers=8)

    # scaler
    if cfg.FP16:
        scaler = GradScaler()
    else:
        scaler = None

    # model to gpu
    model = model.cuda()

    # build optimizer
    global_steps = 0
    optimizer = build_optimizer(model, cfg)
    lr_steps = cfg.TRAIN.SOLVER.STEP_LRS
    lr_steps = {x[0]: x[1] for x in lr_steps}
    base_lr = cfg.TRAIN.SOLVER.BASE_LR
    all_epochs = cfg.TRAIN.EPOCHS
    for epoch_idx in range(all_epochs):
        lr = lr_steps.get(epoch_idx, base_lr)
        print('Start epoch {0}, lr {1}'.format(epoch_idx, lr))
        # update learning rate
        for g in optimizer.param_groups:
            g['lr'] = lr
        loss_records, global_steps = train_epoch(cfg, model, dataloader, optimizer, tb_writer, global_steps, scaler)

        save_state = {
            'global_steps': global_steps,
            "epoch": epoch_idx + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            'loss_records': loss_records
        }
        backup_fpath = os.path.join(checkpoints_path, "model_bak_%06d.pt" % (epoch_idx,))
        torch.save(save_state, backup_fpath)
