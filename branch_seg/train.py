import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from blueberry_seg.runtime import ROOT, device, init_runtime, project_path
from .data import BranchDataset, validate_manifest
from .metrics import confusion, scores, segmentation_loss
from .model import build_model, train_mode, resolve_device


@torch.inference_mode()
def validate(model,loader,selected_device):
    model.eval()
    counts = np.zeros(4,dtype=np.int64)
    losses = []
    for images,targets,_ in loader:
        images,targets = images.to(selected_device),targets.to(selected_device)
        logits = model(images)['out']
        losses.append(float(segmentation_loss(logits,targets).cpu()))
        predictions = (logits.softmax(1)[:,1]>=0.5).cpu().numpy()
        for prediction,target in zip(predictions,targets.cpu().numpy()):
            counts += confusion(prediction,target)
    return scores(counts),float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description='Train two-class branch semantic segmentation')
    parser.add_argument('--config',default='configs/branch_cpu.yaml')
    parser.add_argument('--device')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--output')
    parser.add_argument('--resume',help='Restore model, optimizer, scheduler, random states and epoch')
    args = parser.parse_args()
    config = yaml.safe_load(project_path(args.config).read_text(encoding='utf-8'))
    for key in ('device','epochs','output'):
        if getattr(args,key) is not None:
            config[key] = getattr(args,key)
    init_runtime(config['threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    selected_device = resolve_device(config['device'])
    output = project_path(config['output'])
    if output.exists() and not args.resume:
        raise RuntimeError('Output exists; use a new output directory or explicitly resume')
    output.mkdir(parents=True,exist_ok=True)
    (output/'weights').mkdir(exist_ok=True)
    directory = project_path(config['data'])
    records,counts = validate_manifest(directory,config['allow_assisted_labels'])
    train_set = BranchDataset(directory,records,'train',config['imgsz'],augment=True)
    val_set = BranchDataset(directory,records,'val',config['imgsz'])
    train_loader = DataLoader(train_set,batch_size=config['batch'],shuffle=True,num_workers=config['workers'])
    val_loader = DataLoader(val_set,batch_size=config['batch'],shuffle=False,num_workers=config['workers'])
    model = build_model(project_path(config['pretrained'])).to(selected_device)
    optimizer = torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':config['backbone_lr']},
                                  {'params':model.classifier.parameters(),'lr':config['head_lr']}],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=1e-6)
    start_epoch,best,stale,elapsed_previous = 0,-1.0,0,0.0
    if args.resume:
        checkpoint = torch.load(project_path(args.resume),map_location='cpu',weights_only=True)
        if checkpoint['config']!=config:
            raise ValueError('Resume config differs from checkpoint; do not change epochs/schedule silently')
        if checkpoint['manifest_sha256'] != __import__('hashlib').sha256((directory/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Resume dataset/review changed')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):
                    state[key]=value.to(selected_device)
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch,best,stale = checkpoint['epoch'],checkpoint['best_dice'],checkpoint['stale']
        elapsed_previous = checkpoint['elapsed_seconds']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        saved_np = checkpoint['numpy_rng_state']
        np.random.set_state((saved_np[0],np.asarray(saved_np[1],dtype=np.uint32),saved_np[2],saved_np[3],saved_np[4]))
    started = time.monotonic()
    manifest_sha = __import__('hashlib').sha256((directory/'manifest.json').read_bytes()).hexdigest()
    status = {'status':'training','config':config,'dataset_counts':counts,
              'human_reviewed_labels':sum(bool(r.get('human_reviewed')) for r in records),
              'manifest_sha256':manifest_sha,'note':'Metrics are measured only on labeled pixels of AI-assisted masks, not expert ground truth.'}
    (output/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'CPU/GPU device={selected_device}; train={counts["train"]} val={counts["val"]} test={counts["test"]}; imgsz={config["imgsz"]}',flush=True)
    columns = ['epoch','seconds','train_loss','val_loss','val_iou','val_dice','val_precision','val_recall','backbone_lr','head_lr']
    with (output/'results.csv').open('a' if args.resume else 'w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        if not args.resume:
            writer.writeheader()
        for epoch in range(start_epoch,config['epochs']):
            train_mode(model,epoch<config['freeze_backbone_epochs'])
            losses = []
            for images,targets,_ in train_loader:
                images,targets = images.to(selected_device),targets.to(selected_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)['out']
                loss = segmentation_loss(logits,targets,config['positive_weight'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics,val_loss = validate(model,val_loader,selected_device)
            scheduler.step()
            elapsed = elapsed_previous+time.monotonic()-started
            row = {'epoch':epoch+1,'seconds':elapsed,'train_loss':float(np.mean(losses)),
                   'val_loss':val_loss,**{f'val_{k}':metrics[k] for k in ('iou','dice','precision','recall')},
                   'backbone_lr':optimizer.param_groups[0]['lr'],'head_lr':optimizer.param_groups[1]['lr']}
            writer.writerow(row)
            stream.flush()
            improved = metrics['dice']>best
            if improved:
                best,stale = metrics['dice'],0
            else:
                stale += 1
            np_state = np.random.get_state()
            checkpoint = {'architecture':'deeplabv3_mobilenet_v3_large_branch',
                          'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                          'epoch':epoch+1,'best_dice':best,'stale':stale,'config':config,
                          'manifest_sha256':manifest_sha,'elapsed_seconds':elapsed,
                          'torch_rng_state':torch.get_rng_state(),'python_rng_state':random.getstate(),
                          'numpy_rng_state':[np_state[0],np_state[1].tolist(),int(np_state[2]),int(np_state[3]),float(np_state[4])],
                          'metrics':metrics, 'human_reviewed_labels':status['human_reviewed_labels']}
            destination = output/'weights/last.pt'
            temporary = destination.with_suffix('.tmp')
            torch.save(checkpoint,temporary)
            temporary.replace(destination)
            if improved:
                torch.save(checkpoint,output/'weights/best.pt')
            status.update(completed_epochs=epoch+1,elapsed_seconds=elapsed,last_metrics=metrics,best_val_dice=best)
            (output/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'Epoch {epoch+1}/{config["epochs"]} | loss {row["train_loss"]:.4f} | val IoU {metrics["iou"]:.3f} Dice {metrics["dice"]:.3f} | {elapsed/60:.1f} min',flush=True)
            if config['patience'] and stale>=config['patience']:
                print(f'Early stop: {stale} epochs without validation Dice improvement',flush=True)
                break
    status['status']='training_complete'
    (output/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
