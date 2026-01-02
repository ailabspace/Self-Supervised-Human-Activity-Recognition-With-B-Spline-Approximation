import math
from torch.utils.tensorboard import SummaryWriter
from utils import *
from tqdm import trange

def pretrain(train_loader, model, optimizer, epochs, segments, pose_mask_ratio, warmup_epochs, scheduler_warmup, scheduler_cos, amp = False, work_dir = "./output_dir/"):
    writer = SummaryWriter()
    least_loss = float('inf')
    best_epoch = 0
    model.train()
    start_time = time.time()
    for epoch in trange(epochs, dynamic_ncols=True, position=0, leave=True):
        for batch, (data, label, index) in enumerate(train_loader):
            with torch.no_grad():
                data = data.float().cuda()
            with torch.amp.autocast('cuda', enabled=amp):
                loss = model(data, segments, pose_mask_ratio)

            writer.add_scalar(f"Reconstruction Loss Segments All", loss, epoch)
            optimizer.zero_grad()
            # loss.backward(retain_graph = retain)
            loss.backward()
            optimizer.step()

        if least_loss > loss:
            least_loss = loss
            best_epoch = epoch
            print_log(f"New Best Model Saved", work_dir)
            state_dict = model.state_dict()

        # Learning Rate Schedulers
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        elif epoch == warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler_warmup.get_last_lr()[0]
            scheduler_cos.base_lrs = [group['lr'] for group in optimizer.param_groups]
            scheduler_cos.step()
        else:
            scheduler_cos.step()

        print_log(f"Epoch {epoch + 1} Total Loss: {loss}", work_dir)
        if math.isnan(loss):
            print_log(f"nan detected at {epoch + 1} - Breaking Training Loop", work_dir)
            break


    print_log(f"Final weights saved..", work_dir)
    save_weights(model.state_dict(), work_dir, "final_weights")
    print_log(f"Best Loss: {least_loss} at epoch {best_epoch + 1}", work_dir)
    save_weights(state_dict, work_dir)
    print_log(f"Time taken for pre-training: {compute_duration(start_time, time.time())}", work_dir)
    writer.flush()
    writer.close()