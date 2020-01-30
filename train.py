import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import time
from model import Transformer, LabelSmoothedCE
from dataloader import SequenceLoader
from utils import *

# Data parameters
data_folder = '/media/ssd/transformer data'  # folder with data files

# Model parameters
d_model = 512  # size of vectors throughout the transformer model
n_heads = 8  # number of heads in the multi-head attention
d_queries = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
d_values = 64  # size of value vectors in the multi-head attention
d_inner = 2048  # an intermediate size in the position-wise FC
n_layers = 6  # number of layers in the Encoder and Decoder
dropout = 0.1  # dropout probability
positional_encoding = get_positional_encoding(d_model=d_model,
                                              max_length=160)  # positional encodings up to the maximum possible pad-length

# Learning parameters
checkpoint = 'transformer_checkpoint.pth.tar'  # path to model checkpoint, None if none
tokens_in_batch = 2000  # batch size in target language tokens
batches_per_step = 25000 // tokens_in_batch  # perform a training step, i.e. update parameters, once every so many batches
print_frequency = 20  # print status once every so many steps
n_steps = 100000  # number of training steps
warmup_steps = 8000  # number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official transformer repo.
step = 1  # the step number, start from 1 to prevent math error in the next line
lr = get_lr(step=step, d_model=d_model,
            warmup_steps=warmup_steps)  # see utils.py for learning rate schedule; twice the schedule in the paper, as in the official transformer repo.
start_epoch = 0  # start at this epoch
betas = (0.9, 0.98)  # beta coefficients in the Adam optimizer
epsilon = 1e-9  # epsilon term in the Adam optimizer
label_smoothing = 0.1  # label smoothing co-efficient in the Cross Entropy loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU isn't really practical here
cudnn.benchmark = False  # since input tensor size is variable


def main():
    """
    Training and validation.
    """
    global checkpoint, step, start_epoch, epoch, epochs

    # Initialize data-loaders
    train_loader = SequenceLoader(data_folder="/media/ssd/transformer data",
                                  source_suffix="en",
                                  target_suffix="de",
                                  split="train",
                                  tokens_in_batch=tokens_in_batch)
    val_loader = SequenceLoader(data_folder="/media/ssd/transformer data",
                                source_suffix="en",
                                target_suffix="de",
                                split="val",
                                tokens_in_batch=tokens_in_batch)

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = Transformer(vocab_size=train_loader.bpe_model.vocab_size(),
                            positional_encoding=positional_encoding,
                            d_model=d_model,
                            n_heads=n_heads,
                            d_queries=d_queries,
                            d_values=d_values,
                            d_inner=d_inner,
                            n_layers=n_layers,
                            dropout=dropout)
        optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad],
                                     lr=lr,
                                     betas=betas,
                                     eps=epsilon)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Loss function
    criterion = LabelSmoothedCE(eps=label_smoothing)

    # Move to default device
    model = model.to(device)
    criterion = criterion.to(device)

    # Find total epochs to train
    epochs = (n_steps // (train_loader.n_batches // batches_per_step)) + 1

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Step
        step = epoch * train_loader.n_batches // batches_per_step

        # One epoch's training
        train_loader.create_batches()
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              step=step)

        # One epoch's validation
        val_loader.create_batches()
        validate(val_loader=val_loader,
                 model=model,
                 criterion=criterion)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch, step):
    """
    One epoch's training.

    :param train_loader: loader for training data
    :param model: model
    :param criterion: label-smoothed cross-entropy loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    # Track some metrics
    data_time = AverageMeter()  # data loading time
    step_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss

    # Starting time
    start_data_time = time.time()
    start_step_time = time.time()

    # Batches
    for i, (source_sequences, target_sequences, source_sequence_lengths, target_sequence_lengths) in enumerate(
            train_loader):

        # Move to default device
        source_sequences = source_sequences.to(device)  # (N, max_source_sequence_pad_length_this_batch)
        target_sequences = target_sequences.to(device)  # (N, max_target_sequence_pad_length_this_batch)
        source_sequence_lengths = source_sequence_lengths.to(device)  # (N)
        target_sequence_lengths = target_sequence_lengths.to(device)  # (N)

        # Time taken to load data
        data_time.update(time.time() - start_data_time)

        # Forward prop.
        predicted_sequences = model(source_sequences, target_sequences, source_sequence_lengths,
                                    target_sequence_lengths)  # (N, max_target_sequence_pad_length_this_batch, vocab_size)

        # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
        # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
        # Therefore, pads start after (length - 1) positions
        loss = criterion(inputs=predicted_sequences,
                         targets=target_sequences[:, 1:],
                         lengths=target_sequence_lengths - 1)  # scalar

        # Backward prop.
        (loss / batches_per_step).backward()

        # Keep track of losses
        losses.update(loss.item(), (target_sequence_lengths - 1).sum().item())

        # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
        if (i + 1) % batches_per_step == 0:
            optimizer.step()
            optimizer.zero_grad()

            # This step is now complete
            step += 1

            # Update learning rate after each step
            change_lr(optimizer, new_lr=get_lr(step=step, d_model=d_model, warmup_steps=warmup_steps))

            # Time taken for this training step
            step_time.update(time.time() - start_step_time)

            # Print status
            if step % print_frequency == 0:
                print('Epoch {0}/{1}-----'
                      'Batch {2}/{3}-----'
                      'Step {4}/{5}-----'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----'
                      'Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                      'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(epoch + 1, epochs,
                                                                        i + 1, train_loader.n_batches,
                                                                        step, n_steps,
                                                                        step_time=step_time,
                                                                        data_time=data_time,
                                                                        losses=losses))

            # Reset step time
            start_step_time = time.time()

            # If this is the last one or two epochs, save checkpoints at regular intervals for averaging
            if epoch in [epochs - 1, epochs - 2] and step % 1500 == 0:  # 'epoch' is 0-indexed
                save_checkpoint(epoch, model, optimizer, prefix='step' + str(step) + "_")

        # Reset data time
        start_data_time = time.time()


def validate(val_loader, model, criterion):
    """
    One epoch's validation.

    :param val_loader: loader for validation data
    :param model: model
    :param criterion: label-smoothed cross-entropy loss
    """
    model.eval()  # eval mode disables dropout

    # Prohibit gradient computation explicitly
    with torch.no_grad():
        losses = AverageMeter()
        # Batches
        for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
                tqdm(val_loader, total=val_loader.n_batches)):
            source_sequence = source_sequence.to(device)  # (1, source_sequence_length)
            target_sequence = target_sequence.to(device)  # (1, target_sequence_length)
            source_sequence_length = source_sequence_length.to(device)  # (1)
            target_sequence_length = target_sequence_length.to(device)  # (1)

            # Forward prop.
            predicted_sequence = model(source_sequence, target_sequence, source_sequence_length,
                                       target_sequence_length)  # (1, target_sequence_length, vocab_size)

            # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            loss = criterion(inputs=predicted_sequence,
                             targets=target_sequence[:, 1:],
                             lengths=target_sequence_length - 1)  # scalar

            # Keep track of losses
            losses.update(loss.item(), (target_sequence_length - 1).sum().item())

        print("\nValidation loss: %.3f\n\n" % losses.avg)


if __name__ == '__main__':
    main()
