import torch
from torch import optim, nn
from torch.autograd import Variable

from logic.Model import Model
from preprocessing.preprocessing import tokenize_text, find_longest_sentece, create_batches


def train_model_one_epoch(epoch, model, optimizer, train_batches, valid_batches, criterion, best_valid_acc):
    model.train()
    total_loss, total_tag, cnt = 0.0, 0, 0
    check_valid, log_results = 1100, 500
    correct, total = 0, 0
    train_w, train_labels = train_batches
    valid_w, valid_labels = valid_batches

    for w, labels in zip(train_w, train_labels):
        cnt += 1
        model.zero_grad()
        batch_size = labels.shape[0]
        print(batch_size)
        answer = model(w, 'train')
        answer = Variable(answer, requires_grad=True)
        print(answer.shape)
        print(labels)
        correct += (torch.max(answer, 1)[1].view(labels.shape[0]) == torch.LongTensor(labels)).sum().item()
        total += len(w)
        train_acc = 100. * correct / total
        print(model.parameters())
        print("REQUIRES_GRAD ", answer.requires_grad, " ", labels.requires_grad)
        loss = criterion(answer, labels)
        loss.backward()
        optimizer.step()

        if cnt % check_valid == 0:
            model.eval()
            valid_correct, valid_total, valid_loss = 0, 0, 0
            with torch.no_grad():
                for valid_w, valid_labels in zip(valid_w, valid_labels):
                    answer = model(valid_w, 'valid')
                    valid_correct += (
                        torch.max(answer, 1)[1].view(len(valid_labels)) == torch.LongTensor(valid_labels)).sum().item()
                    valid_loss = criterion(answer, valid_labels)

            valid_acc = 100. * valid_correct / len(batch_size)

            print(
                "EPOCH {}, batch_id {}\nLoss at training {} and accuracy{}\nLoss at validation {} and valid accuracy {}".format(
                    epoch, cnt, loss.item(), train_acc, valid_loss.item(), valid_acc))

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                path, save_classify_layer
                print("NEW RECORD ACHIEVED {}".format(valid_acc))

        elif cnt % log_results == 0:
            print("EPOCH {}, batch_id {}\nLoss at training {} and accuracy{}".format(
                epoch, cnt, loss.item(), train_acc))
    return best_valid_acc


def train(word2embed, word2id, id2word, seq2words, labels, valid_data):
    valid2id, id2valid, validseq2words, valid_labels = tokenize_text(valid_data)
    epochs, ntargets = 10, 3
    best_valid_acc = 0
    max_len_sent = find_longest_sentece(seq2words)

    train_batches = create_batches(seq2words, labels, 32, word2id, max_len_sent)

    valid_batches = create_batches(validseq2words, valid_labels, 32, valid2id, max_len_sent)

    model = Model(word2embed, id2word, id2valid, ntargets, max_len_sent)

    need_grad = lambda x: x.requires_grad
    optimizer = optim.SGD(filter(need_grad, model.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        best_valid_acc = train_model_one_epoch(epoch, model, optimizer, train_batches, valid_batches, criterion,
                                               best_valid_acc)
        print('best_valid_acc: ' + str(best_valid_acc))
