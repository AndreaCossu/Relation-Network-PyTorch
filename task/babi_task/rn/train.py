import torch
import wandb
from torch.utils.data import DataLoader
from src.utils import save_models, saving_path_rn, get_answer, names_models
from src.utils import  BabiDataset, batchify, get_answer_separately
from collections import defaultdict


def train(train_stories, validation_stories, epochs, lstm, rn, criterion, optimizer, no_save, device, result_folder, batch_size, dict_size):

    train_babi_dataset = BabiDataset(train_stories)
    best_acc = 0.
    val_accuracies = []
    val_losses = []
    avg_train_accuracies = []
    avg_train_losses = []

    for epoch in range(1,epochs+1):

        train_accuracies = []
        train_losses = []

        train_dataset = DataLoader(train_babi_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: batchify(data_batch=b, dict_size=dict_size), drop_last=True)

        rn.train()
        lstm.train()

        for batch_id, (question_batch,answer_batch,facts_batch,_,_) in enumerate(train_dataset):
            if (batch_id+1) % 5000 == 0:
                print("Batch ", batch_id, "/", len(train_dataset), " - epoch ", epoch, ".")

            question_batch,answer_batch,facts_batch = question_batch.to(device), \
                                                            answer_batch.to(device), \
                                                            facts_batch.to(device)



            lstm.zero_grad()
            rn.zero_grad()

            h_q = lstm.reset_hidden_state_query()
            h_f = lstm.reset_hidden_state_fact(facts_batch.size(0))

            question_emb, h_q = lstm.process_query(question_batch, h_q)

            facts_emb, h_f = lstm.process_facts(facts_batch, h_f)

            rr = rn(facts_emb, question_emb)

            loss = criterion(rr, answer_batch)

            loss.backward()
            optimizer.step()


            correct, _ = get_answer(rr, answer_batch)

            train_accuracies.append(correct)
            train_losses.append(loss.item())

        avg_train_losses.append(sum(train_losses)/len(train_losses))
        avg_train_accuracies.append(sum(train_accuracies)/len(train_accuracies))

        val_loss, val_accuracy = test(validation_stories,lstm,rn,criterion, device, batch_size, dict_size)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        if not no_save:
            if val_accuracies[-1] > best_acc:
                save_models([(lstm, names_models[0]), (rn, names_models[1])], result_folder, saving_path_rn)
                best_acc = val_accuracies[-1]

        print("Train loss: ", avg_train_losses[-1], ". Validation loss: ", val_losses[-1])
        print("Train accuracy: ", avg_train_accuracies[-1], ". Validation accuracy: ", val_accuracies[-1])
        print()
        train_losses =  []
        train_accuracies = []

        wandb.log({
          'epoch': epoch,
          'train_loss': avg_train_losses[-1],
          'train_accuracy': avg_train_accuracies[-1],
          'val_loss': val_loss,
          'val_accuracy': val_accuracy
        })

    return avg_train_losses, avg_train_accuracies, val_losses, val_accuracies


def test(stories, lstm, rn, criterion, device, batch_size, dict_size):

    with torch.no_grad():

        test_loss = 0.
        test_accuracy = 0.

        rn.eval()
        lstm.eval()

        test_babi_dataset = BabiDataset(stories)
        test_dataset = DataLoader(test_babi_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: batchify(data_batch=b, dict_size=dict_size), drop_last=True)


        for batch_id, (question_batch,answer_batch,facts_batch,_,_) in enumerate(test_dataset):
            if (batch_id+1) % 1000 == 0:
                print("Test batch: ", batch_id, "/", len(test_dataset))

            question_batch,answer_batch,facts_batch = question_batch.to(device), \
                                                            answer_batch.to(device), \
                                                            facts_batch.to(device)


            h_q = lstm.reset_hidden_state_query()
            h_f = lstm.reset_hidden_state_fact(facts_batch.size(0))

            question_emb, h_q = lstm.process_query(question_batch, h_q)

            facts_emb, h_f = lstm.process_facts(facts_batch, h_f)

            rr = rn(facts_emb, question_emb)

            loss = criterion(rr, answer_batch)


            correct, _ = get_answer(rr, answer_batch)

            test_accuracy += correct
            test_loss += loss.item()

        return test_loss / float(len(test_dataset)), test_accuracy / float(len(test_dataset))


def test_separately(stories, lstm, rn, device, batch_size, dict_size):


    with torch.no_grad():

        accuracies = defaultdict(list)

        rn.eval()
        lstm.eval()

        test_babi_dataset = BabiDataset(stories)
        test_dataset = DataLoader(test_babi_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: batchify(data_batch=b, dict_size=dict_size), drop_last=True)


        for batch_id, (question_batch,answer_batch,facts_batch,task_label,_) in enumerate(test_dataset):
            if batch_id % 1000 == 0:
                print("Batch within test: ", batch_id, "/", len(test_dataset))

            question_batch,answer_batch,facts_batch, task_label = question_batch.to(device), \
                                                            answer_batch.to(device), \
                                                            facts_batch.to(device), \
                                                            task_label.tolist()


            h_q = lstm.reset_hidden_state_query()
            h_f = lstm.reset_hidden_state_fact(facts_batch.size(0))

            question_emb, h_q = lstm.process_query(question_batch, h_q)

            facts_emb, h_f = lstm.process_facts(facts_batch, h_f)

            rr = rn(facts_emb, question_emb)

            corrects = get_answer_separately(rr, answer_batch)

            for el, correct in zip(task_label, corrects):
                accuracies[el].append(1.) if correct else accuracies[el].append(0.)

        f = lambda x: sum(x) / float(len(x)) # get mean over each list values of dictionary
        avg_test_acc = {k: f(v) for k,v in accuracies.items()}

        return avg_test_acc
