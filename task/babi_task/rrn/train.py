import torch
import wandb
from torch.utils.data import DataLoader
from src.utils import save_models, saving_path_rrn, get_answer, names_models
from src.utils import  BabiDataset, batchify, get_answer_separately
from collections import defaultdict


REASONING_STEPS = 3
def train(train_stories, validation_stories, epochs, lstm, rrn, criterion, optimizer, batch_size, no_save, device, result_folder):

    train_babi_dataset = BabiDataset(train_stories)
    best_val = 1000.
    val_accuracies = []
    val_losses = []
    avg_train_accuracies = []
    avg_train_losses = []


    for epoch in range(1,epochs+1):

        train_accuracies = []
        train_losses = []

        train_dataset = DataLoader(train_babi_dataset, batch_size=batch_size, shuffle=True, collate_fn=batchify, drop_last=True)

        rrn.train()
        lstm.train()

        for batch_id, (question_batch,answer_batch,facts_batch,_,_) in enumerate(train_dataset):
            if (batch_id+1) % 5000 == 0:
                print("Batch ", batch_id, "/", len(train_dataset), " - epoch ", epoch, ".")

            question_batch,answer_batch,facts_batch = question_batch.to(device), \
                                                            answer_batch.to(device), \
                                                            facts_batch.to(device)



            lstm.zero_grad()
            rrn.zero_grad()

            h_q = lstm.reset_hidden_state_query()
            h_f = lstm.reset_hidden_state_fact(facts_batch.size(0))

            question_emb, h_q = lstm.process_query(question_batch, h_q)

            facts_emb, h_f = lstm.process_facts(facts_batch, h_f)

            input_mlp = torch.cat( (facts_emb, question_emb), dim=2)
            final_input = rrn.process_input(input_mlp)

            correct_rr = 0.
            loss_rr = 0.
            loss = 0.
            for reasoning_step in range(REASONING_STEPS):

                h = rrn.reset_g(batch_id)
                rr, hidden, h = rrn(final_input, facts_emb , h, question_emb)

                loss += criterion(rr, answer_batch)


                with torch.no_grad():
                    correct, _ = get_answer(rr, answer_batch)
                    correct_rr += correct
                    loss_rr += loss.item()

            loss.backward()
            optimizer.step()

            train_accuracies.append(correct_rr / float(REASONING_STEPS))
            train_losses.append(loss_rr / float(REASONING_STEPS))

        avg_train_losses.append(sum(train_losses)/len(train_losses))
        avg_train_accuracies.append(sum(train_accuracies)/len(train_accuracies))

        '''
        val_loss, val_accuracy = test(validation_stories,lstm,rrn,criterion, device, batch_size)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        '''
        val_loss, val_accuracy = 1., 0.3

        if not no_save:
            if val_losses[-1] < best_val:
                save_models([(lstm, names_models[0]), (rrn, names_models[1])], result_folder, saving_path_rrn)
                best_val = val_losses[-1]

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


def test(stories, lstm, rrn, criterion, device, batch_size):

    lstm.eval()
    rrn.eval()

    with torch.no_grad():

        pass

def test_separately(stories, lstm, rrn, device, batch_size):
    lstm.eval()
    rrn.eval()

    with torch.no_grad():

        pass
